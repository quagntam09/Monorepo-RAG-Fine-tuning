"""
Training loop cho DistilBERT Extractive QA.

Xử lý:
1. Setup device (GPU/CPU)
2. Load model, optimizer, scheduler
3. Training loop với validation
4. Checkpoint management
"""

from __future__ import annotations

import dataclasses
import json
import logging
import random
from pathlib import Path
from typing import Any

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .data_loader import build_qa_datasets, load_raw_datasets
from .dataset import prepare_eval_features
from .vietnamese_utils import VietnameseTextProcessor, align_segmentation_offset
from .modeling import build_model
from .config import TrainingConfig
from .optimizer import build_optimizer, build_scheduler
from .metrics import compute_metrics

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed cho reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_dataloader(dataset, config: TrainingConfig, device: torch.device, shuffle: bool) -> DataLoader:
    """Create a DataLoader with CUDA-friendly options when workers are enabled."""
    num_workers = max(0, config.num_workers)
    loader_kwargs = {
        "batch_size": config.batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": config.pin_memory and device.type == "cuda",
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = config.persistent_workers
        loader_kwargs["prefetch_factor"] = max(1, config.prefetch_factor)

    return DataLoader(dataset, **loader_kwargs)


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move tensor batch values to device without blocking pinned-memory copies."""
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _is_better_metric(
    current_value: float,
    best_value: float,
    *,
    greater_is_better: bool = True,
) -> bool:
    return current_value > best_value if greater_is_better else current_value < best_value


def _safe_load_state_dict(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resolve_tokenizer_source(config: TrainingConfig) -> str:
    if not config.init_checkpoint_dir:
        return config.model_name
    checkpoint_dir = Path(config.init_checkpoint_dir)
    required = ("tokenizer.json", "tokenizer_config.json")
    if all((checkpoint_dir / name).exists() for name in required):
        return str(checkpoint_dir)
    return config.model_name


def _load_initial_checkpoint_if_configured(model, config: TrainingConfig) -> str | None:
    if not config.init_checkpoint_dir:
        return None

    checkpoint_dir = Path(config.init_checkpoint_dir)
    checkpoint_file = checkpoint_dir / "pytorch_model.bin"
    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"init_checkpoint_dir được cấu hình nhưng thiếu file trọng số: {checkpoint_file}"
        )

    state_dict = _safe_load_state_dict(checkpoint_file)
    model.load_state_dict(state_dict, strict=True)
    return str(checkpoint_dir)


def _span_exact_match(outputs, batch: dict) -> tuple[int, int]:
    """Return count of QA spans where both start and end positions are correct."""
    if "start_positions" not in batch or "end_positions" not in batch:
        return 0, 0

    start_pred = outputs.start_logits.argmax(dim=-1)
    end_pred = outputs.end_logits.argmax(dim=-1)
    matches = (
        (start_pred == batch["start_positions"])
        & (end_pred == batch["end_positions"])
    )
    return int(matches.sum().item()), int(batch["start_positions"].size(dim=0))


def _write_training_history(output_dir: Path, history: list[dict[str, Any]]) -> Path:
    history_path = output_dir / "training_history.json"
    history_path.write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return history_path


def _plot_training_history(output_dir: Path, history: list[dict[str, Any]]) -> Path | None:
    """Save loss/metric curves. Plotting is optional so training still finishes without matplotlib."""
    if not history:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Không thể vẽ biểu đồ vì thiếu matplotlib. Cài bằng: pip install matplotlib")
        return None

    epochs = [row["epoch"] for row in history]
    plot_path = output_dir / "training_curves.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    loss_ax, metric_ax = axes

    loss_ax.plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train loss")
    valid_losses = [row.get("valid_loss") for row in history]
    if any(value is not None for value in valid_losses):
        loss_ax.plot(epochs, valid_losses, marker="o", label="Validation loss")
    loss_ax.set_title("Loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.grid(True, alpha=0.3)
    loss_ax.legend()

    metric_series = [
        ("train_span_exact_match", "Train accuracy"),
        ("valid_span_exact_match", "Validation accuracy"),
        ("exact_match", "Validation EM"),
        ("f1", "Validation F1"),
    ]
    for key, label in metric_series:
        values = []
        for row in history:
            value = row.get(key)
            if value is not None and key in {"exact_match", "f1"} and float(value) > 1.0:
                value = float(value) / 100.0
            values.append(value)
        if any(value is not None for value in values):
            metric_ax.plot(epochs, values, marker="o", label=label)
    metric_ax.set_title("Accuracy / EM / F1")
    metric_ax.set_xlabel("Epoch")
    metric_ax.set_ylabel("Score")
    metric_ax.set_ylim(0.0, 1.0)
    metric_ax.grid(True, alpha=0.3)
    metric_ax.legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def _build_validation_eval_inputs(
    raw_validation_dataset,
    tokenizer,
    config: TrainingConfig,
) -> dict:
    """
    Build bộ dữ liệu evaluation để tính EM/F1 từ logits.

    Lý do tách riêng:
    - validation loader hiện tại dùng train features (có labels) nên không giữ offset_mapping.
    - để decode span ra text, ta cần offset_mapping + context + references theo format SQuAD.
    """
    eval_features = raw_validation_dataset.map(
        lambda examples: prepare_eval_features(
            examples=examples,
            tokenizer=tokenizer,
            question_column=config.question_column,
            context_column=config.context_column,
            max_length=config.max_length,
            doc_stride=config.doc_stride,
            padding=config.padding,
            use_vietnamese_segmentation=config.use_vietnamese_segmentation,
            segmentation_tool=config.segmentation_tool,
        ),
        batched=True,
        remove_columns=raw_validation_dataset.column_names,
        desc="Preparing validation features for EM/F1",
    )

    vi_processor = None
    if config.use_vietnamese_segmentation:
        try:
            vi_processor = VietnameseTextProcessor(segmentation_tool=config.segmentation_tool)
        except Exception as e:
            logger.warning("Không khởi tạo được VietnameseTextProcessor cho validation metrics: %s", e)

    metric_contexts_by_sample: dict[int, str] = {}
    references = []
    for sample_idx in range(len(raw_validation_dataset)):
        raw_sample = raw_validation_dataset[int(sample_idx)]
        raw_context = raw_sample[config.context_column]
        answers = raw_sample.get(config.answers_column, {"text": [], "answer_start": []})
        answer_texts = list(answers.get("text", [])) if isinstance(answers, dict) else []
        answer_starts = list(answers.get("answer_start", [])) if isinstance(answers, dict) else []

        # Nếu bật segmentation, cần đồng bộ context + answer offsets/text theo cùng không gian ký tự.
        metric_context = raw_context
        metric_answer_texts = answer_texts
        metric_answer_starts = answer_starts
        if vi_processor is not None:
            segmented_context = vi_processor.segment(raw_context)
            metric_context = segmented_context

            aligned_texts: list[str] = []
            aligned_starts: list[int] = []
            for answer_text, answer_start in zip(answer_texts, answer_starts):
                aligned_start, aligned_text = align_segmentation_offset(
                    raw_context=raw_context,
                    raw_answer_text=answer_text,
                    raw_answer_start=answer_start,
                    segmented_context=segmented_context,
                )
                if aligned_start is not None and aligned_text is not None:
                    aligned_texts.append(aligned_text)
                    aligned_starts.append(aligned_start)
            if aligned_texts:
                metric_answer_texts = aligned_texts
                metric_answer_starts = aligned_starts

        metric_contexts_by_sample[int(sample_idx)] = metric_context
        # evaluate.squad yêu cầu mỗi sample có ít nhất 1 ground-truth answer text.
        # Với unanswerable samples, dùng empty-string answer để tránh crash max([]).
        if len(metric_answer_texts) == 0:
            metric_answer_texts = [""]
            metric_answer_starts = [0]
        references.append(
            {
                "id": str(sample_idx),
                "answers": {
                    "text": metric_answer_texts,
                    "answer_start": metric_answer_starts,
                },
            }
        )

    # sample_id do prepare_eval_features trả về map mỗi feature/window về mẫu gốc.
    feature_sample_ids = [int(sample_idx) for sample_idx in eval_features["sample_id"]]

    return {
        "input_ids": eval_features["input_ids"],
        "attention_mask": eval_features["attention_mask"],
        "offset_mapping": eval_features["offset_mapping"],
        "contexts": [metric_contexts_by_sample[sample_idx] for sample_idx in feature_sample_ids],
        "example_ids": [str(sample_idx) for sample_idx in feature_sample_ids],
        "references": references,
    }


def _run_em_f1_validation(
    model,
    device: torch.device,
    validation_eval_inputs: dict,
    batch_size: int,
    use_amp: bool,
) -> dict:
    """Chạy forward trên validation features và tính EM/F1."""
    model.eval()

    all_start_logits: list[np.ndarray] = []
    all_end_logits: list[np.ndarray] = []

    input_ids_all = validation_eval_inputs["input_ids"]
    attention_masks_all = validation_eval_inputs["attention_mask"]
    total = len(input_ids_all)

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_input_ids = torch.tensor(input_ids_all[start:end], dtype=torch.long, device=device)
            batch_attention_mask = torch.tensor(attention_masks_all[start:end], dtype=torch.long, device=device)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                )

            all_start_logits.append(outputs.start_logits.detach().cpu().numpy())
            all_end_logits.append(outputs.end_logits.detach().cpu().numpy())

    if len(all_start_logits) == 0:
        return {"exact_match": 0.0, "f1": 0.0}

    start_logits = np.concatenate(all_start_logits, axis=0)
    end_logits = np.concatenate(all_end_logits, axis=0)

    eval_preds = {
        "start_logits": start_logits,
        "end_logits": end_logits,
        "offset_mapping": validation_eval_inputs["offset_mapping"],
        "contexts": validation_eval_inputs["contexts"],
        "example_ids": validation_eval_inputs["example_ids"],
        "references": validation_eval_inputs["references"],
        "max_answer_length": 30,
    }
    return compute_metrics(eval_preds)


def train(config: TrainingConfig) -> dict[str, float | str | None]:
    """
    Main training loop.

    Args:
        config: TrainingConfig object
    """

    # Setup
    set_seed(config.seed)

    # Validate config parameters
    if config.doc_stride >= config.max_length - 2:
        logger.warning(f"doc_stride ({config.doc_stride}) >= max_length - 2 ({config.max_length - 2}). Adjusting doc_stride to {config.max_length - 3}")
        config.doc_stride = config.max_length - 3

    # Check CUDA availability and setup device
    if config.force_cpu:
        device = torch.device("cpu")
        logger.info("Force CPU training enabled")
    elif torch.cuda.is_available():
        try:
            # Test CUDA with a small tensor
            test_tensor = torch.randn(1, device='cuda')
            del test_tensor
            torch.cuda.empty_cache()
            device = torch.device("cuda")
            logger.info("✓ CUDA is available and working")
        except Exception as e:
            logger.warning(f"CUDA test failed: {e}. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available, using CPU")

    use_amp = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    if device.type == "cuda" and config.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    logger.info(f"Device: {device}")
    logger.info(f"AMP: {use_amp}")
    logger.info(f"TF32: {device.type == 'cuda' and config.use_tf32}")

    # ── Data ────────────────────────────────────────────────────────────────
    logger.info("Loading datasets...")
    tokenizer_source = _resolve_tokenizer_source(config)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    datasets = build_qa_datasets(tokenizer=tokenizer, config=config)

    train_dataset = datasets.get("train")
    valid_dataset = datasets.get("validation")

    if train_dataset is None:
        raise ValueError("Dataset must contain a 'train' split for training.")

    train_loader = _build_dataloader(
        dataset=train_dataset,
        config=config,
        device=device,
        shuffle=True,
    )

    valid_loader = _build_dataloader(
        dataset=valid_dataset,
        config=config,
        device=device,
        shuffle=False,
    ) if valid_dataset else None

    logger.info(f"Train samples: {len(train_dataset)}")
    if valid_dataset:
        logger.info(f"Valid samples: {len(valid_dataset)}")

    raw_valid_dataset = None
    validation_eval_inputs = None
    if valid_dataset is not None:
        raw_datasets = load_raw_datasets(config=config)
        raw_valid_dataset = raw_datasets.get("validation")
        if raw_valid_dataset is not None and config.answers_column in raw_valid_dataset.column_names:
            validation_eval_inputs = _build_validation_eval_inputs(
                raw_validation_dataset=raw_valid_dataset,
                tokenizer=tokenizer,
                config=config,
            )
        else:
            logger.warning("Không thể build validation inputs cho EM/F1 vì thiếu split validation hoặc answers column.")

    # ── Model ───────────────────────────────────────────────────────────────
    logger.info("Loading model...")
    model = build_model(
        model_name=config.model_name,
        dropout=config.dropout,
        freeze_encoder=config.freeze_encoder,
    )
    init_checkpoint_used = _load_initial_checkpoint_if_configured(model=model, config=config)
    if init_checkpoint_used is not None:
        logger.info("Initialized model weights from checkpoint: %s", init_checkpoint_used)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # ── Optimizer & Scheduler ───────────────────────────────────────────────
    grad_accum_steps = max(1, config.gradient_accumulation_steps)
    total_steps = ((len(train_loader) + grad_accum_steps - 1) // grad_accum_steps) * config.epochs
    if total_steps <= 0:
        raise ValueError("Training dataloader is empty; check dataset paths and preprocessing.")

    optimizer = build_optimizer(model=model, config=config)
    scheduler = build_scheduler(
        optimizer=optimizer,
        num_training_steps=total_steps,
        config=config,
    )

    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {max(1, int(total_steps * config.warmup_ratio))}")

    # ── Training Loop ───────────────────────────────────────────────────────
    logger.info("Starting training...")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_config.json").write_text(
        json.dumps(dataclasses.asdict(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Select best checkpoint by configured metric.
    metric_for_best_model = (config.best_metric or "f1").strip()
    if metric_for_best_model not in {"exact_match", "f1"}:
        logger.warning(
            "Unsupported best_metric=%s. Falling back to 'f1'.",
            metric_for_best_model,
        )
        metric_for_best_model = "f1"
    greater_is_better = True
    best_metric_value = float("-inf") if greater_is_better else float("inf")
    best_path = output_dir / "best_model"
    global_step = 0
    history: list[dict[str, Any]] = []
    history_path = output_dir / "training_history.json"
    curves_path: Path | None = None

    for epoch in range(1, config.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{config.epochs}")

        # Training
        model.train()
        train_loss = 0.0
        train_exact = 0
        train_total = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            batch = _move_batch_to_device(batch=batch, device=device)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(**batch)
                if outputs.loss is None:
                    raise RuntimeError("Training batch did not produce a loss. Check start/end labels.")
                loss = outputs.loss / grad_accum_steps

            scaler.scale(loss).backward()
            train_loss += outputs.loss.item()
            exact_count, batch_total = _span_exact_match(outputs=outputs, batch=batch)
            train_exact += exact_count
            train_total += batch_total

            if step % grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config.max_grad_norm,
                )
                scale_before_step = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if not use_amp or scaler.get_scale() >= scale_before_step:
                    scheduler.step()
                    global_step += 1
                    if config.logging_steps and global_step % config.logging_steps == 0:
                        logger.info(
                            "step=%s/%s lr=%.3e loss=%.4f",
                            global_step,
                            total_steps,
                            scheduler.get_last_lr()[0],
                            outputs.loss.item(),
                        )
                optimizer.zero_grad(set_to_none=True)

        avg_train_loss = train_loss / len(train_loader)
        train_span_exact_match = train_exact / max(1, train_total)
        logger.info(f"Train Loss: {avg_train_loss:.4f}")
        logger.info(f"Train Span Accuracy: {train_span_exact_match:.4f}")

        epoch_record: dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "train_loss": float(avg_train_loss),
            "train_span_exact_match": float(train_span_exact_match),
            "valid_loss": None,
            "valid_span_exact_match": None,
            "exact_match": None,
            "f1": None,
        }

        # Validation
        if valid_loader:
            model.eval()
            valid_loss = 0.0
            valid_steps = 0
            valid_exact = 0
            valid_total = 0

            with torch.no_grad():
                for batch in valid_loader:
                    batch = _move_batch_to_device(batch=batch, device=device)

                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        outputs = model(**batch)
                    if outputs.loss is not None:
                        valid_loss += outputs.loss.item()
                        valid_steps += 1
                    exact_count, batch_total = _span_exact_match(outputs=outputs, batch=batch)
                    valid_exact += exact_count
                    valid_total += batch_total

            epoch_metrics = None
            if validation_eval_inputs is not None:
                epoch_metrics = _run_em_f1_validation(
                    model=model,
                    device=device,
                    validation_eval_inputs=validation_eval_inputs,
                    batch_size=config.batch_size,
                    use_amp=use_amp,
                )
                logger.info(
                    "Validation metrics - EM: %.4f | F1: %.4f",
                    epoch_metrics.get("exact_match", 0.0),
                    epoch_metrics.get("f1", 0.0),
                )
                epoch_record["exact_match"] = float(epoch_metrics.get("exact_match", 0.0))
                epoch_record["f1"] = float(epoch_metrics.get("f1", 0.0))

            if valid_steps > 0:
                avg_valid_loss = valid_loss / valid_steps
                valid_span_exact_match = valid_exact / max(1, valid_total)
                logger.info(f"Valid Loss: {avg_valid_loss:.4f}")
                logger.info(f"Valid Span Accuracy: {valid_span_exact_match:.4f}")
                epoch_record["valid_loss"] = float(avg_valid_loss)
                epoch_record["valid_span_exact_match"] = float(valid_span_exact_match)

                # Save best model by configured metric.
                if config.save_best_model and epoch_metrics is not None:
                    current_metric_value = float(epoch_metrics.get(metric_for_best_model, 0.0))
                    is_better = _is_better_metric(
                        current_value=current_metric_value,
                        best_value=best_metric_value,
                        greater_is_better=greater_is_better,
                    )
                    if is_better:
                        best_metric_value = current_metric_value
                        best_path.mkdir(exist_ok=True)
                        model.save_pretrained(best_path)
                        tokenizer.save_pretrained(best_path)
                        (best_path / "training_config.json").write_text(
                            json.dumps(dataclasses.asdict(config), indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        logger.info("✓ Best model saved by %s=%.4f", metric_for_best_model, current_metric_value)
            else:
                logger.info("Valid Loss: unavailable (validation dataset has no labels or loss was not computed). Skipping best-model selection.")

        history.append(epoch_record)
        history_path = _write_training_history(output_dir=output_dir, history=history)
        logger.info("Training history saved to %s", history_path)

        # Save checkpoint
        checkpoint_path = output_dir / f"checkpoint-epoch-{epoch}"
        checkpoint_path.mkdir(exist_ok=True)
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)

    # load_best_model_at_end=True: nạp lại best checkpoint vào model ở cuối train.
    if config.load_best_model and best_path.exists():
        checkpoint_file = best_path / "pytorch_model.bin"
        if checkpoint_file.exists():
            model.load_state_dict(_safe_load_state_dict(checkpoint_file))
            logger.info("Best model loaded at end from %s", best_path)

    curves_path = _plot_training_history(output_dir=output_dir, history=history)
    if curves_path is not None:
        logger.info("Training curves saved to %s", curves_path)

    logger.info("\nTraining completed!")
    return {
        "best_model_path": str(best_path),
        "best_metric_name": metric_for_best_model,
        "best_metric_value": float(best_metric_value),
        "output_dir": str(output_dir),
        "init_checkpoint_dir": init_checkpoint_used,
        "training_history_path": str(history_path),
        "training_curves_path": str(curves_path) if curves_path is not None else None,
    }

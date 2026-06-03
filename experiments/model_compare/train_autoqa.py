from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
import torch
from datasets import DatasetDict

try:
    from .common import (
        AutoQATrainConfig,
        build_dataloader,
        disable_broken_torchvision,
        maybe_freeze_encoder,
        move_batch_to_device,
        resolve_model_source,
        resolve_tokenizer_source,
        select_device,
        set_seed,
        setup_logging,
    )
except ImportError:
    SCRIPT_DIR = Path(__file__).resolve().parent
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from common import (  # type: ignore
        AutoQATrainConfig,
        build_dataloader,
        disable_broken_torchvision,
        maybe_freeze_encoder,
        move_batch_to_device,
        resolve_model_source,
        resolve_tokenizer_source,
        select_device,
        set_seed,
        setup_logging,
    )

disable_broken_torchvision()

from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from training.data_loader import load_raw_datasets
from training.dataset import prepare_eval_features, prepare_train_features
from training.metrics import compute_metrics
from training.optimizer import build_optimizer, build_scheduler
from training.vietnamese_utils import VietnameseTextProcessor, align_segmentation_offset

LOGGER = logging.getLogger(__name__)


def _build_qa_datasets(tokenizer, config: AutoQATrainConfig) -> tuple[DatasetDict, DatasetDict]:
    raw_datasets = load_raw_datasets(config=config)
    processed = DatasetDict()

    for split_name, dataset in raw_datasets.items():
        has_answers = config.answers_column in dataset.column_names
        has_labels = split_name in {"train", "validation"} and has_answers

        if has_labels:
            prepare_fn = prepare_train_features
            prepare_kwargs = {
                "answers_column": config.answers_column,
                "impossible_column": config.impossible_column,
            }
        else:
            prepare_fn = prepare_eval_features
            prepare_kwargs = {}

        tokenized = dataset.map(
            lambda examples: prepare_fn(
                examples=examples,
                tokenizer=tokenizer,
                question_column=config.question_column,
                context_column=config.context_column,
                max_length=config.max_length,
                doc_stride=config.doc_stride,
                padding=config.padding,
                use_vietnamese_segmentation=config.use_vietnamese_segmentation,
                segmentation_tool=config.segmentation_tool,
                **prepare_kwargs,
            ),
            batched=True,
            remove_columns=dataset.column_names,
            desc=f"Tokenizing {split_name}",
        )

        columns = ["input_ids", "attention_mask"]
        if "token_type_ids" in tokenized.column_names:
            columns.append("token_type_ids")

        if has_labels:
            columns.extend(["start_positions", "end_positions"])
        else:
            if "offset_mapping" in tokenized.column_names:
                columns.append("offset_mapping")
            if "sample_id" in tokenized.column_names:
                columns.append("sample_id")

        tokenized.set_format(type="torch", columns=columns)
        processed[split_name] = tokenized

    return raw_datasets, processed


def _build_validation_eval_inputs(raw_validation_dataset, tokenizer, config: AutoQATrainConfig) -> dict:
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
        except Exception as exc:
            LOGGER.warning("VietnameseTextProcessor init failed: %s", exc)

    contexts_by_sample: dict[int, str] = {}
    references = []

    for sample_idx in range(len(raw_validation_dataset)):
        sample = raw_validation_dataset[int(sample_idx)]
        raw_context = sample[config.context_column]
        answers = sample.get(config.answers_column, {"text": [], "answer_start": []})

        answer_texts = list(answers.get("text", [])) if isinstance(answers, dict) else []
        answer_starts = list(answers.get("answer_start", [])) if isinstance(answers, dict) else []

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

        if not metric_answer_texts:
            metric_answer_texts = [""]
            metric_answer_starts = [0]

        contexts_by_sample[int(sample_idx)] = metric_context
        references.append(
            {
                "id": str(sample_idx),
                "answers": {
                    "text": metric_answer_texts,
                    "answer_start": metric_answer_starts,
                },
            }
        )

    feature_sample_ids = [int(sid) for sid in eval_features["sample_id"]]

    payload = {
        "input_ids": eval_features["input_ids"],
        "attention_mask": eval_features["attention_mask"],
        "offset_mapping": eval_features["offset_mapping"],
        "contexts": [contexts_by_sample[sid] for sid in feature_sample_ids],
        "example_ids": [str(sid) for sid in feature_sample_ids],
        "references": references,
    }

    if "token_type_ids" in eval_features.column_names:
        payload["token_type_ids"] = eval_features["token_type_ids"]

    return payload


def _run_em_f1_validation(
    model,
    device: torch.device,
    validation_eval_inputs: dict,
    batch_size: int,
    use_amp: bool,
    max_answer_length: int,
) -> dict:
    model.eval()
    all_start_logits: list[np.ndarray] = []
    all_end_logits: list[np.ndarray] = []

    input_ids_all = validation_eval_inputs["input_ids"]
    attention_masks_all = validation_eval_inputs["attention_mask"]
    token_type_ids_all = validation_eval_inputs.get("token_type_ids")

    with torch.no_grad():
        total = len(input_ids_all)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)

            batch = {
                "input_ids": torch.tensor(input_ids_all[start:end], dtype=torch.long, device=device),
                "attention_mask": torch.tensor(attention_masks_all[start:end], dtype=torch.long, device=device),
            }
            if token_type_ids_all is not None:
                batch["token_type_ids"] = torch.tensor(
                    token_type_ids_all[start:end],
                    dtype=torch.long,
                    device=device,
                )

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(**batch)

            all_start_logits.append(outputs.start_logits.detach().cpu().numpy())
            all_end_logits.append(outputs.end_logits.detach().cpu().numpy())

    if not all_start_logits:
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
        "max_answer_length": max_answer_length,
    }
    return compute_metrics(eval_preds)


def train(config: AutoQATrainConfig) -> dict[str, float | str | None]:
    set_seed(config.seed)

    if config.doc_stride >= config.max_length - 2:
        config.doc_stride = config.max_length - 3

    device = select_device(force_cpu=config.force_cpu)
    use_amp = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if device.type == "cuda" and config.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(resolve_tokenizer_source(config), use_fast=True)
    raw_datasets, processed_datasets = _build_qa_datasets(tokenizer=tokenizer, config=config)

    train_dataset = processed_datasets.get("train")
    valid_dataset = processed_datasets.get("validation")

    if train_dataset is None:
        raise ValueError("Missing train split for training.")

    train_loader = build_dataloader(train_dataset, config=config, device=device, shuffle=True)
    valid_loader = (
        build_dataloader(valid_dataset, config=config, device=device, shuffle=False)
        if valid_dataset is not None
        else None
    )

    validation_eval_inputs = None
    raw_validation_dataset = raw_datasets.get("validation")
    if raw_validation_dataset is not None and config.answers_column in raw_validation_dataset.column_names:
        validation_eval_inputs = _build_validation_eval_inputs(
            raw_validation_dataset=raw_validation_dataset,
            tokenizer=tokenizer,
            config=config,
        )

    model = AutoModelForQuestionAnswering.from_pretrained(resolve_model_source(config))
    if config.freeze_encoder:
        frozen_params = maybe_freeze_encoder(model)
        LOGGER.info("Frozen encoder parameters: %s", f"{frozen_params:,}")
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info("Model: %s", config.model_name)
    LOGGER.info("Total parameters: %s", f"{total_params:,}")
    LOGGER.info("Trainable parameters: %s", f"{trainable_params:,}")

    grad_accum_steps = max(1, config.gradient_accumulation_steps)
    total_steps = ((len(train_loader) + grad_accum_steps - 1) // grad_accum_steps) * config.epochs
    if total_steps <= 0:
        raise ValueError("Empty training dataloader.")

    optimizer = build_optimizer(model=model, config=config)
    scheduler = build_scheduler(optimizer=optimizer, num_training_steps=total_steps, config=config)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_config.json").write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metric_for_best = (config.best_metric or "f1").strip()
    if metric_for_best not in {"exact_match", "f1"}:
        metric_for_best = "f1"

    best_metric_value = float("-inf")
    best_model_path = output_dir / "best_model"
    global_step = 0

    LOGGER.info("Device: %s | AMP: %s", device, use_amp)
    LOGGER.info("Start training for %s epochs", config.epochs)

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device=device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss
                if loss is None:
                    raise RuntimeError("Training step produced no loss.")
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
            total_train_loss += float(outputs.loss.item())

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
                        LOGGER.info(
                            "step=%s/%s lr=%.3e loss=%.4f",
                            global_step,
                            total_steps,
                            scheduler.get_last_lr()[0],
                            float(outputs.loss.item()),
                        )
                optimizer.zero_grad(set_to_none=True)

        avg_train_loss = total_train_loss / max(1, len(train_loader))
        LOGGER.info("Epoch %s train loss: %.4f", epoch, avg_train_loss)

        epoch_metrics = None
        if valid_loader is not None:
            model.eval()
            total_valid_loss = 0.0
            valid_steps = 0
            with torch.no_grad():
                for batch in valid_loader:
                    batch = move_batch_to_device(batch, device=device)
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        outputs = model(**batch)
                    if outputs.loss is not None:
                        total_valid_loss += float(outputs.loss.item())
                        valid_steps += 1

            if valid_steps > 0:
                LOGGER.info("Epoch %s valid loss: %.4f", epoch, total_valid_loss / valid_steps)

            if validation_eval_inputs is not None:
                epoch_metrics = _run_em_f1_validation(
                    model=model,
                    device=device,
                    validation_eval_inputs=validation_eval_inputs,
                    batch_size=config.batch_size,
                    use_amp=use_amp,
                    max_answer_length=config.max_answer_length,
                )
                LOGGER.info(
                    "Epoch %s validation EM/F1: %.4f / %.4f",
                    epoch,
                    float(epoch_metrics.get("exact_match", 0.0)),
                    float(epoch_metrics.get("f1", 0.0)),
                )

        if config.save_best_model and epoch_metrics is not None:
            current = float(epoch_metrics.get(metric_for_best, 0.0))
            if current > best_metric_value:
                best_metric_value = current
                best_model_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_model_path)
                tokenizer.save_pretrained(best_model_path)
                (best_model_path / "training_config.json").write_text(
                    json.dumps(config.to_dict(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                LOGGER.info("Saved new best model (%s=%.4f)", metric_for_best, current)

        checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

    if config.load_best_model and best_model_path.exists():
        model = AutoModelForQuestionAnswering.from_pretrained(best_model_path)
        model.to(device)
        LOGGER.info("Best model reloaded from %s", best_model_path)

    result = {
        "model_name": config.model_name,
        "tokenizer_name": resolve_tokenizer_source(config),
        "best_model_path": str(best_model_path),
        "best_metric_name": metric_for_best,
        "best_metric_value": float(best_metric_value) if best_metric_value != float("-inf") else None,
        "output_dir": str(output_dir),
        "init_checkpoint_dir": config.init_checkpoint_dir,
    }
    (output_dir / "training_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune generic HF QA models (DistilRoBERTa/TinyBERT)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AutoQATrainConfig.from_yaml(args.config)
    setup_logging(config.log_level)
    result = train(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

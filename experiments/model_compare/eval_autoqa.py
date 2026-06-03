from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import torch

try:
    from .common import AutoQATrainConfig, disable_broken_torchvision, select_device, setup_logging
except ImportError:
    SCRIPT_DIR = Path(__file__).resolve().parent
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from common import AutoQATrainConfig, disable_broken_torchvision, select_device, setup_logging  # type: ignore

disable_broken_torchvision()

from transformers import AutoModelForQuestionAnswering, AutoTokenizer

try:
    from .train_autoqa import (
        _build_qa_datasets,
        _build_validation_eval_inputs,
        _run_em_f1_validation,
    )
except ImportError:
    from train_autoqa import (  # type: ignore
        _build_qa_datasets,
        _build_validation_eval_inputs,
        _run_em_f1_validation,
    )

LOGGER = logging.getLogger(__name__)


def _evaluate_loss_and_span_em(model: torch.nn.Module, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    span_exact_match = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(device=device)
                for key, value in batch.items()
                if hasattr(value, "to")
            }
            outputs = model(**batch)
            if outputs.loss is not None:
                total_loss += float(outputs.loss.item())

            start_pred = outputs.start_logits.argmax(dim=-1)
            end_pred = outputs.end_logits.argmax(dim=-1)
            span_exact_match += (
                (start_pred == batch["start_positions"]) &
                (end_pred == batch["end_positions"])
            ).sum().item()
            total += batch["start_positions"].size(dim=0)

    return {
        "loss": total_loss / max(1, len(loader)),
        "span_exact_match": span_exact_match / max(1, total),
    }


def evaluate_checkpoint(
    checkpoint_dir: str | Path,
    config: AutoQATrainConfig,
) -> dict[str, float | str]:
    checkpoint_dir = Path(checkpoint_dir)
    device = select_device(force_cpu=config.force_cpu)

    tokenizer_source = str(checkpoint_dir) if checkpoint_dir.exists() else config.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)

    raw_datasets, processed_datasets = _build_qa_datasets(tokenizer=tokenizer, config=config)
    validation_dataset = processed_datasets.get("validation")
    raw_validation_dataset = raw_datasets.get("validation")

    if validation_dataset is None or raw_validation_dataset is None:
        raise ValueError("Validation split is required for evaluation.")

    loader = torch.utils.data.DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)
    model = AutoModelForQuestionAnswering.from_pretrained(str(checkpoint_dir))
    model.to(device)

    loss_metrics = _evaluate_loss_and_span_em(model=model, loader=loader, device=device)
    validation_eval_inputs = _build_validation_eval_inputs(
        raw_validation_dataset=raw_validation_dataset,
        tokenizer=tokenizer,
        config=config,
    )
    em_f1_metrics = _run_em_f1_validation(
        model=model,
        device=device,
        validation_eval_inputs=validation_eval_inputs,
        batch_size=config.batch_size,
        use_amp=config.use_amp and device.type == "cuda",
        max_answer_length=config.max_answer_length,
    )

    return {
        "checkpoint_dir": str(checkpoint_dir),
        "model_name": config.model_name,
        "validation_file": config.validation_file,
        "loss": float(loss_metrics["loss"]),
        "span_exact_match": float(loss_metrics["span_exact_match"]),
        "exact_match": float(em_f1_metrics["exact_match"]),
        "f1": float(em_f1_metrics["f1"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generic HF QA checkpoint")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint-dir", required=True, help="Path to checkpoint (best_model)")
    parser.add_argument("--output-json", default=None, help="Optional output json path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AutoQATrainConfig.from_yaml(args.config)
    setup_logging(config.log_level)

    metrics = evaluate_checkpoint(checkpoint_dir=args.checkpoint_dir, config=config)
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        LOGGER.info("Saved metrics to %s", out_path)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

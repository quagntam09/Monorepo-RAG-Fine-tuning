from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .config import TrainingConfig
from .data_loader import build_qa_datasets, load_raw_datasets
from .modeling import build_model
from .trainer import _build_validation_eval_inputs, _run_em_f1_validation

logger = logging.getLogger(__name__)


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """
    Evaluation cho extractive QA.

    Returns:
        {"loss": float, "span_exact_match": float}
    """
    model.eval()
    total_loss = 0.0
    exact_match = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device=device) for k, v in batch.items() if hasattr(v, "to")}
            outputs = model(**batch)

            total_loss += outputs.loss.item()

            start_pred = outputs.start_logits.argmax(dim=-1)
            end_pred = outputs.end_logits.argmax(dim=-1)
            exact_match += (
                (start_pred == batch["start_positions"]) &
                (end_pred == batch["end_positions"])
            ).sum().item()
            total += batch["start_positions"].size(dim=0)

    return {
        "loss": total_loss / max(1, len(loader)),
        "span_exact_match": exact_match / max(1, total),
    }


def _select_device(force_cpu: bool = False) -> torch.device:
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def evaluate_checkpoint(
    checkpoint_dir: str | Path,
    config: TrainingConfig,
) -> dict[str, float | str]:
    checkpoint_dir = Path(checkpoint_dir)
    device = _select_device(config.force_cpu)

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    datasets = build_qa_datasets(tokenizer=tokenizer, config=config)
    validation_dataset = datasets.get("validation")
    if validation_dataset is None:
        raise ValueError("Validation split is required for checkpoint evaluation.")

    raw_validation_dataset = load_raw_datasets(config=config).get("validation")
    if raw_validation_dataset is None:
        raise ValueError("Raw validation split is required for EM/F1 evaluation.")

    model = build_model(
        model_name=config.model_name,
        dropout=config.dropout,
        freeze_encoder=config.freeze_encoder,
    )
    checkpoint_file = checkpoint_dir / "pytorch_model.bin"
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {checkpoint_file}")
    model.load_state_dict(torch.load(checkpoint_file, map_location=device))
    model.to(device)

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
    )

    loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)
    loss_metrics = evaluate(model=model, loader=loader, device=device)
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "loss": float(loss_metrics["loss"]),
        "span_exact_match": float(loss_metrics["span_exact_match"]),
        "exact_match": float(em_f1_metrics["exact_match"]),
        "f1": float(em_f1_metrics["f1"]),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a DistilBERT QA checkpoint")
    parser.add_argument("--config", default="config/defaults.yaml")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints/best_model")
    return parser.parse_args(argv)


def _resolve_config_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        repo_candidate = Path(__file__).resolve().parents[2] / path
        if repo_candidate.exists():
            return repo_candidate
    return path


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args(argv)
    config_path = _resolve_config_path(args.config)
    config = TrainingConfig.from_yaml(config_path) if config_path.exists() else TrainingConfig()
    metrics = evaluate_checkpoint(args.checkpoint_dir, config)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

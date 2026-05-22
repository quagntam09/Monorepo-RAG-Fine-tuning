from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from .config import TrainingConfig
from .evaluate import evaluate_checkpoint
from .export import export_reader_artifact
from .trainer import train


def _load_config(config_path: str) -> TrainingConfig:
    path = Path(config_path)
    if not path.is_absolute():
        repo_candidate = Path(__file__).resolve().parents[2] / path
        if repo_candidate.exists():
            path = repo_candidate
    if path.exists():
        return TrainingConfig.from_yaml(path)
    return TrainingConfig()


def _apply_overrides(
    config: TrainingConfig,
    args: argparse.Namespace,
    ignored_keys: set[str] | None = None,
) -> TrainingConfig:
    ignored = set(ignored_keys or set()) | {"config"}
    for key, value in vars(args).items():
        if key in ignored:
            continue
        if value is not None and hasattr(config, key):
            setattr(config, key, value)
    return config


def _train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for extractive QA")
    parser.add_argument("--config", default="config/defaults.yaml")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--doc_stride", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--artifact_dir", type=str)
    parser.add_argument("--artifact_name", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--validation_file", type=str)
    parser.add_argument("--force_cpu", action="store_true", default=None)
    parser.add_argument("--best_metric", type=str)
    parser.add_argument("--load_best_model", action="store_true", default=None)
    parser.add_argument("--save_best_model", action="store_true", default=None)
    parser.add_argument("--use_vietnamese_segmentation", action="store_true", default=None)
    return parser


def _eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a DistilBERT QA checkpoint")
    parser.add_argument("--config", default="config/defaults.yaml")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--force-cpu", action="store_true", default=None)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--validation_file", type=str)
    return parser


def _export_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a best DistilBERT QA checkpoint to a reader artifact")
    parser.add_argument("--config", default="config/defaults.yaml")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--artifact-name", default=None)
    parser.add_argument("--metric-name", default=None)
    parser.add_argument("--metric-value", type=float, default=None)
    parser.add_argument("--opset-version", type=int, default=None)
    parser.add_argument("--no-quantize", action="store_true")
    return parser


def train_main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _train_parser().parse_args(argv)
    config = _apply_overrides(_load_config(args.config), args)
    result = train(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def eval_main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _eval_parser().parse_args(argv)
    config = _apply_overrides(_load_config(args.config), args)
    checkpoint_dir = args.checkpoint_dir or f"{config.output_dir}/best_model"
    metrics = evaluate_checkpoint(checkpoint_dir=checkpoint_dir, config=config)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def export_main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _export_parser().parse_args(argv)
    config = _load_config(args.config)
    checkpoint_dir = args.checkpoint_dir or f"{config.output_dir}/best_model"
    artifact_dir = args.artifact_dir or config.artifact_dir
    model_name = args.model_name or config.model_name
    max_length = args.max_length or config.max_length
    artifact_name = args.artifact_name or config.artifact_name
    metric_name = args.metric_name or config.best_metric
    opset_version = args.opset_version or config.onnx_opset_version
    artifact_path = export_reader_artifact(
        checkpoint_dir=checkpoint_dir,
        artifact_dir=artifact_dir,
        model_name_or_path=model_name,
        max_length=max_length,
        quantize=not args.no_quantize,
        opset_version=opset_version,
        artifact_name=artifact_name,
        metric_name=metric_name,
        metric_value=args.metric_value,
    )
    print(json.dumps({"artifact_dir": str(artifact_path)}, indent=2, ensure_ascii=False))

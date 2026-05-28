from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Sequence

import onnx
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoTokenizer

from .config import TrainingConfig
from .model_config import build_config
from .modeling import DistilBertForQuestionAnswering

logger = logging.getLogger(__name__)


class ONNXQuestionAnsweringWrapper(nn.Module):
    """Export wrapper that keeps the DistilBERT attention mask contract intact."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.start_logits, outputs.end_logits


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _safe_load_state_dict(checkpoint_dir: Path) -> dict:
    checkpoint_path = checkpoint_dir / "pytorch_model.bin"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint weights: {checkpoint_path}")
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def infer_metric_from_checkpoint(checkpoint_dir: str | Path, metric_name: str) -> float | None:
    checkpoint_path = Path(checkpoint_dir)
    metric_key = (metric_name or "").strip().lower()
    if not metric_key:
        return None

    candidates = [
        checkpoint_path / "training_result.json",
        checkpoint_path.parent / "training_result.json",
    ]
    for path in candidates:
        payload = _read_json(path)
        if not payload:
            continue

        if "best_metric_name" in payload and "best_metric_value" in payload:
            name = str(payload.get("best_metric_name", "")).strip().lower()
            value = payload.get("best_metric_value")
            if name == metric_key and isinstance(value, (int, float)):
                return float(value)

        value = payload.get(metric_key)
        if isinstance(value, (int, float)):
            return float(value)

    return None


def _tokenizer_source(checkpoint_dir: Path, model_name_or_path: str) -> str:
    required = ["tokenizer.json", "tokenizer_config.json"]
    if all((checkpoint_dir / name).exists() for name in required):
        return str(checkpoint_dir)
    return model_name_or_path


def export_reader_artifact(
    checkpoint_dir: str | Path,
    artifact_dir: str | Path,
    model_name_or_path: str,
    *,
    max_length: int = 384,
    quantize: bool = True,
    opset_version: int = 14,
    artifact_name: str = "run_best",
    metric_name: str | None = None,
    metric_value: float | None = None,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(model_name_or_path)
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = "eager"
    if hasattr(config, "attn_implementation"):
        config.attn_implementation = "eager"

    model = DistilBertForQuestionAnswering(config=config, dropout=getattr(config, "dropout", 0.1))
    state_dict = _safe_load_state_dict(checkpoint_dir)
    model.load_state_dict(state_dict)
    model.eval()

    wrapper = ONNXQuestionAnsweringWrapper(model)
    wrapper.eval()

    dummy_input_ids = torch.randint(0, config.vocab_size, (1, max_length), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, max_length), dtype=torch.long)

    onnx_path = artifact_dir / "model.onnx"
    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["start_logits", "end_logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_length"},
            "attention_mask": {0: "batch_size", 1: "seq_length"},
            "start_logits": {0: "batch_size", 1: "seq_length"},
            "end_logits": {0: "batch_size", 1: "seq_length"},
        },
        opset_version=opset_version,
        dynamo=False,
        do_constant_folding=True,
    )

    onnx.checker.check_model(onnx.load(str(onnx_path)))

    quantized_path = artifact_dir / "model_quantized.onnx"
    if quantize:
        quantize_dynamic(str(onnx_path), str(quantized_path), weight_type=QuantType.QInt8)

    tokenizer = AutoTokenizer.from_pretrained(_tokenizer_source(checkpoint_dir, model_name_or_path))
    tokenizer.save_pretrained(artifact_dir)

    metadata = {
        "artifact_name": artifact_name,
        "artifact_type": "distilbert_onnx_reader",
        "export_time_utc": dt.datetime.now(dt.UTC).isoformat(),
        "source_checkpoint_dir": str(checkpoint_dir),
        "model_name": model_name_or_path,
        "metric": {
            "name": metric_name,
            "value": metric_value,
        },
        "onnx": {
            "model_path": str(onnx_path),
            "model_size_bytes": onnx_path.stat().st_size,
            "model_sha256": _sha256_file(onnx_path),
            "quantized_model_path": str(quantized_path) if quantized_path.exists() else None,
            "quantized_model_size_bytes": quantized_path.stat().st_size if quantized_path.exists() else None,
            "quantized_model_sha256": _sha256_file(quantized_path) if quantized_path.exists() else None,
        },
        "tokenizer": {
            "tokenizer_json_path": str(artifact_dir / "tokenizer.json"),
            "tokenizer_json_sha256": _sha256_file(artifact_dir / "tokenizer.json"),
            "tokenizer_config_path": str(artifact_dir / "tokenizer_config.json"),
            "tokenizer_config_sha256": _sha256_file(artifact_dir / "tokenizer_config.json"),
        },
        "runtime": {
            "max_length": max_length,
            "opset_version": opset_version,
        },
        "checkpoint_config": _read_json(checkpoint_dir / "config.json"),
        "training_config": _read_json(checkpoint_dir / "training_config.json"),
    }
    (artifact_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    return artifact_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export best DistilBERT QA checkpoint to a reader artifact")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints/best_model")
    parser.add_argument("--artifact-dir", default="artifacts/readers/run_best")
    parser.add_argument("--model-name", default="distilbert-base-multilingual-cased")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--artifact-name", default="run_best")
    parser.add_argument("--metric-name", default="f1")
    parser.add_argument("--metric-value", type=float, default=None)
    parser.add_argument("--opset-version", type=int, default=14)
    parser.add_argument("--no-quantize", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    artifact_dir = export_reader_artifact(
        checkpoint_dir=args.checkpoint_dir,
        artifact_dir=args.artifact_dir,
        model_name_or_path=args.model_name,
        max_length=args.max_length,
        quantize=not args.no_quantize,
        opset_version=args.opset_version,
        artifact_name=args.artifact_name,
        metric_name=args.metric_name,
        metric_value=args.metric_value,
    )
    logger.info("Exported reader artifact to %s", artifact_dir)


if __name__ == "__main__":
    main()

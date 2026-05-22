from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from training.export import export_reader_artifact


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class DummyTokenizer:
    def save_pretrained(self, artifact_dir: str | Path) -> None:
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        (artifact_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")


class DummyModel:
    def __init__(self, config, dropout: float = 0.1):
        self.config = config
        self.dropout = dropout

    def load_state_dict(self, state_dict):
        self.state_dict = state_dict

    def eval(self):
        return self


class TestExportArtifact(unittest.TestCase):
    def test_export_metadata_and_checksums(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_dir = root / "best_model"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "pytorch_model.bin").write_bytes(b"weights")

            with (
                patch("training.export.build_config", return_value=SimpleNamespace(vocab_size=16, dropout=0.1)),
                patch("training.export.DistilBertForQuestionAnswering", DummyModel),
                patch("training.export._safe_load_state_dict", return_value={}),
                patch("training.export._resolve_git_commit", return_value="deadbeef"),
                patch("training.export.AutoTokenizer.from_pretrained", return_value=DummyTokenizer()),
                patch("training.export.torch.onnx.export", side_effect=lambda *args, **kwargs: Path(args[2]).write_bytes(b"onnx")),
                patch("training.export.onnx.load", return_value=SimpleNamespace()),
                patch("training.export.onnx.checker.check_model", return_value=None),
                patch("training.export.quantize_dynamic", side_effect=lambda *args, **kwargs: Path(args[1]).write_bytes(b"qonnx")),
            ):
                artifact_dir = export_reader_artifact(
                    checkpoint_dir=checkpoint_dir,
                    artifact_dir=root / "artifacts/readers/run_best",
                    model_name_or_path="distilbert-base-multilingual-cased",
                    quantize=True,
                    artifact_name="run_best",
                    metric_name="f1",
                    metric_value=0.91,
                )

            metadata = json.loads((artifact_dir / "model_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["artifact_name"], "run_best")
            self.assertEqual(metadata["metric"]["name"], "f1")
            self.assertEqual(metadata["metric"]["value"], 0.91)
            self.assertEqual(metadata["onnx"]["model_sha256"], _sha256(b"onnx"))
            self.assertEqual(metadata["onnx"]["quantized_model_sha256"], _sha256(b"qonnx"))
            self.assertEqual(metadata["tokenizer"]["tokenizer_json_sha256"], _sha256(b"{}"))
            self.assertEqual(metadata["tokenizer"]["tokenizer_config_sha256"], _sha256(b"{}"))


if __name__ == "__main__":
    unittest.main()


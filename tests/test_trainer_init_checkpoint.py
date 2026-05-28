from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from training.config import TrainingConfig
from training.trainer import _load_initial_checkpoint_if_configured, _resolve_tokenizer_source


class TrainerInitCheckpointTests(unittest.TestCase):
    def test_resolve_tokenizer_source_uses_init_checkpoint_when_tokenizer_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = Path(tmp_dir)
            (checkpoint_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            (checkpoint_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")

            config = TrainingConfig(
                model_name="distilbert-base-multilingual-cased",
                init_checkpoint_dir=str(checkpoint_dir),
            )
            self.assertEqual(_resolve_tokenizer_source(config), str(checkpoint_dir))

    def test_resolve_tokenizer_source_falls_back_to_model_name_when_tokenizer_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = TrainingConfig(
                model_name="distilbert-base-multilingual-cased",
                init_checkpoint_dir=str(Path(tmp_dir)),
            )
            self.assertEqual(_resolve_tokenizer_source(config), "distilbert-base-multilingual-cased")

    def test_load_initial_checkpoint_applies_saved_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = Path(tmp_dir)
            checkpoint_file = checkpoint_dir / "pytorch_model.bin"

            source_model = nn.Linear(4, 2)
            torch.nn.init.constant_(source_model.weight, 0.5)
            torch.nn.init.constant_(source_model.bias, 0.25)
            torch.save(source_model.state_dict(), checkpoint_file)

            target_model = nn.Linear(4, 2)
            torch.nn.init.zeros_(target_model.weight)
            torch.nn.init.zeros_(target_model.bias)

            config = TrainingConfig(
                model_name="distilbert-base-multilingual-cased",
                init_checkpoint_dir=str(checkpoint_dir),
            )
            loaded_from = _load_initial_checkpoint_if_configured(target_model, config)

            self.assertEqual(loaded_from, str(checkpoint_dir))
            self.assertTrue(torch.equal(target_model.weight, source_model.weight))
            self.assertTrue(torch.equal(target_model.bias, source_model.bias))

    def test_load_initial_checkpoint_raises_when_weight_file_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = TrainingConfig(
                model_name="distilbert-base-multilingual-cased",
                init_checkpoint_dir=str(Path(tmp_dir)),
            )
            with self.assertRaises(FileNotFoundError):
                _load_initial_checkpoint_if_configured(nn.Linear(2, 2), config)


if __name__ == "__main__":
    unittest.main()

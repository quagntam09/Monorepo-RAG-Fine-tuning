from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class TrainingConfig:
    """Training configuration cho DistilBERT Extractive QA."""

    # ── Model Architecture ────────────────────────────────────────────────────
    model_name: str = "distilbert-base-multilingual-cased"
    dropout: float = 0.1
    freeze_encoder: bool = False  # True = chỉ train QA head

    # ── Dataset Configuration ────────────────────────────────────────────────
    dataset_name: Optional[str] = "taidng/UIT-ViQuAD2.0"
    dataset_config_name: Optional[str] = None
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None

    # Column names
    question_column: str = "question"
    context_column: str = "context"
    answers_column: str = "answers"
    impossible_column: str = "is_impossible"
    plausible_answers_column: str = "plausible_answers"

    # Text processing
    max_length: int = 384
    doc_stride: int = 128
    padding: str = "max_length"
    cache_dir: Optional[str] = None

    # Vietnamese segmentation
    use_vietnamese_segmentation: bool = False
    segmentation_tool: str = "underthesea"  # 'underthesea' hoặc 'pyvi'

    # ── Training Loop ────────────────────────────────────────────────────────
    batch_size: int = 16  # Tăng từ 8 lên 16 cho QA
    epochs: int = 3
    learning_rate: float = 3e-5  # Optimal cho fine-tuning
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # ── GPU / Throughput ────────────────────────────────────────────────────
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    use_amp: bool = True
    use_tf32: bool = True
    force_cpu: bool = False  # Force CPU training even if CUDA is available

    # ── Evaluation & Checkpointing ──────────────────────────────────────────
    eval_steps: Optional[int] = None  # Evaluate sau N steps (None = per epoch)
    save_steps: Optional[int] = None  # Save checkpoint sau N steps
    eval_strategy: str = "epoch"  # "epoch", "steps", "no"
    save_strategy: str = "epoch"
    save_best_model: bool = True
    best_metric: str = "f1"  # "exact_match" hoặc "f1"
    load_best_model: bool = True

    # ── Output ─────────────────────────────────────────────────────────────
    output_dir: str = "outputs/checkpoints"
    artifact_dir: str = "artifacts/readers/run_best"
    artifact_name: str = "run_best"
    onnx_opset_version: int = 14
    seed: int = 42

    # Logging
    logging_steps: int = 100
    log_level: str = "info"

    # ── Điều chỉnh fine-tuning sau này ─────────────────────────────────────────
    # Thêm field mới tại đây mà không cần chỉnh trainer.py:
    #   label_smoothing: float = 0.0
    #   bf16: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        """Load config từ file YAML."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def to_yaml(self, path: str | Path) -> None:
        """Lưu config ra file YAML."""
        import dataclasses
        Path(path).write_text(
            yaml.dump(data=dataclasses.asdict(self), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

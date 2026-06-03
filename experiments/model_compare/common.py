from __future__ import annotations

import dataclasses
import importlib
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import transformers.utils as transformers_utils
import yaml
from transformers.utils import import_utils as transformers_import_utils

LOGGER = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def disable_broken_torchvision() -> None:
    """Treat torchvision as unavailable when binary ops fail to import."""
    try:
        importlib.import_module("torchvision")
    except Exception:
        for module_name in list(sys.modules):
            if module_name == "torchvision" or module_name.startswith("torchvision."):
                sys.modules.pop(module_name, None)
        transformers_import_utils.is_torchvision_available = lambda: False
        transformers_utils.is_torchvision_available = lambda: False


@dataclass
class AutoQATrainConfig:
    model_name: str = "distilbert-base-multilingual-cased"
    tokenizer_name: Optional[str] = None
    init_checkpoint_dir: Optional[str] = None
    freeze_encoder: bool = False

    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None

    question_column: str = "question"
    context_column: str = "context"
    answers_column: str = "answers"
    impossible_column: str = "is_impossible"
    plausible_answers_column: str = "plausible_answers"

    max_length: int = 384
    doc_stride: int = 128
    padding: str = "max_length"
    cache_dir: Optional[str] = None

    use_vietnamese_segmentation: bool = False
    segmentation_tool: str = "underthesea"

    batch_size: int = 12
    epochs: int = 2
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    use_amp: bool = True
    use_tf32: bool = True
    force_cpu: bool = False

    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_best_model: bool = True
    best_metric: str = "f1"
    load_best_model: bool = True

    output_dir: str = "outputs/model_compare"
    seed: int = 42
    logging_steps: int = 100
    log_level: str = "info"

    max_answer_length: int = 30

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AutoQATrainConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def setup_logging(level: str) -> None:
    level_name = (level or "info").upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        try:
            _ = torch.randn(1, device="cuda")
            return torch.device("cuda")
        except Exception as exc:
            LOGGER.warning("CUDA probe failed (%s). Falling back to CPU.", exc)
    return torch.device("cpu")


def build_dataloader(dataset, config: AutoQATrainConfig, device: torch.device, shuffle: bool):
    from torch.utils.data import DataLoader

    num_workers = max(0, config.num_workers)
    kwargs = {
        "batch_size": config.batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": config.pin_memory and device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = config.persistent_workers
        kwargs["prefetch_factor"] = max(1, config.prefetch_factor)
    return DataLoader(dataset, **kwargs)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        if hasattr(value, "to")
        else value
        for key, value in batch.items()
    }


def maybe_freeze_encoder(model: torch.nn.Module) -> int:
    encoder_prefixes = (
        "bert.",
        "roberta.",
        "distilbert.",
        "deberta.",
        "deberta_v2.",
        "albert.",
        "electra.",
        "mobilebert.",
        "mpnet.",
        "xlm_roberta.",
    )
    frozen_params = 0
    for name, param in model.named_parameters():
        if name.startswith(encoder_prefixes):
            param.requires_grad = False
            frozen_params += param.numel()
    return frozen_params


def resolve_tokenizer_source(config: AutoQATrainConfig) -> str:
    if config.init_checkpoint_dir:
        checkpoint_dir = Path(config.init_checkpoint_dir)
        if (checkpoint_dir / "tokenizer.json").exists() and (checkpoint_dir / "tokenizer_config.json").exists():
            return str(checkpoint_dir)
    if config.tokenizer_name:
        return config.tokenizer_name
    return config.model_name


def resolve_model_source(config: AutoQATrainConfig) -> str:
    if config.init_checkpoint_dir:
        return str(config.init_checkpoint_dir)
    return config.model_name

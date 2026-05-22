from __future__ import annotations

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from transformers import get_linear_schedule_with_warmup

from .config import TrainingConfig


def build_optimizer(model: torch.nn.Module, config: TrainingConfig) -> AdamW:
    """
    AdamW với weight decay — không áp lên bias và LayerNorm.
    Tách param group giúp fine-tuning ổn định hơn.
    """
    no_decay = {"bias", "LayerNorm.weight"}
    param_groups = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(params=param_groups, lr=config.learning_rate)


def build_scheduler(
    optimizer: AdamW,
    num_training_steps: int,
    config: TrainingConfig,
) -> LRScheduler:
    """Linear warmup → linear decay scheduler."""
    num_warmup_steps = max(1, int(num_training_steps * config.warmup_ratio))
    return get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

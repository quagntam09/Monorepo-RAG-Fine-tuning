"""
DistilBERT Question-Answering model for extractive QA.

Kiến trúc:
1. Pretrained DistilBERT encoder (từ HuggingFace)
2. QA head với 2 linear layers cho start/end token prediction
3. Cross-entropy loss cho training

Note: Sử dụng HuggingFace's DistilBertModel cho reliability và pre-trained weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import transformers.utils as transformers_utils
from transformers.utils import import_utils as transformers_import_utils


def _disable_broken_torchvision() -> None:
    """Treat torchvision as unavailable when its binary ops cannot be imported."""
    try:
        importlib.import_module("torchvision")
    except Exception:
        for module_name in list(sys.modules):
            if module_name == "torchvision" or module_name.startswith("torchvision."):
                sys.modules.pop(module_name, None)
        transformers_import_utils.is_torchvision_available = lambda: False
        transformers_utils.is_torchvision_available = lambda: False


_disable_broken_torchvision()

from transformers import (  # noqa: E402
    DistilBertModel,
    DistilBertConfig,
)

from .model_config import build_config


@dataclass
class QAModelOutput:
    """Output từ QA model."""

    loss: Optional[torch.Tensor] = None
    start_logits: torch.Tensor = None
    end_logits: torch.Tensor = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


class DistilBertForQuestionAnswering(nn.Module):
    """
    DistilBERT model cho extractive question-answering.

    Architecture:
    1. DistilBertModel: Pretrained encoder
    2. Dropout + 2 linear layers: Start/end token prediction

    Input:
        - input_ids: Token IDs (batch_size, seq_len)
        - attention_mask: Attention mask (batch_size, seq_len)
        - start_positions: Ground truth start positions (batch_size,) - optional
        - end_positions: Ground truth end positions (batch_size,) - optional

    Output:
        - start_logits: (batch_size, seq_len)
        - end_logits: (batch_size, seq_len)
        - loss: Sum of start + end cross-entropy losses (optional)
    """

    def __init__(self, config: DistilBertConfig, dropout: float = 0.1):
        """
        Args:
            config: DistilBertConfig from HuggingFace
            dropout: Dropout rate cho QA head
        """
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size

        # Pretrained encoder
        self.distilbert = DistilBertModel(config=config)

        # QA head
        self.dropout = nn.Dropout(p=dropout)
        self.qa_classifier = nn.Linear(in_features=self.hidden_size, out_features=2)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")

        self._init_weights()

    def _init_weights(self):
        """Initialize QA head weights."""
        init_std = self.config.initializer_range

        if hasattr(self.qa_classifier, "weight"):
            nn.init.normal_(self.qa_classifier.weight, mean=0.0, std=init_std)

        if hasattr(self.qa_classifier, "bias") and self.qa_classifier.bias is not None:
            nn.init.zeros_(self.qa_classifier.bias)

    def freeze_encoder(self) -> None:
        """Freeze encoder, train only QA head."""
        for param in self.distilbert.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save model state and config in HuggingFace-compatible format."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_directory / "pytorch_model.bin")
        if hasattr(self.config, "to_json_file"):
            self.config.to_json_file(save_directory / "config.json")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> QAModelOutput:
        """
        Forward pass.

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len) - optional
            start_positions: (batch_size,) - optional, for training
            end_positions: (batch_size,) - optional, for training
            **kwargs: Additional arguments for DistilBert (e.g., token_type_ids)

        Returns:
            QAModelOutput with start_logits, end_logits, and loss (if labels provided)
        """

        # DistilBERT forward pass
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Get sequence output (last hidden state)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        # Apply dropout
        sequence_output = self.dropout(sequence_output)

        # QA classifier
        logits = self.qa_classifier(sequence_output)  # (batch_size, seq_len, 2)
        start_logits, end_logits = logits.split(split_size=1, dim=-1)

        start_logits = start_logits.squeeze(-1).contiguous()  # (batch_size, seq_len)
        end_logits = end_logits.squeeze(-1).contiguous()      # (batch_size, seq_len)

        # Compute loss
        loss = None
        if start_positions is not None and end_positions is not None:
            # Ensure correct shapes
            if start_positions.dim() > 1:
                start_positions = start_positions.squeeze(-1)
            if end_positions.dim() > 1:
                end_positions = end_positions.squeeze(-1)

            # Clamp positions to valid range
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # CrossEntropyLoss
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(input=start_logits, target=start_positions)
            end_loss = loss_fct(input=end_logits, target=end_positions)
            loss = (start_loss + end_loss) / 2.0

        return QAModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )


def build_model(
    model_name: str,
    dropout: float = 0.1,
    freeze_encoder: bool = False,
) -> DistilBertForQuestionAnswering:
    """
    Build DistilBERT QA model từ pretrained checkpoint.

    Args:
        model_name: HuggingFace model name (VD: "distilbert-base-multilingual-cased")
        dropout: Dropout rate cho QA head
        freeze_encoder: Có freeze encoder hay không

    Returns:
        DistilBertForQuestionAnswering model
    """

    # Load config
    config = build_config(model_name=model_name, num_labels=2, dropout=dropout)

    # Create model
    model = DistilBertForQuestionAnswering(config=config, dropout=dropout)

    # Load pretrained encoder weights
    pretrained_model = DistilBertModel.from_pretrained(model_name)
    model.distilbert.load_state_dict(pretrained_model.state_dict())

    # Optionally freeze encoder
    if freeze_encoder:
        model.freeze_encoder()

    return model



# The custom DistilBERT PyTorch implementation was removed.
# The project uses the HuggingFace DistilBERT encoder + lightweight QA head above.

"""
Question-Answering head for DistilBERT.

Bao gồm:
1. QAHead: Đầu ra 2 logits (start token position + end token position)
2. Hỗ trợ tính loss cho training
3. Post-processing utilities để extract answers từ logits
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import DistilBertConfig
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class QuestionAnsweringOutput:
    """Output của QA model."""
    
    loss: Optional[torch.Tensor] = None
    start_logits: torch.Tensor = None  # Shape: (batch_size, seq_length)
    end_logits: torch.Tensor = None    # Shape: (batch_size, seq_length)
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


class QAHead(nn.Module):
    """
    Simple QA head: 2 linear layers cho start/end token prediction.
    
    Input: DistilBERT hidden states (batch_size, seq_len, hidden_size)
    Output: start_logits, end_logits (batch_size, seq_len)
    
    Loss: CrossEntropyLoss(start_logits, start_positions) + CrossEntropyLoss(end_logits, end_positions)
    """
    
    def __init__(self, config: DistilBertConfig, dropout: float = 0.1):
        """
        Args:
            config: DistilBertConfig
            dropout: Dropout rate cho QA head
        """
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(dropout)
        
        # Linear layers để dự đoán start/end token positions
        self.start_dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.end_dense = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.start_classifier = nn.Linear(self.hidden_size, 1)  # Output: 1 score per token
        self.end_classifier = nn.Linear(self.hidden_size, 1)
        
        # Activation function
        self.activation = nn.ReLU()
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
    ) -> QuestionAnsweringOutput:
        """
        Args:
            hidden_states: DistilBERT hidden states (batch_size, seq_len, hidden_size)
            start_positions: Ground truth start positions (batch_size,) - optional
            end_positions: Ground truth end positions (batch_size,) - optional
            
        Returns:
            QuestionAnsweringOutput với start_logits, end_logits, và loss (nếu labels được cung cấp)
        """
        
        # Start logits
        start_output = self.dropout(hidden_states)
        start_output = self.start_dense(start_output)
        start_output = self.activation(start_output)
        start_output = self.dropout(start_output)
        start_logits = self.start_classifier(start_output).squeeze(-1)  # (batch_size, seq_len)
        
        # End logits
        end_output = self.dropout(hidden_states)
        end_output = self.end_dense(end_output)
        end_output = self.activation(end_output)
        end_output = self.dropout(end_output)
        end_logits = self.end_classifier(end_output).squeeze(-1)  # (batch_size, seq_len)
        
        loss = None
        
        # Tính loss nếu labels được cung cấp (training mode)
        if start_positions is not None and end_positions is not None:
            # CrossEntropyLoss expects: (batch_size, num_classes) vs (batch_size,)
            # Ở đây seq_len được coi như "num_classes" (mỗi position là một class)
            start_loss = self.loss_fn(start_logits, start_positions)
            end_loss = self.loss_fn(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2.0
        
        return QuestionAnsweringOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
        )


def extract_answer_span(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    tokens: list[str],
    tokenizer,
    offset_mapping: Optional[list] = None,
    context: Optional[str] = None,
    max_answer_length: int = 30,
    n_best_size: int = 20,
) -> dict:
    """
    Trích xuất answer text từ start/end logits.
    
    Args:
        start_logits: (seq_len,) - logits cho start position
        end_logits: (seq_len,) - logits cho end position
        tokens: List các tokens
        tokenizer: HuggingFace tokenizer
        offset_mapping: Optional list của (char_start, char_end) cho mỗi token
        context: Optional context text để extract substring
        max_answer_length: Độ dài tối đa của answer (tokens)
        n_best_size: Số best spans để xem xét (để chọn highest score)
        
    Returns:
        Dict với:
        - text: Extracted answer text
        - start_pos: Start token position
        - end_pos: End token position
        - score: Confidence score
    """
    
    device = start_logits.device
    start_logits = start_logits.cpu().numpy()
    end_logits = end_logits.cpu().numpy()
    
    # Lấy top n_best predictions cho start/end
    start_indices = torch.argsort(torch.tensor(start_logits), descending=True)[:n_best_size]
    end_indices = torch.argsort(torch.tensor(end_logits), descending=True)[:n_best_size]
    
    # Tìm best span (start <= end, within max_answer_length)
    best_score = -float("inf")
    best_start = 0
    best_end = 0
    
    for start_idx in start_indices:
        start_idx = int(start_idx)
        for end_idx in end_indices:
            end_idx = int(end_idx)
            
            # Validate span
            if end_idx < start_idx:
                continue
            if end_idx - start_idx + 1 > max_answer_length:
                continue
            
            # Calculate score
            span_score = start_logits[start_idx] + end_logits[end_idx]
            
            if span_score > best_score:
                best_score = span_score
                best_start = start_idx
                best_end = end_idx
    
    # Extract answer text
    if offset_mapping and context:
        # Method 1: Từ offset mapping và context
        start_char = offset_mapping[best_start][0] if best_start < len(offset_mapping) else 0
        end_char = offset_mapping[best_end][1] if best_end < len(offset_mapping) else len(context)
        answer_text = context[start_char:end_char].strip()
    else:
        # Method 2: Từ tokens
        answer_tokens = tokens[best_start:best_end + 1]
        answer_text = tokenizer.convert_tokens_to_string(answer_tokens)
    
    return {
        "text": answer_text,
        "start_pos": best_start,
        "end_pos": best_end,
        "score": float(best_score),
    }


def post_process_predictions(
    predictions: dict,
    tokenizer,
    max_answer_length: int = 30,
) -> dict:
    """
    Post-process model predictions để lấy final answers.
    
    Args:
        predictions: Dict chứa:
            - start_logits: (batch_size, seq_len)
            - end_logits: (batch_size, seq_len)
            - tokens: List[List[str]]
            - offset_mapping: Optional
            - context: Optional context texts
        tokenizer: HuggingFace tokenizer
        max_answer_length: Max answer length
        
    Returns:
        Dict chứa extracted answers cho mỗi sample
    """
    
    batch_size = predictions["start_logits"].shape[0]
    results = []
    
    for idx in range(batch_size):
        start_logits = predictions["start_logits"][idx]
        end_logits = predictions["end_logits"][idx]
        
        tokens = predictions["tokens"][idx] if "tokens" in predictions else None
        offset_mapping = predictions["offset_mapping"][idx] if "offset_mapping" in predictions else None
        context = predictions["context"][idx] if "context" in predictions else None
        
        answer = extract_answer_span(
            start_logits=start_logits,
            end_logits=end_logits,
            tokens=tokens,
            tokenizer=tokenizer,
            offset_mapping=offset_mapping,
            context=context,
            max_answer_length=max_answer_length,
        )
        
        results.append(answer)
    
    return {
        "answers": results,
    }

"""
Model configuration utilities for DistilBERT QA.

Cung cấp functions để:
1. Load pre-trained DistilBERT config
2. Customize config cho QA task
3. Initialize model from config
"""

from __future__ import annotations

from transformers import AutoConfig, DistilBertConfig


def build_config(
    model_name: str,
    num_labels: int = 2,  # For QA: start and end token (binary classification per position)
    dropout: float = 0.1,
    attention_dropout: float = 0.1,
) -> DistilBertConfig:
    """
    Tạo DistilBertConfig cho QA task.
    
    Args:
        model_name: HuggingFace model name (VD: "distilbert-base-multilingual-cased")
        num_labels: Số labels (2 cho QA: start/end position prediction)
        dropout: Dropout rate cho feed-forward layers
        attention_dropout: Dropout rate cho attention weights
        
    Returns:
        DistilBertConfig configured cho QA
    """
    
    # Load pre-trained config
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model_name,
        num_labels=num_labels,
    )
    
    # Customize cho QA task
    if isinstance(config, DistilBertConfig):
        config.dropout = dropout
        config.attention_dropout = attention_dropout
        config.qa_classifier_dropout = dropout  # SQuAD-style QA head dropout
    
    return config


def get_config_summary(config: DistilBertConfig) -> dict:
    """
    Lấy summary của config để logging/debugging.
    
    Args:
        config: DistilBertConfig object
        
    Returns:
        Dict với các tham số quan trọng
    """
    return {
        "model_type": config.model_type,
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "intermediate_size": config.intermediate_size if hasattr(config, "intermediate_size") else "N/A",
        "hidden_dropout_prob": config.dropout if hasattr(config, "dropout") else "N/A",
        "attention_probs_dropout_prob": config.attention_dropout if hasattr(config, "attention_dropout") else "N/A",
        "initializer_range": config.initializer_range,
        "num_labels": config.num_labels,
    }


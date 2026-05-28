"""
Data loading utilities for QA datasets.

Hỗ trợ:
1. Tải từ HuggingFace Hub (VD: taidng/UIT-ViQuAD2.0)
2. Tải từ local files (SQuAD-format JSON, CSV, TSV)
3. Tự động preprocessing với prepare_train_features/prepare_eval_features
"""

from __future__ import annotations

from pathlib import Path
import logging
from typing import Optional

from datasets import DatasetDict, load_dataset

from .dataset import prepare_train_features, prepare_eval_features


logger = logging.getLogger(__name__)


def _infer_local_format(file_path: str) -> str:
    """Xác định định dạng file từ extension."""
    suffix = Path(file_path).suffix.lower()

    if suffix in {".json", ".jsonl"}:
        return "json"

    if suffix == ".csv":
        return "csv"

    if suffix == ".tsv":
        return "csv"  # TSV được xử lý như CSV với delimiter

    raise ValueError(f"Không hỗ trợ định dạng file: {file_path}")


def load_raw_datasets(config) -> DatasetDict:
    """
    Tải dataset QA từ HuggingFace Hub hoặc từ file local.

    Hỗ trợ hai cách:
    1. HuggingFace Hub: config.dataset_name = "user/dataset-name"
    2. Local files: config.train_file, config.validation_file, config.test_file

    Args:
        config: TrainingConfig object với các tham số dataset

    Returns:
        DatasetDict với splits "train", "validation", "test" (tùy khả dụng)
    """

    # Cách 1: Load từ HuggingFace Hub
    if config.dataset_name:
        logger.info(f"Loading dataset từ HuggingFace Hub: {config.dataset_name}")

        datasets = load_dataset(
            path=config.dataset_name,
            name=config.dataset_config_name,
            cache_dir=config.cache_dir,
        )

        logger.info(f"Dataset splits: {list(datasets.keys())}")
        return datasets

    # Cách 2: Load từ local files
    data_files: dict[str, str] = {}

    if config.train_file:
        data_files["train"] = config.train_file

    if config.validation_file:
        data_files["validation"] = config.validation_file

    if config.test_file:
        data_files["test"] = config.test_file

    if not data_files:
        raise ValueError(
            "Cần cung cấp dataset_name hoặc ít nhất một trong: "
            "train_file, validation_file, test_file"
        )

    # Xác định format
    first_file = next(iter(data_files.values()))
    data_format = _infer_local_format(file_path=first_file)

    logger.info(f"Loading local dataset từ files: {list(data_files.keys())}")
    logger.info(f"Detected format: {data_format}")

    load_kwargs = {
        "data_files": data_files,
        "cache_dir": config.cache_dir,
    }

    # Xử lý TSV files (CSV với delimiter khác)
    if data_format == "csv":
        tsv_files = [f for f in data_files.values() if f.endswith(".tsv")]
        if tsv_files:
            load_kwargs["delimiter"] = "\t"

    datasets = load_dataset(path=data_format, **load_kwargs)

    logger.info(f"Loaded splits: {list(datasets.keys())}")
    return datasets


def build_qa_datasets(tokenizer, config, is_training: bool = True) -> DatasetDict:
    """
    Tải và tokenize QA datasets.

    Args:
        tokenizer: HuggingFace tokenizer (phải hỗ trợ offset_mapping)
        config: TrainingConfig object
        is_training: True nếu dùng train/eval features, False nếu chỉ cần tokens

    Returns:
        DatasetDict với các splits đã được tokenized:
        - input_ids: Tokenized sequence IDs
        - attention_mask: Attention mask
        - start_positions & end_positions (nếu is_training=True)
    """

    raw_datasets = load_raw_datasets(config=config)

    processed = DatasetDict()

    for split_name, dataset in raw_datasets.items():
        logger.info(f"Processing split '{split_name}' ({len(dataset)} samples)")

        # Chọn hàm xử lý tùy theo split
        has_answers = config.answers_column in dataset.column_names
        has_context_labels = split_name in {"train", "validation"} and has_answers

        if has_context_labels:
            prepare_fn = prepare_train_features
            prepare_kwargs = {
                "answers_column": config.answers_column,
                "impossible_column": config.impossible_column,
            }
        else:
            prepare_fn = prepare_eval_features
            prepare_kwargs = {}

        # Tokenize
        processed_dataset = dataset.map(
            lambda examples: prepare_fn(
                examples=examples,
                tokenizer=tokenizer,
                question_column=config.question_column,
                context_column=config.context_column,
                max_length=config.max_length,
                doc_stride=config.doc_stride,
                padding=config.padding,
                use_vietnamese_segmentation=config.use_vietnamese_segmentation,
                segmentation_tool=config.segmentation_tool,
                **prepare_kwargs,
            ),
            batched=True,
            remove_columns=dataset.column_names,
            desc=f"Tokenizing {split_name}",
        )

        # Set PyTorch format
        if has_context_labels:
            # Training/validation loss: cần start/end positions
            processed_dataset.set_format(
                type="torch",
                columns=[
                    "input_ids",
                    "attention_mask",
                    "start_positions",
                    "end_positions",
                ],
            )
        else:
            # Evaluation: giữ offset mapping và sample_id để post-process predictions
            eval_columns = ["input_ids", "attention_mask"]
            if "offset_mapping" in processed_dataset.column_names:
                eval_columns.append("offset_mapping")
            if "sample_id" in processed_dataset.column_names:
                eval_columns.append("sample_id")
            processed_dataset.set_format(
                type="torch",
                columns=eval_columns,
            )

        processed[split_name] = processed_dataset
        logger.info(f"  → {len(processed_dataset)} features after tokenization")

    return processed


def load_dataset_for_inference(
    context: str,
    question: str,
    tokenizer,
    config,
) -> dict:
    """
    Chuẩn bị single sample cho inference (không yêu cầu answers).

    Args:
        context: Context text
        question: Question text
        tokenizer: HuggingFace tokenizer
        config: Config object (max_length, doc_stride, etc.)

    Returns:
        Dict với input_ids, attention_mask, offset_mapping, etc. cho inference
    """
    from .dataset import prepare_eval_features

    examples = {
        config.question_column: [question],
        config.context_column: [context],
    }

    features = prepare_eval_features(
        examples=examples,
        tokenizer=tokenizer,
        question_column=config.question_column,
        context_column=config.context_column,
        max_length=config.max_length,
        doc_stride=config.doc_stride,
        padding=config.padding,
        use_vietnamese_segmentation=config.use_vietnamese_segmentation,
        segmentation_tool=config.segmentation_tool,
    )

    # Convert to tensors
    import torch

    result = {}
    for key in ["input_ids", "attention_mask"]:
        if key in features:
            result[key] = torch.tensor([features[key][0]], dtype=torch.long)

    # Keep offset_mapping for post-processing
    if "offset_mapping" in features:
        result["offset_mapping"] = features["offset_mapping"]

    return result

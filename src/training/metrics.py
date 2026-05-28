"""
Evaluation metrics for Extractive Question-Answering (QA).

Cung cấp:
1. Exact Match (EM): Predicted answer khớp hoàn toàn với gold answer
2. F1-Score: Token-level overlap giữa predicted và gold answers
3. compute_metrics: Main function cho evaluation pipeline
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

_SQUAD_METRIC = None


def _load_squad_metric():
    """Lazy-load evaluate metric để tránh crash khi import module."""
    global _SQUAD_METRIC
    if _SQUAD_METRIC is not None:
        return _SQUAD_METRIC
    try:
        import evaluate  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Thiếu thư viện 'evaluate'. Cài bằng: pip install evaluate"
        ) from exc
    _SQUAD_METRIC = evaluate.load("squad")
    return _SQUAD_METRIC


def normalize_answer(s: str) -> str:
    """
    Chuẩn hóa text để so sánh: lowercase + remove articles/punctuation/extra spaces.
    
    Args:
        s: Raw text
        
    Returns:
        Normalized text
    """
    
    def remove_articles(text):
        """Remove a, an, the."""
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    
    def white_space_fix(text):
        """Loại bỏ extra whitespace."""
        return ' '.join(text.split())
    
    def remove_punc(text):
        """Loại bỏ punctuation."""
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        """Lowercase."""
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    """
    Tính F1-Score giữa prediction và ground truth dựa trên token overlap.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Trong QA:
    - Precision = common tokens / len(prediction tokens)
    - Recall = common tokens / len(ground_truth tokens)
    
    Args:
        prediction: Predicted answer text
        ground_truth: Ground truth answer text
        
    Returns:
        F1-Score (0.0 to 1.0)
    """
    
    # Normalize texts
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    # Common tokens
    common_tokens = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_common = sum(common_tokens.values())
    
    # Handle edge cases
    if len(prediction_tokens) == 0 and len(ground_truth_tokens) == 0:
        return 1.0 if prediction == ground_truth else 0.0
    
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return 0.0 if num_common == 0 else 0.0
    
    # Calculate precision and recall
    precision = num_common / len(prediction_tokens)
    recall = num_common / len(ground_truth_tokens)
    
    # Calculate F1
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """
    Kiểm tra exact match: prediction == ground_truth (sau khi normalize).
    
    Args:
        prediction: Predicted answer text
        ground_truth: Ground truth answer text
        
    Returns:
        True nếu khớp, False nếu không
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def compute_exact_and_f1(
    prediction: str,
    ground_truth_list: list[str],
) -> Tuple[float, float]:
    """
    Tính EM và F1 cho một prediction vs multiple gold answers (SQuAD-style).
    
    Lấy maximum F1 và maximum EM từ tất cả gold answers.
    
    Args:
        prediction: Predicted answer text
        ground_truth_list: List các gold answer texts
        
    Returns:
        (exact_match, f1_score) - max values across all gold answers
    """
    
    if not ground_truth_list:
        # Nếu không có gold answers, return 0
        return 0.0, 0.0
    
    exact_matches = [exact_match_score(prediction, gt) for gt in ground_truth_list]
    f1_scores = [f1_score(prediction, gt) for gt in ground_truth_list]
    
    # Return max values
    max_exact_match = max(exact_matches) if exact_matches else 0.0
    max_f1 = max(f1_scores) if f1_scores else 0.0
    
    return float(max_exact_match), max_f1


def compute_metrics(eval_preds) -> dict:
    """
    Tính EM/F1 theo chuẩn SQuAD từ logits + metadata validation.

    `eval_preds` là dict do validation loop truyền vào, gồm:
    - start_logits: np.ndarray/list, shape (N, seq_len)
    - end_logits: np.ndarray/list, shape (N, seq_len)
    - offset_mapping: list[list[tuple[int, int]]]
    - contexts: list[str] (context đã dùng lúc tokenize)
    - example_ids: list[str] (id của sample gốc)
    - references: list[dict] theo format SQuAD:
      {"id": "...", "answers": {"text": [...], "answer_start": [...]} }
    
    Returns:
        {"exact_match": float, "f1": float}
    """
    if not isinstance(eval_preds, dict):
        raise ValueError("eval_preds phải là dict chứa logits và metadata.")

    start_logits = np.asarray(eval_preds.get("start_logits", []))
    end_logits = np.asarray(eval_preds.get("end_logits", []))
    offset_mappings = eval_preds.get("offset_mapping", [])
    contexts = eval_preds.get("contexts", [])
    example_ids = eval_preds.get("example_ids", [])
    references = eval_preds.get("references", [])
    max_answer_length = int(eval_preds.get("max_answer_length", 30))

    if start_logits.size == 0 or end_logits.size == 0:
        return {"exact_match": 0.0, "f1": 0.0}

    n_samples = min(
        len(start_logits),
        len(end_logits),
        len(offset_mappings),
        len(contexts),
        len(example_ids),
    )
    if n_samples == 0:
        return {"exact_match": 0.0, "f1": 0.0}

    predictions: list[dict] = []
    n_best_size = 20
    for idx in range(n_samples):
        offsets = offset_mappings[idx]
        context = contexts[idx]

        pred_text = ""
        if not isinstance(offsets, list):
            offsets = list(offsets)

        start_indexes = np.argsort(start_logits[idx])[-n_best_size:][::-1]
        end_indexes = np.argsort(end_logits[idx])[-n_best_size:][::-1]

        best_score = float("-inf")
        best_text = ""
        for s_idx in start_indexes:
            for e_idx in end_indexes:
                if e_idx < s_idx:
                    continue
                if (e_idx - s_idx + 1) > max_answer_length:
                    continue
                if not (0 <= s_idx < len(offsets) and 0 <= e_idx < len(offsets)):
                    continue

                s_offset = offsets[s_idx]
                e_offset = offsets[e_idx]
                if s_offset is None or e_offset is None:
                    continue

                start_char, _ = s_offset
                _, end_char = e_offset
                if not (
                    isinstance(start_char, (int, np.integer))
                    and isinstance(end_char, (int, np.integer))
                    and end_char > start_char >= 0
                    and end_char <= len(context)
                ):
                    continue

                cand_text = context[start_char:end_char].strip()
                cand_score = float(start_logits[idx][s_idx] + end_logits[idx][e_idx])
                if cand_score > best_score:
                    best_score = cand_score
                    best_text = cand_text

        pred_text = best_text

        predictions.append({"id": str(example_ids[idx]), "prediction_text": pred_text})

    # Chỉ giữ references tương ứng prediction ids đang có (tránh mismatch do overflow/window).
    pred_ids = {p["id"] for p in predictions}
    filtered_references = [r for r in references if str(r.get("id")) in pred_ids]

    if len(filtered_references) == 0:
        return {"exact_match": 0.0, "f1": 0.0}

    # Defensive sanitize: metric SQuAD cần answers.text không rỗng cho từng reference.
    sanitized_references = []
    for ref in filtered_references:
        answers = ref.get("answers", {}) if isinstance(ref, dict) else {}
        texts = list(answers.get("text", [])) if isinstance(answers, dict) else []
        starts = list(answers.get("answer_start", [])) if isinstance(answers, dict) else []

        if len(texts) == 0:
            texts = [""]
            starts = [0]
        elif len(starts) < len(texts):
            starts = starts + [0] * (len(texts) - len(starts))

        sanitized_references.append(
            {
                "id": str(ref.get("id")),
                "answers": {"text": texts, "answer_start": starts},
            }
        )

    squad_metric = _load_squad_metric()
    metric_result = squad_metric.compute(
        predictions=predictions,
        references=sanitized_references,
    )
    return {
        "exact_match": float(metric_result.get("exact_match", 0.0)),
        "f1": float(metric_result.get("f1", 0.0)),
    }


def compute_metrics_from_logits(
    start_logits: list,
    end_logits: list,
    gold_answers: list,
    tokens_lists: Optional[list] = None,
    offset_mappings: Optional[list] = None,
    contexts: Optional[list] = None,
    tokenizer = None,
    max_answer_length: int = 30,
) -> dict:
    """
    Compute EM + F1 directly từ model logits (dùng trong training loop).
    
    Args:
        start_logits: List[Tensor] - (seq_len,) logits cho mỗi sample
        end_logits: List[Tensor] - (seq_len,) logits cho mỗi sample
        gold_answers: List[List[str]] - Gold answer texts (có thể multiple per sample)
        tokens_lists: Optional[List[List[str]]] - Tokenized sequences
        offset_mappings: Optional[List[List[Tuple]]] - Character offsets
        contexts: Optional[List[str]] - Context texts
        tokenizer: HuggingFace tokenizer (needed for token-to-string conversion)
        max_answer_length: Max length của extracted answer
        
    Returns:
        Dict chứa EM + F1 metrics
    """
    
    import torch
    
    predictions = []
    
    for idx in range(len(start_logits)):
        start_logit = start_logits[idx]
        end_logit = end_logits[idx]
        
        # Convert to numpy if torch tensor
        if isinstance(start_logit, torch.Tensor):
            start_logit = start_logit.cpu().numpy()
        if isinstance(end_logit, torch.Tensor):
            end_logit = end_logit.cpu().numpy()
        
        # Find best span
        start_idx = start_logit.argmax()
        end_idx = end_logit.argmax()
        
        # Validate and limit answer length
        if end_idx < start_idx:
            end_idx = start_idx
        
        if end_idx - start_idx + 1 > max_answer_length:
            end_idx = start_idx + max_answer_length - 1
        
        # Extract answer text
        if tokens_lists and idx < len(tokens_lists):
            tokens = tokens_lists[idx]
            if start_idx < len(tokens) and end_idx < len(tokens):
                answer_tokens = tokens[start_idx:end_idx + 1]
                if tokenizer:
                    answer_text = tokenizer.convert_tokens_to_string(answer_tokens)
                else:
                    answer_text = " ".join(answer_tokens)
            else:
                answer_text = ""
        else:
            answer_text = ""
        
        predictions.append(answer_text)
    
    # Compute metrics
    exact_matches = []
    f1_scores = []
    
    for pred, gold_list in zip(predictions, gold_answers):
        if not gold_list:
            gold_list = [""]
        
        em, f1 = compute_exact_and_f1(pred, gold_list)
        exact_matches.append(em)
        f1_scores.append(f1)
    
    avg_em = sum(exact_matches) / len(exact_matches) * 100 if exact_matches else 0.0
    avg_f1 = sum(f1_scores) / len(f1_scores) * 100 if f1_scores else 0.0
    
    return {
        "exact_match": avg_em,
        "f1": avg_f1,
        "predictions": predictions,
    }

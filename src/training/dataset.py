"""
Dataset classes and preprocessing functions for Extractive Question-Answering (QA).

Cung cấp:
1. QAExample dataclass: Đại diện một mẫu QA
2. prepare_train_features: Chuyển raw QA data → tokenized features với token positions
3. prepare_eval_features: Chuẩn bị evaluation samples (không có labels)
4. Hỗ trợ Vietnamese word segmentation + sliding window xử lý long context
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, List
import logging

from .vietnamese_utils import VietnameseTextProcessor, align_segmentation_offset


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QAExample:
    """
    Một mẫu extractive QA (tương tự SQuAD format).

    Attributes:
        question: Câu hỏi
        context: Đoạn văn bản chứa câu trả lời
        answers: Dict với keys 'text' (list) và 'answer_start' (list)
        id: ID của mẫu (để tracking)
        is_impossible: Liệu câu hỏi có đáp án hay không
    """
    question: str
    context: str
    answers: dict[str, list[Any]]  # {"text": [...], "answer_start": [...]}
    id: str | None = None
    is_impossible: bool = False
    plausible_answers: Optional[dict[str, list[Any]]] = None


def prepare_train_features(
    examples: dict,
    tokenizer,
    question_column: str = "question",
    context_column: str = "context",
    answers_column: str = "answers",
    impossible_column: str = "is_impossible",
    max_length: int = 384,
    doc_stride: int = 128,
    padding: str = "max_length",
    use_vietnamese_segmentation: bool = True,
    segmentation_tool: str = "underthesea",
) -> dict:
    """
    Chuyển raw QA examples thành tokenized training features.

    Xử lý:
    1. Vietnamese word segmentation (tùy chọn)
    2. Tokenization với sliding window (cho long context)
    3. Character-to-token conversion để xác định answer span
    4. Handling unanswerable questions

    Args:
        examples: Dict chứa batch samples từ dataset
            - question: List[str]
            - context: List[str]
            - answers: List[Dict] với keys "text" và "answer_start"
            - is_impossible: List[bool] (optional)
        tokenizer: HuggingFace tokenizer (phải có offset_mapping support)
        question_column: Column name cho questions
        context_column: Column name cho contexts
        answers_column: Column name cho answers
        impossible_column: Column name cho is_impossible flag
        max_length: Max sequence length (384 hoặc 512)
        doc_stride: Stride cho sliding window xử lý long context
        padding: Cách padding ("max_length" hoặc "longest")
        use_vietnamese_segmentation: Có dùng Vietnamese word segmentation
        segmentation_tool: "underthesea" hoặc "pyvi"

    Returns:
        Dict với keys:
        - input_ids: Tokenized sequence IDs
        - attention_mask: Attention mask
        - start_positions: List token indices cho answer start
        - end_positions: List token indices cho answer end

    Lưu ý:
        - Unanswerable questions được set start/end = CLS token index
        - Answers bị truncate (cắt khỏi context) cũng được đánh dấu là unanswerable
    """

    # Initialize Vietnamese processor
    vi_processor = None
    if use_vietnamese_segmentation:
        try:
            vi_processor = VietnameseTextProcessor(segmentation_tool=segmentation_tool)
        except Exception as e:
            logger.warning(f"Failed to initialize Vietnamese processor: {e}. Skipping segmentation.")
            use_vietnamese_segmentation = False

    # Determine padding side
    pad_on_right = tokenizer.padding_side == "right"

    # Tạo bản mutable để có thể cập nhật context/answers sau khi align.
    questions = list(examples[question_column])
    contexts = list(examples[context_column])
    aligned_answers = [dict(a) if isinstance(a, dict) else a for a in examples[answers_column]]

    if use_vietnamese_segmentation and vi_processor:
        for idx in range(len(contexts)):
            raw_question = questions[idx]
            raw_context = contexts[idx]
            answers = examples[answers_column][idx]
            is_impossible = bool(examples[impossible_column][idx]) if impossible_column in examples else False

            seg_question = vi_processor.segment(raw_question)
            seg_context = vi_processor.segment(raw_context)

            # Mặc định: dùng text đã segment.
            questions[idx] = seg_question
            contexts[idx] = seg_context

            # Không cần align cho mẫu impossible/không có answers.
            if is_impossible or not isinstance(answers, dict) or len(answers.get("answer_start", [])) == 0:
                continue

            raw_answer_start = answers["answer_start"][0]
            raw_answer_text = answers["text"][0]

            new_answer_start, segmented_answer_text = align_segmentation_offset(
                raw_context=raw_context,
                raw_answer_text=raw_answer_text,
                raw_answer_start=raw_answer_start,
                segmented_context=seg_context,
            )

            if new_answer_start is None or segmented_answer_text is None:
                # Edge case: align fail -> fallback sample này về raw để không phá vỡ nhãn.
                logger.warning(
                    "Align offset thất bại ở sample idx=%s. Fallback sang raw context để giữ answer_start gốc.",
                    idx,
                )
                questions[idx] = raw_question
                contexts[idx] = raw_context
                continue

            # Cập nhật answers đã align cho sample hiện tại (ít nhất answer đầu tiên).
            updated_answers = dict(answers)
            updated_texts = list(answers.get("text", []))
            updated_starts = list(answers.get("answer_start", []))

            updated_texts[0] = segmented_answer_text
            updated_starts[0] = new_answer_start

            updated_answers["text"] = updated_texts
            updated_answers["answer_start"] = updated_starts
            aligned_answers[idx] = updated_answers

    # Tokenize with sliding window
    tokenized_examples = tokenizer(
        text=questions if pad_on_right else contexts,
        text_pair=contexts if pad_on_right else questions,
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=padding,
    )

    # Extract mappings
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    start_positions: list[int] = []
    end_positions: list[int] = []

    # Process each tokenized example
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(batch_index=i)

        # Get original sample
        sample_index = sample_mapping[i]
        answers = aligned_answers[sample_index]
        is_impossible = False

        if impossible_column in examples:
            is_impossible = bool(examples[impossible_column][sample_index])

        # Unanswerable case: set start/end to CLS token
        if is_impossible or len(answers.get("answer_start", [])) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        # Get answer position in original context
        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        # Find token indices for context/question parts
        token_start_index = 0
        while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
            token_end_index -= 1

        # Check if answer is completely within context tokens
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            # Answer is outside this window (truncated)
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        # Find answer start token
        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
            token_start_index += 1
        start_positions.append(token_start_index - 1)

        # Find answer end token
        while offsets[token_end_index][1] >= end_char:
            token_end_index -= 1
        end_positions.append(token_end_index + 1)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions

    return tokenized_examples


def prepare_eval_features(
    examples: dict,
    tokenizer,
    question_column: str = "question",
    context_column: str = "context",
    max_length: int = 384,
    doc_stride: int = 128,
    padding: str = "max_length",
    use_vietnamese_segmentation: bool = True,
    segmentation_tool: str = "underthesea",
) -> dict:
    """
    Chuẩn bị evaluation features (không tính toán answer positions).

    Dùng cho validation/test set để:
    - Lưu offset mappings cho post-processing
    - Không yêu cầu answer labels
    - Giữ nguyên sample IDs để so sánh với gold labels sau này

    Args:
        examples: Dict chứa batch samples (chỉ cần question + context)
        tokenizer: HuggingFace tokenizer
        question_column: Column name cho questions
        context_column: Column name cho contexts
        max_length: Max sequence length
        doc_stride: Stride cho sliding window
        padding: Cách padding
        use_vietnamese_segmentation: Có dùng Vietnamese segmentation
        segmentation_tool: "underthesea" hoặc "pyvi"

    Returns:
        Dict với keys:
        - input_ids, attention_mask, offset_mapping, sample_id (để tracking)
    """

    # Initialize Vietnamese processor
    vi_processor = None
    if use_vietnamese_segmentation:
        try:
            vi_processor = VietnameseTextProcessor(segmentation_tool=segmentation_tool)
        except Exception as e:
            logger.warning(f"Failed to initialize Vietnamese processor: {e}. Skipping segmentation.")
            use_vietnamese_segmentation = False

    pad_on_right = tokenizer.padding_side == "right"

    # Vietnamese segmentation
    questions = list(examples[question_column])
    contexts = list(examples[context_column])

    if use_vietnamese_segmentation and vi_processor:
        questions = [vi_processor.segment(q) for q in questions]
        contexts = [vi_processor.segment(c) for c in contexts]

    # Tokenize
    tokenized_examples = tokenizer(
        text=questions if pad_on_right else contexts,
        text_pair=contexts if pad_on_right else questions,
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=padding,
    )

    # Keep sample mapping và offset mapping cho post-processing
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples["offset_mapping"]

    # Chỉ giữ offset cho tokens thuộc context; token khác gán None.
    context_index = 1 if pad_on_right else 0
    for i in range(len(offset_mapping)):
        sequence_ids = tokenized_examples.sequence_ids(batch_index=i)
        offset_mapping[i] = [
            o if sequence_ids[k] == context_index else None
            for k, o in enumerate(offset_mapping[i])
        ]

    # Add sample IDs for later matching with predictions
    tokenized_examples["sample_id"] = sample_mapping

    return tokenized_examples


class QADataset:
    """
    PyTorch Dataset wrapper cho QA examples.

    Tùy chọn: Có thể được extend để support streaming từ HuggingFace Hub.
    """

    def __init__(
        self,
        examples: list[QAExample],
        tokenizer,
        max_length: int = 384,
        doc_stride: int = 128,
        is_training: bool = True,
        use_vietnamese_segmentation: bool = True,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.is_training = is_training
        self.use_vietnamese_segmentation = use_vietnamese_segmentation

        # Preprocess features
        self.features = self._preprocess_features()

    def _preprocess_features(self) -> list[dict]:
        """Preprocess all examples to features."""
        features = []

        for example in self.examples:
            examples_dict = {
                "question": [example.question],
                "context": [example.context],
                "answers": [example.answers],
                "is_impossible": [example.is_impossible],
            }

            if self.is_training:
                feature_dict = prepare_train_features(
                    examples_dict,
                    self.tokenizer,
                    max_length=self.max_length,
                    doc_stride=self.doc_stride,
                    use_vietnamese_segmentation=self.use_vietnamese_segmentation,
                )
            else:
                feature_dict = prepare_eval_features(
                    examples_dict,
                    self.tokenizer,
                    max_length=self.max_length,
                    doc_stride=self.doc_stride,
                    use_vietnamese_segmentation=self.use_vietnamese_segmentation,
                )

            # Convert batch of 1 to single sample features
            for key in feature_dict:
                if isinstance(feature_dict[key], list):
                    for idx, value in enumerate(feature_dict[key]):
                        if idx >= len(features):
                            features.append({})
                        features[idx][key] = value

        return features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict:
        return self.features[idx]

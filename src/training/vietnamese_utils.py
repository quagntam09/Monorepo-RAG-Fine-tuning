"""
Vietnamese language utilities for text processing and token mapping.

Module cung cấp:
1. Vietnamese word segmentation (RDRSegmenter từ underthesea)
2. Character-to-token position mapping để xác định answer span
3. Utility functions để xử lý Vietnamese text trong QA task
"""

from __future__ import annotations

from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


def align_segmentation_offset(
    raw_context: str,
    raw_answer_text: str,
    raw_answer_start: int,
    segmented_context: str,
) -> Tuple[Optional[int], Optional[str]]:
    """
    Căn chỉnh answer offset từ raw context sang segmented context.

    Bài toán:
    - `underthesea.word_tokenize(..., format="text")` chèn dấu `_` giữa từ ghép.
    - Vì vậy chỉ số ký tự `answer_start` trên raw context không còn đúng trên segmented context.
    - Hàm này tạo map ký tự raw -> segmented, bỏ qua ảnh hưởng của `_`.

    Trả về:
    - new_answer_start: vị trí bắt đầu answer trong segmented_context
    - segmented_answer_text: text answer tương ứng trong segmented_context

    Edge cases:
    - answer_start ngoài phạm vi
    - answer_text không khớp tại answer_start
    - không map được do text khác biệt bất thường sau segment
    """
    if not isinstance(raw_context, str) or not isinstance(segmented_context, str):
        return None, None
    if not isinstance(raw_answer_text, str) or raw_answer_text == "":
        return None, None
    if raw_answer_start < 0 or raw_answer_start >= len(raw_context):
        return None, None

    # Nếu answer_start sai nhưng answer_text có trong context, thử tìm lại vị trí gần nhất.
    expected_end = raw_answer_start + len(raw_answer_text)
    if expected_end > len(raw_context) or raw_context[raw_answer_start:expected_end] != raw_answer_text:
        recovered_start = raw_context.find(raw_answer_text)
        if recovered_start < 0:
            return None, None
        logger.warning(
            "answer_start mismatch được tự phục hồi: %s -> %s",
            raw_answer_start,
            recovered_start,
        )
        raw_answer_start = recovered_start
        expected_end = raw_answer_start + len(raw_answer_text)

    # Tạo chuỗi segmented "rút gọn" bỏ toàn bộ '_', đồng thời giữ map ngược về index thật.
    collapsed_chars: list[str] = []
    collapsed_to_segmented_idx: list[int] = []
    for seg_idx, ch in enumerate(segmented_context):
        if ch == "_":
            continue
        collapsed_chars.append(ch)
        collapsed_to_segmented_idx.append(seg_idx)
    collapsed_segmented = "".join(collapsed_chars)

    # Map từng ký tự raw_context sang index tương ứng trong segmented_context (thông qua collapsed).
    raw_to_segmented_idx: dict[int, int] = {}
    i = 0  # pointer raw_context
    j = 0  # pointer collapsed_segmented

    # Thuật toán two-pointers, ưu tiên match trực tiếp ký tự.
    # Vì ta chỉ bỏ '_' nên đa số ký tự sẽ khớp tuần tự.
    while i < len(raw_context) and j < len(collapsed_segmented):
        raw_ch = raw_context[i]
        seg_ch = collapsed_segmented[j]

        if raw_ch == seg_ch:
            raw_to_segmented_idx[i] = collapsed_to_segmented_idx[j]
            i += 1
            j += 1
            continue

        # Cho phép "điều chỉnh nhẹ" nếu có khác biệt khoảng trắng/căn lề sau segment.
        if raw_ch.isspace() and not seg_ch.isspace():
            i += 1
            continue
        if seg_ch.isspace() and not raw_ch.isspace():
            j += 1
            continue

        # Mismatch bất thường: thử dịch cục bộ 1 ký tự ở collapsed.
        if j + 1 < len(collapsed_segmented) and raw_ch == collapsed_segmented[j + 1]:
            j += 1
            continue

        # Mismatch bất thường: thử dịch cục bộ 1 ký tự ở raw.
        if i + 1 < len(raw_context) and raw_context[i + 1] == seg_ch:
            i += 1
            continue

        # Không xử lý được mismatch -> fail để caller fallback.
        return None, None

    raw_answer_end = expected_end - 1
    if raw_answer_start not in raw_to_segmented_idx or raw_answer_end not in raw_to_segmented_idx:
        return None, None

    new_answer_start = raw_to_segmented_idx[raw_answer_start]
    new_answer_end = raw_to_segmented_idx[raw_answer_end]
    if new_answer_end < new_answer_start:
        return None, None

    segmented_answer_text = segmented_context[new_answer_start:new_answer_end + 1]
    if segmented_answer_text == "":
        return None, None

    return new_answer_start, segmented_answer_text


class VietnameseTextProcessor:
    """
    Xử lý text tiếng Việt: segmentation + character-to-token mapping.

    Purpose:
        - Word segmentation để tối ưu tokenization
        - Ánh xạ vị trí ký tự sang vị trí token cho answer span prediction
    """

    def __init__(self, segmentation_tool: str = "underthesea"):
        """
        Args:
            segmentation_tool: "underthesea" (RDRSegmenter) hoặc "pyvi" (ViTokenizer)
        """
        self.segmentation_tool = segmentation_tool
        self._segmenter = None

        if segmentation_tool == "underthesea":
            try:
                from underthesea import word_tokenize
            except ImportError as exc:
                raise ImportError("underthesea không được cài đặt. Cài đặt: pip install underthesea") from exc
            self._segmenter = word_tokenize

        elif segmentation_tool == "pyvi":
            try:
                from pyvi import ViTokenizer
            except ImportError as exc:
                raise ImportError("pyvi không được cài đặt. Cài đặt: pip install pyvi") from exc
            self._segmenter = ViTokenizer.tokenize

    def segment(self, text: str) -> str:
        """
        Thực hiện word segmentation trên text tiếng Việt.

        Args:
            text: Raw Vietnamese text

        Returns:
            Text với word boundaries được đánh dấu (từ được cách nhau bằng dấu cách)

        Example:
            >>> processor = VietnameseTextProcessor()
            >>> processor.segment("Tôi yêu Việt Nam")
            "Tôi yêu Việt Nam"
        """
        if not text or not isinstance(text, str):
            return text

        try:
            if self.segmentation_tool == "underthesea":
                # format="text" để underthesea trả về string có '_' giữa từ ghép
                # VD: "thành phố" -> "thành_phố"
                return self._segmenter(text=text, format="text")

            elif self.segmentation_tool == "pyvi":
                # ViTokenizer
                return self._segmenter(text)

            else:
                logger.warning(f"Unknown segmentation tool: {self.segmentation_tool}")
                return text

        except Exception as e:
            logger.warning(f"Segmentation failed: {e}. Returning original text.")
            return text

    def get_char_to_token_mapping(
        self,
        text: str,
        tokenizer,
        max_length: int = 512,
        add_special_tokens: bool = True,
    ) -> dict:
        """
        Tạo mapping từ character position sang token position.

        Được dùng để:
        - Chuyển đổi answer_start (char index) → token index
        - Xác định token position của câu trả lời

        Args:
            text: Raw Vietnamese text
            tokenizer: HuggingFace tokenizer (với offset_mapping)
            max_length: Max sequence length
            add_special_tokens: Có thêm special tokens ([CLS], [SEP], etc.)

        Returns:
            Dict chứa:
                - "tokens": list các tokens
                - "token_ids": list các token IDs
                - "offset_mapping": list các (char_start, char_end) cho mỗi token
                - "char_to_token": mapping từ char index → token index
        """
        # Segment Vietnamese text
        segmented_text = self.segment(text)

        # Tokenize với offset_mapping
        encoding = tokenizer(
            segmented_text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
            add_special_tokens=add_special_tokens,
        )

        offset_mapping = encoding.get("offset_mapping", [])
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

        # Tạo char_to_token mapping
        char_to_token = {}
        for token_idx, (char_start, char_end) in enumerate(offset_mapping):
            if char_start == char_end == 0:  # Special tokens
                continue
            for char_idx in range(char_start, char_end):
                char_to_token[char_idx] = token_idx

        return {
            "tokens": tokens,
            "token_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "offset_mapping": offset_mapping,
            "char_to_token": char_to_token,
        }

    def find_answer_span_tokens(
        self,
        context: str,
        answer_text: str,
        answer_start: int,
        tokenizer,
        max_length: int = 512,
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Tìm token positions (start_token, end_token) của answer trong context.

        Giải quyết vấn đề:
        - Vietnamese text sau segmentation thay đổi độ dài
        - Character indices cần được mapping sang token indices
        - Xử lý trường hợp answer bị truncate hoặc không tìm thấy

        Args:
            context: Full context text (Vietnamese)
            answer_text: Answer text (substring của context)
            answer_start: Character position của answer trong context (0-indexed)
            tokenizer: HuggingFace tokenizer
            max_length: Max sequence length

        Returns:
            (start_token_idx, end_token_idx) hoặc (None, None) nếu không tìm thấy

        Example:
            >>> start, end = processor.find_answer_span_tokens(
            ...     context="Hà Nội là thủ đô của Việt Nam",
            ...     answer_text="thủ đô",
            ...     answer_start=13,
            ...     tokenizer=tokenizer
            ... )
        """
        # Validate answer position
        if answer_start + len(answer_text) > len(context):
            logger.warning(
                f"Answer out of bounds: answer_start={answer_start}, "
                f"answer_len={len(answer_text)}, context_len={len(context)}"
            )
            return None, None

        # Verify answer matches context
        if context[answer_start : answer_start + len(answer_text)] != answer_text:
            logger.warning(
                f"Answer text mismatch at position {answer_start}. "
                f"Expected: '{answer_text}', Got: '{context[answer_start : answer_start + len(answer_text)]}'"
            )
            return None, None

        # Segment and get char-to-token mapping
        segmented_context = self.segment(context)

        # Tokenize
        encoding = tokenizer(
            segmented_context,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
            add_special_tokens=True,
        )

        offset_mapping = encoding.get("offset_mapping", [])

        # Find token span
        answer_char_start = answer_start
        answer_char_end = answer_start + len(answer_text)

        start_token_idx = None
        end_token_idx = None

        for token_idx, (char_start, char_end) in enumerate(offset_mapping):
            if char_start == char_end == 0:  # Special token
                continue

            # Start token: first token overlapping with answer start
            if start_token_idx is None and char_end > answer_char_start:
                start_token_idx = token_idx

            # End token: last token overlapping with answer end
            if char_start < answer_char_end:
                end_token_idx = token_idx

            # No need to continue if we've passed the answer
            if char_start >= answer_char_end:
                break

        # Validate found span
        if start_token_idx is None or end_token_idx is None:
            logger.warning(
                f"Could not map answer span. "
                f"start_token={start_token_idx}, end_token={end_token_idx}"
            )
            return None, None

        if start_token_idx > end_token_idx:
            logger.warning(
                f"Invalid token span: start={start_token_idx} > end={end_token_idx}"
            )
            return None, None

        return start_token_idx, end_token_idx

    def extract_answer_from_tokens(
        self,
        tokens: List[str],
        start_token_idx: int,
        end_token_idx: int,
        tokenizer,
    ) -> str:
        """
        Trích xuất answer text từ token indices.

        Đảo ngược quá trình tokenization để lấy lại text từ tokens.

        Args:
            tokens: List các tokens
            start_token_idx: Start token index
            end_token_idx: End token index (inclusive)
            tokenizer: HuggingFace tokenizer

        Returns:
            Answer text được reconstruct từ tokens
        """
        if start_token_idx > end_token_idx or start_token_idx < 0:
            return ""

        # Giới hạn indices
        end_token_idx = min(end_token_idx, len(tokens) - 1)

        # Lấy tokens và convert back to string
        answer_tokens = tokens[start_token_idx : end_token_idx + 1]

        # Remove special tokens
        answer_tokens = [
            token for token in answer_tokens
            if token not in [tokenizer.cls_token, tokenizer.sep_token,
                            tokenizer.pad_token, "[CLS]", "[SEP]", "[PAD]"]
        ]

        # Convert subword tokens to text
        answer_text = tokenizer.convert_tokens_to_string(answer_tokens)

        return answer_text.strip()

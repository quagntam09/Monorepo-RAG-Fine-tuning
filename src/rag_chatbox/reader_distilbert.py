from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


@dataclass(frozen=True)
class ReaderAnswer:
    answer: str
    span_score: float
    start_score: float
    end_score: float
    source: str
    page: str
    context: str


def select_best_reader_answer(
    ranked_answers: list[ReaderAnswer],
    min_span_score: float,
) -> ReaderAnswer | None:
    """Return the highest-ranked non-empty answer that clears the score floor."""
    for candidate in ranked_answers:
        if candidate.span_score < min_span_score:
            continue
        if (candidate.answer or "").strip():
            return candidate
    return None


class DistilBertOnnxReader:
    def __init__(
        self,
        artifact_dir: str,
        max_length: int = 384,
        max_answer_length: int = 30,
        n_best_size: int = 20,
        require_metadata: bool = True,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.max_length = max_length
        self.max_answer_length = max_answer_length
        self.n_best_size = n_best_size

        self.model_path = validate_reader_artifact(self.artifact_dir, require_metadata=require_metadata)

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.artifact_dir))
        self.session = ort.InferenceSession(str(self.model_path))
        self.input_names = {inp.name for inp in self.session.get_inputs()}

    def _build_input_feed(self, encoding: Any) -> dict[str, np.ndarray]:
        input_feed: dict[str, np.ndarray] = {}

        if "input_ids" not in self.input_names or "attention_mask" not in self.input_names:
            raise ValueError(f"Unsupported ONNX inputs: {sorted(self.input_names)}")

        input_ids = encoding["input_ids"].astype(np.int64)
        attention_mask = encoding["attention_mask"].astype(np.int64)
        input_feed["input_ids"] = input_ids
        input_feed["attention_mask"] = attention_mask

        if "token_type_ids" in self.input_names:
            if "token_type_ids" in encoding:
                input_feed["token_type_ids"] = encoding["token_type_ids"].astype(np.int64)
            else:
                input_feed["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

        return input_feed

    def _extract_start_end_logits(self, outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        output_names = [out.name.lower() for out in self.session.get_outputs()]

        start_idx = next((idx for idx, name in enumerate(output_names) if "start" in name), 0)
        end_idx = next((idx for idx, name in enumerate(output_names) if "end" in name), 1 if len(outputs) > 1 else 0)

        start_logits = outputs[start_idx]
        end_logits = outputs[end_idx]
        return start_logits, end_logits

    def _best_span_from_logits(
        self,
        start_logits: np.ndarray,
        end_logits: np.ndarray,
        offset_mapping: np.ndarray,
        sequence_ids: list[int | None],
        context: str,
    ) -> tuple[str, float, float, float]:
        start_indexes = np.argsort(start_logits)[-self.n_best_size :][::-1]
        end_indexes = np.argsort(end_logits)[-self.n_best_size :][::-1]

        best_span = None
        best_score = float("-inf")
        for start_idx in start_indexes:
            for end_idx in end_indexes:
                if end_idx < start_idx:
                    continue
                if end_idx - start_idx + 1 > self.max_answer_length:
                    continue
                if sequence_ids[start_idx] != 1 or sequence_ids[end_idx] != 1:
                    continue

                start_char, _ = offset_mapping[start_idx]
                _, end_char = offset_mapping[end_idx]
                if end_char <= start_char:
                    continue

                score = float(start_logits[start_idx] + end_logits[end_idx])
                if score > best_score:
                    best_score = score
                    best_span = (int(start_idx), int(end_idx), int(start_char), int(end_char))

        if best_span is None:
            start_idx = int(start_logits.argmax())
            end_idx = start_idx
            return (
                "",
                float(start_logits[start_idx] + end_logits[end_idx]),
                float(start_logits[start_idx]),
                float(end_logits[end_idx]),
            )

        start_idx, end_idx, start_char, end_char = best_span
        answer_text = context[start_char:end_char].strip()
        return answer_text, best_score, float(start_logits[start_idx]), float(end_logits[end_idx])

    def _run_pairs(self, questions: list[str], contexts: list[str]) -> list[tuple[str, float, float, float]]:
        if len(questions) != len(contexts):
            raise ValueError("questions and contexts must have the same length")
        if not questions:
            return []

        encoding = self.tokenizer(
            questions,
            contexts,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="np",
        )

        outputs = self.session.run(None, self._build_input_feed(encoding))
        start_logits, end_logits = self._extract_start_end_logits(outputs)

        results: list[tuple[str, float, float, float]] = []
        for idx, context in enumerate(contexts):
            results.append(
                self._best_span_from_logits(
                    start_logits=start_logits[idx],
                    end_logits=end_logits[idx],
                    offset_mapping=encoding["offset_mapping"][idx],
                    sequence_ids=encoding.sequence_ids(idx),
                    context=context,
                )
            )
        return results

    def _run_single(self, question: str, context: str) -> tuple[str, float, float, float]:
        return self._run_pairs([question], [context])[0]

    def answer_on_question_context_pairs(
        self,
        questions: list[str],
        contexts: list[str],
    ) -> list[tuple[str, float, float, float]]:
        return self._run_pairs(questions, contexts)

    def answer_on_documents(self, question: str, docs: list[Any]) -> list[ReaderAnswer]:
        ranked: list[ReaderAnswer] = []
        use_batch = hasattr(self, "tokenizer") and hasattr(self, "session")
        if use_batch:
            contexts = [doc.page_content or "" for doc in docs]
            predictions = self._run_pairs([question] * len(contexts), contexts) if contexts else []
        else:
            predictions = [self._run_single(question=question, context=(doc.page_content or "")) for doc in docs]

        for doc, (answer, span_score, start_score, end_score) in zip(docs, predictions):
            context = doc.page_content or ""
            source = str(doc.metadata.get("source", "Unknown"))
            raw_page = doc.metadata.get("page_number", doc.metadata.get("page"))
            if isinstance(raw_page, int):
                page = str(raw_page + 1)
            elif raw_page is None:
                page = "N/A"
            else:
                page = str(raw_page)

            ranked.append(
                ReaderAnswer(
                    answer=answer,
                    span_score=span_score,
                    start_score=start_score,
                    end_score=end_score,
                    source=source,
                    page=page,
                    context=context,
                )
            )

        ranked.sort(key=lambda x: x.span_score, reverse=True)
        return ranked


def validate_reader_artifact(artifact_dir: str | Path, require_metadata: bool = True) -> Path:
    resolved_dir = Path(artifact_dir)
    if not resolved_dir.exists():
        raise FileNotFoundError(f"Reader artifact_dir not found: {resolved_dir}")
    if not resolved_dir.is_dir():
        raise NotADirectoryError(f"Reader artifact_dir must be a directory: {resolved_dir}")

    model_quantized = resolved_dir / "model_quantized.onnx"
    model_default = resolved_dir / "model.onnx"
    if model_quantized.exists():
        model_path = model_quantized
    elif model_default.exists():
        model_path = model_default
    else:
        raise FileNotFoundError(f"ONNX file not found in: {resolved_dir}")

    required_files = [
        resolved_dir / "tokenizer.json",
        resolved_dir / "tokenizer_config.json",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Reader artifact is missing required tokenizer files: {', '.join(missing)}"
        )

    metadata_path = resolved_dir / "model_metadata.json"
    if not metadata_path.exists():
        if require_metadata:
            raise FileNotFoundError(
                f"Missing metadata file: {metadata_path}. Re-export ONNX to generate metadata."
            )
        print(f"[WARN] Missing metadata file: {metadata_path}")
        return model_path

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Metadata file is not valid JSON: {metadata_path}") from exc

    required_metadata_keys = {"artifact_name", "artifact_type", "export_time_utc", "onnx", "tokenizer"}
    missing_keys = sorted(required_metadata_keys - set(metadata.keys()))
    if missing_keys:
        raise ValueError(f"Metadata file is missing required keys: {missing_keys}")
    if metadata.get("artifact_type") != "distilbert_onnx_reader":
        raise ValueError(f"Unsupported artifact_type: {metadata.get('artifact_type')!r}")

    def _sha256_file(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    onnx_metadata = metadata.get("onnx", {})
    expected_hash = None
    if model_path.name == "model_quantized.onnx":
        expected_hash = onnx_metadata.get("quantized_model_sha256")
    else:
        expected_hash = onnx_metadata.get("model_sha256")
    if expected_hash:
        actual_hash = _sha256_file(model_path)
        if actual_hash.lower() != str(expected_hash).lower():
            raise ValueError(
                f"ONNX checksum mismatch for {model_path.name}: expected {expected_hash}, got {actual_hash}"
            )

    tokenizer_metadata = metadata.get("tokenizer", {})
    tokenizer_json_path = resolved_dir / "tokenizer.json"
    tokenizer_config_path = resolved_dir / "tokenizer_config.json"
    for path, expected in [
        (tokenizer_json_path, tokenizer_metadata.get("tokenizer_json_sha256")),
        (tokenizer_config_path, tokenizer_metadata.get("tokenizer_config_sha256")),
    ]:
        if expected:
            actual_hash = _sha256_file(path)
            if actual_hash.lower() != str(expected).lower():
                raise ValueError(
                    f"Tokenizer checksum mismatch for {path.name}: expected {expected}, got {actual_hash}"
                )

    return model_path

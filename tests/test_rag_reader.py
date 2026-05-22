from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from rag_chatbox.reader_distilbert import (
    DistilBertOnnxReader,
    ReaderAnswer,
    select_best_reader_answer,
    validate_reader_artifact,
)


class TestReaderValidation(unittest.TestCase):
    def test_validate_reader_artifact_requires_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "model.onnx").write_bytes(b"onnx")
            (root / "tokenizer.json").write_text("{}", encoding="utf-8")
            (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                validate_reader_artifact(root, require_metadata=True)

            selected = validate_reader_artifact(root, require_metadata=False)
            self.assertEqual(selected.name, "model.onnx")

    def test_validate_reader_artifact_with_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "model_quantized.onnx").write_bytes(b"onnx")
            (root / "tokenizer.json").write_text("{}", encoding="utf-8")
            (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (root / "model_metadata.json").write_text(
                json.dumps(
                    {
                        "artifact_name": "run_best",
                        "artifact_type": "distilbert_onnx_reader",
                        "export_time_utc": "2026-05-14T00:00:00Z",
                        "onnx": {},
                        "tokenizer": {},
                    }
                ),
                encoding="utf-8",
            )

            selected = validate_reader_artifact(root, require_metadata=True)
            self.assertEqual(selected.name, "model_quantized.onnx")


class TestReaderAnswers(unittest.TestCase):
    def test_answer_on_documents_sorted_desc(self):
        reader = DistilBertOnnxReader.__new__(DistilBertOnnxReader)

        def fake_run_single(question: str, context: str):
            if "A" in context:
                return "answer-a", 1.0, 0.5, 0.5
            return "answer-b", 2.0, 1.0, 1.0

        reader._run_single = fake_run_single

        docs = [
            Document(page_content="context A", metadata={"source": "a.pdf", "page": 0}),
            Document(page_content="context B", metadata={"source": "b.pdf", "page": 1}),
        ]

        ranked = reader.answer_on_documents(question="q", docs=docs)
        self.assertEqual(ranked[0].answer, "answer-b")
        self.assertEqual(ranked[0].source, "b.pdf")

    def test_select_best_reader_answer_skips_empty_top_span(self):
        ranked_answers = [
            ReaderAnswer(
                answer="",
                span_score=3.0,
                start_score=1.5,
                end_score=1.5,
                source="a.pdf",
                page="1",
                context="context a",
            ),
            ReaderAnswer(
                answer="valid",
                span_score=2.5,
                start_score=1.2,
                end_score=1.3,
                source="b.pdf",
                page="2",
                context="context b",
            ),
        ]

        selected = select_best_reader_answer(ranked_answers, min_span_score=0.0)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.answer, "valid")
        self.assertEqual(selected.source, "b.pdf")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from rag_chatbox.config import AppConfig
from rag_chatbox.evaluate_eval_set import evaluate_eval_set
from rag_chatbox.reader_distilbert import ReaderAnswer


class FakeReader:
    def answer_on_documents(self, question, docs):
        return [
            ReaderAnswer(
                answer="Hà Nội",
                span_score=1.0,
                start_score=0.5,
                end_score=0.5,
                source="demo.pdf",
                page="1",
                context=docs[0].page_content,
            )
        ]


class FakeChatbot:
    def answer(self, question: str) -> str:
        return "Hà Nội\n\nNguồn:\n- demo.pdf (Page 1)"


class TestEvaluateEvalSet(unittest.TestCase):
    def test_rag_eval_aggregates_metrics(self):
        docs = [Document(page_content="Hà Nội là thủ đô của Việt Nam.", metadata={"source": "demo.pdf", "page": 0})]

        with tempfile.TemporaryDirectory() as tmpdir:
            eval_file = Path(tmpdir) / "questions.jsonl"
            eval_file.write_text(
                json.dumps(
                    {
                        "id": "q1",
                        "question": "Thủ đô của Việt Nam là gì?",
                        "expected_answer": "Hà Nội",
                        "expected_sources": [{"source": "demo.pdf", "page": 1}],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("rag_chatbox.evaluate_eval_set._build_components", return_value=(lambda _q: docs, FakeReader(), FakeChatbot())):
                result = evaluate_eval_set(
                    config=AppConfig(reader_min_span_score=0.0),
                    eval_file=eval_file,
                    mode="rag",
                    top_k=5,
                )

        self.assertEqual(result["summary"]["samples"], 1)
        self.assertEqual(result["summary"]["retrieval_hit_rate"], 1.0)
        self.assertEqual(result["summary"]["reader_exact_match"], 1.0)
        self.assertEqual(result["summary"]["rag_exact_match"], 1.0)
        self.assertEqual(result["summary"]["citation_hit_rate"], 1.0)
        self.assertEqual(result["rows"][0]["rag_citations"], ["demo.pdf (page 1)"])


if __name__ == "__main__":
    unittest.main()

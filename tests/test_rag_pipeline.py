from __future__ import annotations

import unittest

from langchain_core.documents import Document

from rag_chatbox.config import AppConfig
from rag_chatbox.rag_pipeline import _finalize_answer, _prepare_chain_payload
from rag_chatbox.reader_distilbert import ReaderAnswer


class FakeReader:
    def __init__(self, answer: str, score: float):
        self.answer = answer
        self.score = score

    def answer_on_documents(self, question, docs):
        return [
            ReaderAnswer(
                answer=self.answer,
                span_score=self.score,
                start_score=self.score / 2,
                end_score=self.score / 2,
                source="demo.pdf",
                page="1",
                context=docs[0].page_content,
            )
        ]


class TestPrepareChainPayload(unittest.TestCase):
    def test_no_docs_returns_idk(self):
        config = AppConfig(reader_min_span_score=0.0)

        payload = _prepare_chain_payload(
            question="what",
            chat_history="",
            retriever=lambda _q: [],
            reader=FakeReader(answer="x", score=1.0),
            config=config,
        )

        self.assertEqual(payload["reader_answer"], "I don't know.")
        self.assertEqual(payload["reader_span_score"], "0.0000")
        self.assertEqual(payload["allowed_sources"], [])

    def test_docs_with_reader_answer(self):
        docs = [Document(page_content="The answer is candidate.", metadata={"source": "demo.pdf", "page": 0})]
        config = AppConfig(reader_min_span_score=0.0)

        payload = _prepare_chain_payload(
            question="what",
            chat_history="",
            retriever=lambda _q: docs,
            reader=FakeReader(answer="candidate", score=1.0),
            config=config,
        )

        self.assertEqual(payload["reader_answer"], "candidate")
        self.assertEqual(payload["reader_span_score"], "1.0000")
        self.assertEqual(payload["allowed_sources"][0]["label"], "demo.pdf (Page 1)")

    def test_finalize_answer_returns_grounded_sources(self):
        final = _finalize_answer(
            answer="candidate\n\nNguồn:\n- demo.pdf (Page 1)\n- fake.pdf (Page 9)",
            allowed_sources=[{"source": "demo.pdf", "page": 1, "label": "demo.pdf (Page 1)"}],
        )

        self.assertIn("demo.pdf (Page 1)", final)
        self.assertIn("candidate", final)
        self.assertNotIn("fake.pdf", final)

    def test_finalize_answer_adds_sources_when_missing(self):
        final = _finalize_answer(
            answer="candidate",
            allowed_sources=[{"source": "demo.pdf", "page": 1, "label": "demo.pdf (Page 1)"}],
        )

        self.assertIn("Nguồn:", final)
        self.assertEqual(final, "candidate\n\nNguồn:\n- Không có trích dẫn hợp lệ trong context.")


if __name__ == "__main__":
    unittest.main()

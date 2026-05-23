from __future__ import annotations

import unittest

from langchain_core.documents import Document

from rag_chatbox.config import AppConfig
from rag_chatbox.retrieval import _make_retrieval_fn, _parse_rewritten_queries


class FakeVectorstore:
    def __init__(self, candidates: list[tuple[Document, float]] | None = None, query_map: dict[str, list[tuple[Document, float]]] | None = None):
        self.candidates = candidates or []
        self.query_map = query_map or {}
        self.calls: list[str] = []

    def similarity_search_with_relevance_scores(self, question: str, k: int):
        self.calls.append(question)
        if question in self.query_map:
            return self.query_map[question][:k]
        return self.candidates[:k]


class TestRetrievalThreshold(unittest.TestCase):
    def test_returns_empty_when_below_threshold(self):
        doc = Document(page_content="irrelevant", metadata={"source": "demo.pdf", "page": 0})
        retriever = _make_retrieval_fn(
            FakeVectorstore([(doc, 0.1)]),
            AppConfig(score_threshold=0.5, top_k=5, fetch_k=5),
        )

        results = retriever("question")
        self.assertEqual(results, [])

    def test_returns_docs_above_threshold(self):
        doc = Document(page_content="relevant", metadata={"source": "demo.pdf", "page": 0})
        retriever = _make_retrieval_fn(
            FakeVectorstore([(doc, 0.9)]),
            AppConfig(score_threshold=0.5, top_k=5, fetch_k=5),
        )

        results = retriever("question")
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0].metadata["rerank_score"], 0.72)

    def test_query_rewrite_expands_retrieval_space(self):
        doc_origin = Document(page_content="alpha content", metadata={"source": "a.pdf", "page": 0})
        doc_rewrite = Document(page_content="beta content", metadata={"source": "b.pdf", "page": 1})
        vectorstore = FakeVectorstore(
            query_map={
                "question": [(doc_origin, 0.55)],
                "question rewrite": [(doc_rewrite, 0.95)],
            }
        )
        retriever = _make_retrieval_fn(
            vectorstore,
            AppConfig(score_threshold=0.5, top_k=5, fetch_k=5, query_rewrite_max_variants=2),
            query_rewriter=lambda _q: ["question", "question rewrite"],
        )

        results = retriever("question")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].metadata["source"], "b.pdf")
        self.assertIn("question rewrite", vectorstore.calls)
        self.assertIsNotNone(retriever.last_trace)
        self.assertEqual(retriever.last_trace["decision"], "selected_top_k_by_rerank")
        self.assertIn("chunk_preview", retriever.last_trace["selected"][0])

    def test_retrieval_cache_reuses_previous_result(self):
        doc = Document(page_content="relevant", metadata={"source": "demo.pdf", "page": 0})
        vectorstore = FakeVectorstore([(doc, 0.9)])
        retriever = _make_retrieval_fn(
            vectorstore,
            AppConfig(score_threshold=0.5, top_k=5, fetch_k=5, retrieval_cache_ttl_sec=30.0),
        )

        first = retriever("question")
        second = retriever("question")
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertEqual(len(vectorstore.calls), 1)
        self.assertTrue(retriever.last_trace["retrieval_cache_hit"])

    def test_query_rewrite_cache_avoids_rewriter_recompute(self):
        doc = Document(page_content="alpha content", metadata={"source": "a.pdf", "page": 0})
        vectorstore = FakeVectorstore(query_map={"question": [(doc, 0.9)]})
        calls = {"rewriter": 0}

        def query_rewriter(_q: str):
            calls["rewriter"] += 1
            return ["question"]

        retriever = _make_retrieval_fn(
            vectorstore,
            AppConfig(
                score_threshold=0.5,
                top_k=5,
                fetch_k=5,
                query_rewrite_max_variants=2,
                query_rewrite_cache_ttl_sec=30.0,
                retrieval_cache_ttl_sec=0.0,
            ),
            query_rewriter=query_rewriter,
        )

        retriever("question")
        retriever("question")
        self.assertEqual(calls["rewriter"], 1)


class TestQueryRewriteParsing(unittest.TestCase):
    def test_parse_json_array(self):
        parsed = _parse_rewritten_queries('["q1", "q2", "q1"]', max_variants=3)
        self.assertEqual(parsed, ["q1", "q2"])

    def test_parse_markdown_lines(self):
        raw = "- rewrite one\n- rewrite two\n- rewrite one"
        parsed = _parse_rewritten_queries(raw, max_variants=5)
        self.assertEqual(parsed, ["rewrite one", "rewrite two"])


if __name__ == "__main__":
    unittest.main()

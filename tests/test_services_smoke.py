from __future__ import annotations

import unittest


class TestServiceImports(unittest.TestCase):
    def test_reader_service_imports(self):
        from rag_chatbox.services.reader_service import app

        self.assertEqual(app.title, "RAG Reader ONNX Service")

    def test_synthesis_service_imports(self):
        from rag_chatbox.services.synthesis_service import app

        self.assertEqual(app.title, "RAG LLM Synthesis Service")


if __name__ == "__main__":
    unittest.main()

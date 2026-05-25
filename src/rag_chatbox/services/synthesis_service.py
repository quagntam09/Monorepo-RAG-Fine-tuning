from __future__ import annotations

import argparse
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Sequence

import httpx
from fastapi import FastAPI, HTTPException, Request
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from pydantic import BaseModel, Field

from rag_chatbox.config import load_config
from rag_chatbox.ingestion import load_documents, split_documents
from rag_chatbox.prompt_template import PROMPT_TEMPLATE
from rag_chatbox.rag_pipeline import _finalize_answer, _format_allowed_sources, _split_answer_and_sources
from rag_chatbox.retrieval import build_retriever, format_docs_with_metadata

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    question: str
    chat_history: str = ""


class ChatResponse(BaseModel):
    answer: str
    retrieval_sources: list[str] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


def _page_label(doc: Any) -> str:
    source = str(doc.metadata.get("source", "Unknown"))
    raw_page = doc.metadata.get("page_number", doc.metadata.get("page", "N/A"))
    if isinstance(raw_page, int):
        page = raw_page + 1
    else:
        page = raw_page
    return f"{source} (Page {page})"


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config = load_config()
        documents = load_documents(config)
        chunks = split_documents(documents, config)
        retriever = build_retriever(documents, chunks, config)

        app.state.config = config
        app.state.retriever = retriever
        app.state.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        app.state.llm = OllamaLLM(
            model=config.llm_model,
            temperature=config.temperature,
            base_url=config.ollama_base_url,
        )
        app.state.reader_service_url = os.getenv("RAG_READER_SERVICE_URL", "http://localhost:8081").rstrip("/")
        app.state.timeout_sec = float(os.getenv("RAG_READER_SERVICE_TIMEOUT_SEC", "30"))
        yield

    app = FastAPI(title="RAG LLM Synthesis Service", version="1.0.0", lifespan=lifespan)
    app.state.config = None
    app.state.retriever = None
    app.state.prompt = None
    app.state.llm = None
    app.state.reader_service_url = None
    app.state.timeout_sec = None

    @app.get("/healthz")
    async def healthz(request: Request) -> dict[str, Any]:
        config = request.app.state.config
        return {
            "status": "ok",
            "reader_service_url": request.app.state.reader_service_url,
            "llm_model": getattr(config, "llm_model", None),
            "embedding_model": getattr(config, "embedding_model", None),
        }

    @app.post("/v1/chat/ask", response_model=ChatResponse)
    async def ask(request: ChatRequest, http_request: Request) -> ChatResponse:
        config = http_request.app.state.config
        retriever = http_request.app.state.retriever
        prompt = http_request.app.state.prompt
        llm = http_request.app.state.llm
        reader_service_url = http_request.app.state.reader_service_url
        timeout_sec = http_request.app.state.timeout_sec

        if config is None or retriever is None or prompt is None or llm is None or reader_service_url is None:
            raise HTTPException(status_code=503, detail="synthesis service is not ready")

        question = (request.question or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="question must not be empty")

        docs = retriever(question)
        retrieval_trace = getattr(retriever, "last_trace", None)
        retrieval_sources = [_page_label(doc) for doc in docs]

        if not docs:
            answer = "Mình không biết.\n\nNguồn:\n- Không có trích dẫn hợp lệ trong context."
            return ChatResponse(
                answer=answer,
                retrieval_sources=[],
                debug={
                    "retrieval": retrieval_trace,
                    "generation": {"llm_called": False, "decision": "no_context_retrieved"},
                },
            )

        payload_docs = []
        for doc in docs:
            source = str(doc.metadata.get("source", "Unknown"))
            raw_page = doc.metadata.get("page_number", doc.metadata.get("page", "N/A"))
            page = raw_page + 1 if isinstance(raw_page, int) else raw_page
            payload_docs.append(
                {
                    "content": doc.page_content,
                    "source": source,
                    "page": page,
                }
            )

        reader_request = {
            "question": question,
            "documents": payload_docs,
            "min_span_score": float(config.reader_min_span_score),
            "n_best": int(min(5, max(1, config.reader_n_best_size))),
        }

        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            response = await client.post(f"{reader_service_url}/v1/reader/answers", json=reader_request)
        if response.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"reader service failed: {response.text}")
        reader_result = response.json()

        allowed_sources = _format_allowed_sources(docs)
        context_block = format_docs_with_metadata(docs)
        chain_payload = {
            "context": context_block,
            "reader_answer": reader_result.get("selected_answer", "I don't know."),
            "reader_span_score": f"{float(reader_result.get('selected_span_score', 0.0)):.4f}",
            "chat_history": request.chat_history,
            "question": question,
        }

        raw_answer = await asyncio.to_thread(lambda: (prompt | llm | StrOutputParser()).invoke(chain_payload))
        final_answer = _finalize_answer(raw_answer, allowed_sources)
        answer_body, answer_sources = _split_answer_and_sources(final_answer)

        return ChatResponse(
            answer=final_answer,
            retrieval_sources=retrieval_sources,
            debug={
                "retrieval": retrieval_trace,
                "reader": reader_result,
                "generation": {
                    "llm_called": True,
                    "final_answer_preview": answer_body[:200],
                    "final_sources": answer_sources,
                },
            },
        )

    return app


app = create_app()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM synthesis service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    import uvicorn

    uvicorn.run("rag_chatbox.services.synthesis_service:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from dataclasses import dataclass
from contextlib import asynccontextmanager
from typing import Any, Sequence

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from rag_chatbox.config import load_config
from rag_chatbox.reader_distilbert import DistilBertOnnxReader, ReaderAnswer, select_best_reader_answer

logger = logging.getLogger(__name__)


class ReaderDocument(BaseModel):
    content: str
    source: str = "Unknown"
    page: str | int | None = None


class ReaderRequest(BaseModel):
    question: str
    documents: list[ReaderDocument] = Field(default_factory=list)
    min_span_score: float = 0.0
    n_best: int = 5


class ReaderCandidate(BaseModel):
    source: str
    page: str
    answer: str
    span_score: float
    start_score: float
    end_score: float


class ReaderResponse(BaseModel):
    selected_answer: str
    selected_span_score: float
    selection_reason: str
    candidates: list[ReaderCandidate]


@dataclass
class _PendingRequest:
    request: ReaderRequest
    future: asyncio.Future[ReaderResponse]


class _MicroBatchReaderRunner:
    def __init__(self, reader: DistilBertOnnxReader, max_batch_size: int, batch_timeout_ms: int) -> None:
        self.reader = reader
        self.max_batch_size = max(1, int(max_batch_size))
        self.batch_timeout_sec = max(0.001, float(batch_timeout_ms) / 1000.0)
        self.queue: asyncio.Queue[_PendingRequest] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="reader-microbatch-runner")

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def infer(self, request: ReaderRequest) -> ReaderResponse:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ReaderResponse] = loop.create_future()
        await self.queue.put(_PendingRequest(request=request, future=future))
        return await future

    async def _run(self) -> None:
        while True:
            first = await self.queue.get()
            batch = [first]
            deadline = asyncio.get_running_loop().time() + self.batch_timeout_sec

            while len(batch) < self.max_batch_size:
                timeout = deadline - asyncio.get_running_loop().time()
                if timeout <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    break
                batch.append(item)

            try:
                self._process_batch(batch)
            except Exception as exc:
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)

    def _process_batch(self, batch: list[_PendingRequest]) -> None:
        flat_questions: list[str] = []
        flat_contexts: list[str] = []
        mappings: list[tuple[int, ReaderDocument]] = []

        for idx, item in enumerate(batch):
            for doc in item.request.documents:
                flat_questions.append(item.request.question)
                flat_contexts.append(doc.content or "")
                mappings.append((idx, doc))

        predictions = self.reader.answer_on_question_context_pairs(flat_questions, flat_contexts) if flat_questions else []

        grouped: list[list[ReaderCandidate]] = [[] for _ in batch]
        for (batch_idx, doc), (answer, span_score, start_score, end_score) in zip(mappings, predictions):
            page = "N/A" if doc.page is None else str(doc.page)
            grouped[batch_idx].append(
                ReaderCandidate(
                    source=str(doc.source),
                    page=page,
                    answer=answer,
                    span_score=float(span_score),
                    start_score=float(start_score),
                    end_score=float(end_score),
                )
            )

        for idx, item in enumerate(batch):
            candidates = sorted(grouped[idx], key=lambda x: x.span_score, reverse=True)
            top_n = max(1, int(item.request.n_best))
            min_span_score = float(item.request.min_span_score)

            ranked_answers = [
                ReaderAnswer(
                    answer=cand.answer,
                    span_score=cand.span_score,
                    start_score=cand.start_score,
                    end_score=cand.end_score,
                    source=cand.source,
                    page=cand.page,
                    context="",
                )
                for cand in candidates
            ]
            selected = select_best_reader_answer(ranked_answers, min_span_score=min_span_score) if ranked_answers else None
            if selected is None:
                response = ReaderResponse(
                    selected_answer="I don't know.",
                    selected_span_score=0.0,
                    selection_reason="no_candidate_above_min_span_score_or_empty",
                    candidates=candidates[:top_n],
                )
            else:
                response = ReaderResponse(
                    selected_answer=selected.answer,
                    selected_span_score=float(selected.span_score),
                    selection_reason="selected_best_non_empty_above_min_span_score",
                    candidates=candidates[:top_n],
                )

            if not item.future.done():
                item.future.set_result(response)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config = load_config()
        reader = DistilBertOnnxReader(
            artifact_dir=config.reader_artifact_dir,
            max_length=config.reader_max_length,
            max_answer_length=config.reader_max_answer_length,
            n_best_size=config.reader_n_best_size,
            require_metadata=config.reader_require_metadata,
        )

        max_batch_size = int(os.getenv("READER_SERVICE_MAX_BATCH_SIZE", "16"))
        batch_timeout_ms = int(os.getenv("READER_SERVICE_BATCH_TIMEOUT_MS", "20"))
        runner = _MicroBatchReaderRunner(reader=reader, max_batch_size=max_batch_size, batch_timeout_ms=batch_timeout_ms)
        runner.start()

        app.state.config = config
        app.state.runner = runner
        app.state.max_batch_size = max_batch_size
        app.state.batch_timeout_ms = batch_timeout_ms
        logger.info(
            "reader_service_started max_batch_size=%s batch_timeout_ms=%s",
            max_batch_size,
            batch_timeout_ms,
        )
        try:
            yield
        finally:
            runner = app.state.runner
            if runner is not None:
                await runner.stop()

    app = FastAPI(title="RAG Reader ONNX Service", version="1.0.0", lifespan=lifespan)
    app.state.runner = None
    app.state.config = None
    app.state.max_batch_size = None
    app.state.batch_timeout_ms = None

    @app.get("/healthz")
    async def healthz(request: Request) -> dict[str, Any]:
        return {
            "status": "ok",
            "artifact_dir": getattr(request.app.state.config, "reader_artifact_dir", None),
            "max_batch_size": request.app.state.max_batch_size,
            "batch_timeout_ms": request.app.state.batch_timeout_ms,
        }

    @app.post("/v1/reader/answers", response_model=ReaderResponse)
    async def reader_answers(request: ReaderRequest, http_request: Request) -> ReaderResponse:
        runner = http_request.app.state.runner
        if runner is None:
            raise HTTPException(status_code=503, detail="reader service is not ready")
        return await runner.infer(request)

    return app


app = create_app()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reader service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    import uvicorn

    uvicorn.run("rag_chatbox.services.reader_service:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()

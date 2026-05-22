from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import OllamaLLM

from .config import AppConfig
from .ingestion import load_documents, split_documents
from .reader_distilbert import DistilBertOnnxReader, select_best_reader_answer
from .retrieval import build_retriever, format_docs_with_metadata

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are an intelligent assistant. Answer the question based on the following context and extracted answer candidate.
{context}

Reader Candidate Answer:
{reader_answer}
Reader Candidate Score:
{reader_span_score}

Previous Conversation History:
{chat_history}

Requirements:
- Only answer based on the provided context. Ignore personal/world knowledge not present in the context.
- If the answer is not explicitly supported by the context, say you don't know.
- Prefer the Reader Candidate Answer when it is supported by context.
- Always respond in the same language as the user's question. If the user's question is in Vietnamese, respond in Vietnamese.
- At the end of your response, add a section named `Nguồn:` with bullet points that only reference file/page pairs present in the context.
- Do not cite a source that is not present in the context block above.

Question: {question}
Answer:"""


@dataclass
class RagChatbot:
    ask_fn: Callable[[str, str], str]
    chat_history: str = ""

    def answer(self, question: str, chat_history: str = "") -> str:
        return self.ask_fn(question, chat_history)

    def ask(self, question: str) -> str:
        answer = self.answer(question, self.chat_history)
        logger.info(
            "rag_final_answer=%s",
            json.dumps(
                {
                    "question": question,
                    "answer": answer,
                },
                ensure_ascii=False,
            ),
        )
        self.chat_history += f"\nUser: {question}\nAssistant: {answer}\n"
        return answer


def _page_label(doc: Any) -> str:
    source = str(doc.metadata.get("source", "Unknown"))
    raw_page = doc.metadata.get("page_number", doc.metadata.get("page", "N/A"))
    if isinstance(raw_page, int):
        page = raw_page + 1
    else:
        page = raw_page
    return f"{source} (Page {page})"


def _normalize_source_label(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _format_allowed_sources(docs: list[Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for doc in docs:
        label = _page_label(doc)
        key = _normalize_source_label(label)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "source": str(doc.metadata.get("source", "Unknown")),
                "page": doc.metadata.get("page_number", doc.metadata.get("page", "N/A")),
                "label": label,
            }
        )
    return sources


def _split_answer_and_sources(answer: str) -> tuple[str, list[str]]:
    lines = (answer or "").splitlines()
    source_header_index = None
    for index, line in enumerate(lines):
        if re.match(r"^\s*(Nguồn|Sources?)\s*:?\s*$", line, flags=re.IGNORECASE):
            source_header_index = index
            break

    if source_header_index is None:
        return answer.strip(), []

    body = "\n".join(lines[:source_header_index]).strip()
    source_lines = [line.strip() for line in lines[source_header_index + 1 :] if line.strip()]
    return body, source_lines


def _finalize_answer(answer: str, allowed_sources: list[dict[str, Any]]) -> str:
    body, source_lines = _split_answer_and_sources(answer)

    if not allowed_sources:
        body = body.strip() or "Mình không biết."
        return body + "\n\nNguồn:\n- Không có trích dẫn hợp lệ trong context."

    allowed_labels = {_normalize_source_label(item["label"]): item["label"] for item in allowed_sources}

    valid_sources: list[str] = []
    for raw_line in source_lines:
        cleaned = re.sub(r"^[\-\*\d\.\)\s]+", "", raw_line).strip()
        key = _normalize_source_label(cleaned)
        if key in allowed_labels:
            valid_sources.append(allowed_labels[key])

    if not body:
        body = "Mình không biết."

    if not valid_sources:
        return body.rstrip() + "\n\nNguồn:\n- Không có trích dẫn hợp lệ trong context."

    return body.rstrip() + "\n\nNguồn:\n" + "\n".join(f"- {label}" for label in valid_sources)


def _prepare_chain_payload(
    question: str,
    chat_history: str,
    retriever: Callable[[str], list[Any]],
    reader: DistilBertOnnxReader,
    config: AppConfig,
) -> dict[str, Any]:
    docs = retriever(question)
    context_block = format_docs_with_metadata(docs)
    retrieval_sources = [
        {
            "source": str(doc.metadata.get("source", "Unknown")),
            "page": doc.metadata.get("page_number", doc.metadata.get("page", "N/A")),
            "rerank_score": doc.metadata.get("rerank_score"),
        }
        for doc in docs
    ]
    allowed_sources = _format_allowed_sources(docs)

    if not docs:
        logger.info(
            "rag_retrieval=%s",
            json.dumps(
                {
                    "question": question,
                    "retrieved_sources": retrieval_sources,
                    "reader_answer": "I don't know.",
                    "reader_span_score": 0.0,
                },
                ensure_ascii=False,
            ),
        )
        return {
            "context": "No relevant context was retrieved.",
            "reader_answer": "I don't know.",
            "reader_span_score": "0.0000",
            "chat_history": chat_history,
            "question": question,
            "allowed_sources": allowed_sources,
        }

    ranked_answers = reader.answer_on_documents(question=question, docs=docs)

    reader_answer = "I don't know."
    reader_span_score = 0.0
    if ranked_answers:
        top_answer = ranked_answers[0]
        reader_span_score = top_answer.span_score
        selected_answer = select_best_reader_answer(ranked_answers, config.reader_min_span_score)
        if selected_answer is not None:
            reader_answer = selected_answer.answer
            reader_span_score = selected_answer.span_score

    logger.info(
        "rag_retrieval=%s",
        json.dumps(
            {
                "question": question,
                "retrieved_sources": retrieval_sources,
                "reader_answer": reader_answer,
                "reader_span_score": reader_span_score,
            },
            ensure_ascii=False,
        ),
    )

    return {
        "context": context_block,
        "reader_answer": reader_answer,
        "reader_span_score": f"{reader_span_score:.4f}",
        "chat_history": chat_history,
        "question": question,
        "allowed_sources": allowed_sources,
    }


def build_chatbot(
    config: AppConfig,
    *,
    retriever: Callable[[str], list[Any]] | None = None,
    reader: DistilBertOnnxReader | None = None,
) -> RagChatbot:
    if retriever is None or reader is None:
        print("Đang tải dữ liệu và khởi tạo Retriever...")
        documents = load_documents(config)
        chunks = split_documents(documents, config)
        retriever = build_retriever(documents, chunks, config)
        reader = DistilBertOnnxReader(
            artifact_dir=config.reader_artifact_dir,
            max_length=config.reader_max_length,
            max_answer_length=config.reader_max_answer_length,
            n_best_size=config.reader_n_best_size,
            require_metadata=config.reader_require_metadata,
        )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = OllamaLLM(model=config.llm_model, temperature=config.temperature)
    runtime_state: dict[str, list[dict[str, Any]]] = {"allowed_sources": []}

    def _prepare_payload(question: str, chat_history: str) -> dict[str, Any]:
        payload = _prepare_chain_payload(
            question=question,
            chat_history=chat_history,
            retriever=retriever,
            reader=reader,
            config=config,
        )
        runtime_state["allowed_sources"] = list(payload.get("allowed_sources", []))
        return {
            "context": payload["context"],
            "reader_answer": payload["reader_answer"],
            "reader_span_score": payload["reader_span_score"],
            "chat_history": payload["chat_history"],
            "question": payload["question"],
        }

    chain = (
        RunnableLambda(
            lambda x: _prepare_payload(
                question=x["question"],
                chat_history=x["chat_history"],
            )
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    def ask_fn(question: str, chat_history: str) -> str:
        raw_answer = chain.invoke({"question": question, "chat_history": chat_history})
        return _finalize_answer(raw_answer, runtime_state["allowed_sources"])

    return RagChatbot(ask_fn=ask_fn)

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from .config import AppConfig
from .ingestion import load_documents, split_documents
from .prompt_template import PROMPT_TEMPLATE
from .reader_distilbert import DistilBertOnnxReader, select_best_reader_answer
from .retrieval import build_retriever, format_docs_with_metadata

logger = logging.getLogger(__name__)


@dataclass
class RagChatbot:
    ask_fn: Callable[[str, str], tuple[str, dict[str, Any] | None]]
    chat_history: str = ""
    last_debug: dict[str, Any] | None = None

    def answer(self, question: str, chat_history: str = "") -> str:
        answer, _ = self.ask_fn(question, chat_history)
        return answer

    def ask(self, question: str) -> str:
        answer, debug_trace = self.ask_fn(question, self.chat_history)
        self.last_debug = debug_trace
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

    def get_last_debug_trace(self) -> dict[str, Any] | None:
        return self.last_debug


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


def _source_basename(source: str) -> str:
    src = (source or "").strip().replace("\\", "/")
    return os.path.basename(src) if src else ""


def _to_source_page_key(source: str, page: int | str) -> str:
    source_name = _normalize_source_label(_source_basename(source))
    page_value = _normalize_source_label(str(page))
    return f"{source_name}::page::{page_value}"


def _label_to_source_page_key(label: str) -> str | None:
    match = re.match(r"^(?P<source>.+?)\s*\(page\s*(?P<page>[^\)]+)\)\s*$", label.strip(), flags=re.IGNORECASE)
    if not match:
        return None
    source = match.group("source").strip()
    page_raw = match.group("page").strip()
    try:
        page: int | str = int(page_raw)
    except ValueError:
        page = page_raw
    return _to_source_page_key(source, page)


def _preview(text: str, limit: int = 160) -> str:
    flattened = " ".join((text or "").split())
    if len(flattened) <= limit:
        return flattened
    return flattened[: max(0, limit - 3)].rstrip() + "..."


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


def _deduplicate_text(text: str) -> str:
    # 1. Line-level deduplication
    lines = (text or "").splitlines()
    deduped_lines = []
    seen_lines = set()
    for line in lines:
        cleaned_line = " ".join(line.strip().lower().split())
        if cleaned_line:
            if cleaned_line in seen_lines:
                continue
            seen_lines.add(cleaned_line)
        deduped_lines.append(line)
    text = "\n".join(deduped_lines).strip()

    # 2. Sentence-level deduplication (for intra-line loops)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    deduped_sentences = []
    seen_sentences = set()
    for sentence in sentences:
        cleaned_sentence = " ".join(sentence.strip().lower().split())
        if cleaned_sentence:
            if cleaned_sentence in seen_sentences:
                continue
            seen_sentences.add(cleaned_sentence)
        deduped_sentences.append(sentence)
    return " ".join(deduped_sentences).strip()


def _finalize_answer(answer: str, allowed_sources: list[dict[str, Any]]) -> str:
    body, source_lines = _split_answer_and_sources(answer)
    body = _deduplicate_text(body)

    if not allowed_sources:
        body = body.strip() or "Mình không biết."
        return body + "\n\nNguồn:\n- Không có trích dẫn hợp lệ trong context."

    allowed_labels = {_normalize_source_label(item["label"]): item["label"] for item in allowed_sources}
    allowed_key_map: dict[str, str] = {}
    for item in allowed_sources:
        page = item.get("page", "N/A")
        key = _to_source_page_key(str(item.get("source", "Unknown")), page)
        allowed_key_map[key] = str(item.get("label", ""))

    valid_sources: list[str] = []
    seen: set[str] = set()
    for raw_line in source_lines:
        cleaned = re.sub(r"^[\-\*\d\.\)\s]+", "", raw_line).strip()
        key = _normalize_source_label(cleaned)
        mapped_label = allowed_labels.get(key)
        if mapped_label:
            canonical = _normalize_source_label(mapped_label)
            if canonical not in seen:
                valid_sources.append(mapped_label)
                seen.add(canonical)
            continue

        source_key = _label_to_source_page_key(cleaned)
        if source_key is None:
            continue
        mapped_label = allowed_key_map.get(source_key)
        if mapped_label:
            canonical = _normalize_source_label(mapped_label)
            if canonical not in seen:
                valid_sources.append(mapped_label)
                seen.add(canonical)

    if not body:
        body = "Mình không biết."

    if not valid_sources:
        fallback_sources = [str(item.get("label", "")) for item in allowed_sources[:3] if item.get("label")]
        if fallback_sources:
            return body.rstrip() + "\n\nNguồn:\n" + "\n".join(f"- {label}" for label in fallback_sources)
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
    retrieval_trace = getattr(retriever, "last_trace", None)
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
    ranked_answers = reader.answer_on_documents(question=question, docs=docs) if docs else []

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
            "debug_trace": {
                "question_received": question,
                "chat_history_chars": len(chat_history or ""),
                "retrieval": retrieval_trace
                or {
                    "question": question,
                    "selected": [],
                    "decision": "no_chunks_after_threshold_filter",
                },
                "reader": {
                    "min_span_score": float(config.reader_min_span_score),
                    "candidate_count": 0,
                    "selected_answer": "I don't know.",
                    "selected_span_score": 0.0,
                    "selection_reason": "no_context_retrieved",
                },
            },
        }

    reader_answer = "I don't know."
    reader_span_score = 0.0
    reader_decision = "no_candidate_generated"
    if ranked_answers:
        top_answer = ranked_answers[0]
        reader_span_score = top_answer.span_score
        selected_answer = select_best_reader_answer(ranked_answers, config.reader_min_span_score)
        if selected_answer is not None:
            reader_answer = selected_answer.answer
            reader_span_score = selected_answer.span_score
            reader_decision = "selected_best_non_empty_above_min_span_score"
        else:
            reader_decision = "all_candidates_empty_or_below_min_span_score"

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

    reader_candidates = [
        {
            "rank": idx + 1,
            "source": candidate.source,
            "page": candidate.page,
            "span_score": float(candidate.span_score),
            "start_score": float(candidate.start_score),
            "end_score": float(candidate.end_score),
            "answer_preview": _preview(candidate.answer, limit=120),
        }
        for idx, candidate in enumerate(ranked_answers[: min(5, len(ranked_answers))])
    ]

    return {
        "context": context_block,
        "reader_answer": reader_answer,
        "reader_span_score": f"{reader_span_score:.4f}",
        "chat_history": chat_history,
        "question": question,
        "allowed_sources": allowed_sources,
        "debug_trace": {
            "question_received": question,
            "chat_history_chars": len(chat_history or ""),
            "retrieval": retrieval_trace
            or {
                "question": question,
                "selected": retrieval_sources,
                "decision": "selected_top_k_by_rerank",
            },
            "reader": {
                "min_span_score": float(config.reader_min_span_score),
                "candidate_count": len(ranked_answers),
                "top_candidates": reader_candidates,
                "selected_answer": _preview(reader_answer, limit=160),
                "selected_span_score": float(reader_span_score),
                "selection_reason": reader_decision,
            },
        },
    }


def build_chatbot(
    config: AppConfig,
    *,
    retriever: Callable[[str], list[Any]] | None = None,
    reader: DistilBertOnnxReader | None = None,
) -> RagChatbot:
    if retriever is None and reader is None:
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
    elif retriever is None:
        print("Đang tải dữ liệu và khởi tạo Retriever...")
        documents = load_documents(config)
        chunks = split_documents(documents, config)
        retriever = build_retriever(documents, chunks, config)
    elif reader is None:
        reader = DistilBertOnnxReader(
            artifact_dir=config.reader_artifact_dir,
            max_length=config.reader_max_length,
            max_answer_length=config.reader_max_answer_length,
            n_best_size=config.reader_n_best_size,
            require_metadata=config.reader_require_metadata,
        )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = OllamaLLM(
        model=config.llm_model,
        temperature=config.temperature,
        num_predict=120,
        base_url=config.ollama_base_url,
    )
    runtime_state: dict[str, Any] = {"allowed_sources": [], "debug_trace": None}

    def _prepare_payload(question: str, chat_history: str) -> dict[str, Any]:
        payload = _prepare_chain_payload(
            question=question,
            chat_history=chat_history,
            retriever=retriever,
            reader=reader,
            config=config,
        )
        runtime_state["allowed_sources"] = list(payload.get("allowed_sources", []))
        runtime_state["debug_trace"] = payload.get("debug_trace")
        return {
            "context": payload["context"],
            "reader_answer": payload["reader_answer"],
            "reader_span_score": payload["reader_span_score"],
            "chat_history": payload["chat_history"],
            "question": payload["question"],
        }

    def ask_fn(question: str, chat_history: str) -> tuple[str, dict[str, Any] | None]:
        payload = _prepare_payload(question=question, chat_history=chat_history)
        no_context = not runtime_state.get("allowed_sources")
        if no_context:
            final_answer = "Mình không biết.\n\nNguồn:\n- Không có trích dẫn hợp lệ trong context."
            raw_answer = final_answer
        else:
            raw_answer = (prompt | llm | StrOutputParser()).invoke(payload)
            final_answer = _finalize_answer(raw_answer, runtime_state["allowed_sources"])

        debug_trace = runtime_state.get("debug_trace")
        if debug_trace is not None:
            body, source_lines = _split_answer_and_sources(final_answer)
            debug_trace = dict(debug_trace)
            debug_trace["generation"] = {
                "llm_model": config.llm_model,
                "temperature": float(config.temperature),
                "llm_called": not no_context,
                "decision": "skipped_llm_due_to_empty_context" if no_context else "generated_with_llm_and_source_filter",
                "raw_answer_preview": _preview(str(raw_answer), limit=200),
                "final_answer_preview": _preview(body, limit=200),
                "final_sources": source_lines,
            }
        return final_answer, debug_trace

    return RagChatbot(ask_fn=ask_fn)

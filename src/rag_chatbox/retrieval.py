from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM

from .config import AppConfig

MANIFEST_FILE = "manifest.json"
logger = logging.getLogger(__name__)
QUERY_REWRITE_PROMPT = """You rewrite user questions for dense retrieval.
Return ONLY a JSON array of short rewritten search queries in the same language as the input.
- Keep meaning unchanged.
- Focus on key entities, technical terms, and synonyms.
- Do not answer the question.
- Return at most {max_variants} items.

Question: {question}
JSON:"""


def _build_embeddings(config: AppConfig) -> OllamaEmbeddings:
    return OllamaEmbeddings(model=config.embedding_model)


def _fingerprint_documents(documents: list[Document]) -> str:
    hasher = hashlib.sha256()

    for doc in documents:
        source = str(doc.metadata.get("source", ""))
        page = str(doc.metadata.get("page_number", doc.metadata.get("page", "")))
        content_hash = hashlib.sha256((doc.page_content or "").encode("utf-8")).hexdigest()
        hasher.update(f"{source}|{page}|{content_hash}\n".encode("utf-8"))

    return hasher.hexdigest()


def _build_manifest(documents: list[Document], chunks: list[Document], config: AppConfig) -> dict[str, Any]:
    return {
        "embedding_model": config.embedding_model,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "data_dir": config.data_dir,
        "file_glob": config.file_glob,
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "document_fingerprint": _fingerprint_documents(documents),
    }


def _load_manifest(index_dir: Path) -> dict[str, Any] | None:
    manifest_path = index_dir / MANIFEST_FILE
    if not manifest_path.exists():
        return None

    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _save_manifest(index_dir: Path, manifest: dict[str, Any]) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = index_dir / MANIFEST_FILE
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _load_index(index_dir: Path, config: AppConfig) -> FAISS:
    embeddings = _build_embeddings(config)
    return FAISS.load_local(
        folder_path=str(index_dir),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


def _save_index(index_dir: Path, vectorstore: FAISS) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", (text or "").lower()))


def _overlap_score(question: str, content: str) -> float:
    q_tokens = _tokenize(question)
    if not q_tokens:
        return 0.0
    c_tokens = _tokenize(content)
    return len(q_tokens & c_tokens) / len(q_tokens)


def _dedupe_non_empty(items: list[str], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in items:
        text = " ".join((raw or "").split()).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if limit is not None and len(result) >= limit:
            break
    return result


def _extract_json_array(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if text.startswith("[") and text.endswith("]"):
        return text

    match = re.search(r"\[[\s\S]*\]", text)
    return match.group(0) if match else ""


def _parse_rewritten_queries(raw_text: str, max_variants: int) -> list[str]:
    max_variants = max(0, int(max_variants))
    if max_variants == 0:
        return []

    candidates: list[str] = []
    parsed = _extract_json_array(raw_text)
    if parsed:
        try:
            payload = json.loads(parsed)
            if isinstance(payload, list):
                candidates = [item for item in payload if isinstance(item, str)]
        except json.JSONDecodeError:
            candidates = []

    if not candidates:
        for line in (raw_text or "").splitlines():
            cleaned = re.sub(r"^\s*[-*\d\.\)]\s*", "", line).strip()
            if cleaned:
                candidates.append(cleaned)

    return _dedupe_non_empty(candidates, limit=max_variants)


def _build_query_rewriter(config: AppConfig) -> Callable[[str], list[str]] | None:
    if not config.query_rewrite_enabled:
        return None

    rewrite_model = (config.query_rewrite_model or config.llm_model or "").strip()
    if not rewrite_model:
        logger.warning("Query rewriting is enabled but no rewrite model is configured; fallback to original query")
        return None

    llm = OllamaLLM(model=rewrite_model, temperature=config.query_rewrite_temperature)
    max_variants = max(0, config.query_rewrite_max_variants)

    def rewrite(question: str) -> list[str]:
        question = (question or "").strip()
        if not question:
            return []

        prompt = QUERY_REWRITE_PROMPT.format(max_variants=max_variants, question=question)
        try:
            raw = llm.invoke(prompt)
            variants = _parse_rewritten_queries(str(raw), max_variants=max_variants)
            queries = _dedupe_non_empty([question, *variants], limit=max_variants + 1)
            if len(queries) > 1:
                logger.info(
                    "Query rewrite generated %d variants for question=%r",
                    len(queries) - 1,
                    question,
                )
            return queries
        except Exception as exc:
            logger.warning("Query rewrite failed for question=%r: %s", question, exc)
            return [question]

    return rewrite


def _doc_key(doc: Document) -> tuple[str, str, str, str]:
    source = str(doc.metadata.get("source", ""))
    page = str(doc.metadata.get("page_number", doc.metadata.get("page", "")))
    start_index = str(doc.metadata.get("start_index", ""))
    content_hash = hashlib.sha256((doc.page_content or "").encode("utf-8")).hexdigest()
    return source, page, start_index, content_hash


def _make_retrieval_fn(
    vectorstore: FAISS,
    config: AppConfig,
    query_rewriter: Callable[[str], list[str]] | None = None,
) -> Callable[[str], list[Document]]:
    def retrieve(question: str) -> list[Document]:
        queries = [question]
        if query_rewriter is not None:
            rewritten = query_rewriter(question)
            queries = _dedupe_non_empty(rewritten if rewritten else [question], limit=config.query_rewrite_max_variants + 1)
            if question.strip():
                queries = _dedupe_non_empty([question, *queries], limit=config.query_rewrite_max_variants + 1)

        merged: dict[tuple[str, str, str, str], tuple[Document, float]] = {}
        for q in queries:
            candidates = vectorstore.similarity_search_with_relevance_scores(q, k=config.fetch_k)
            for doc, rel_score in candidates:
                if rel_score < config.score_threshold:
                    continue
                overlap = _overlap_score(question, doc.page_content)
                rerank_score = (0.8 * rel_score) + (0.2 * overlap)
                key = _doc_key(doc)
                previous = merged.get(key)
                if previous is None or rerank_score > previous[1]:
                    doc.metadata["matched_query"] = q
                    merged[key] = (doc, rerank_score)

        if not merged:
            logger.info(
                "No retrieved chunks cleared score threshold for question=%r",
                question,
            )
            return []

        filtered = list(merged.values())
        filtered.sort(key=lambda x: x[1], reverse=True)
        selected: list[Document] = []
        for doc, score in filtered[: config.top_k]:
            doc.metadata["rerank_score"] = float(score)
            selected.append(doc)
        return selected

    return retrieve


def build_retriever(documents: list[Document], chunks: list[Document], config: AppConfig) -> Callable[[str], list[Document]]:
    if not documents:
        raise ValueError(f"No documents were loaded from {config.data_dir!r} with glob {config.file_glob!r}")
    if not chunks:
        raise ValueError("Document split returned 0 chunks; check chunk_size/chunk_overlap and source files")

    index_dir = Path(config.faiss_index_dir)
    current_manifest = _build_manifest(documents, chunks, config)
    cached_manifest = _load_manifest(index_dir)

    if cached_manifest == current_manifest:
        try:
            vectorstore = _load_index(index_dir, config)
            print(f"Đã load FAISS cache từ: {index_dir}")
            return _make_retrieval_fn(vectorstore, config, _build_query_rewriter(config))
        except Exception:
            print("Cache FAISS không hợp lệ, sẽ rebuild index...")

    print("Đang build FAISS index mới...")
    embeddings = _build_embeddings(config)
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    _save_index(index_dir, vectorstore)
    _save_manifest(index_dir, current_manifest)
    print(f"Đã lưu FAISS cache tại: {index_dir}")

    return _make_retrieval_fn(vectorstore, config, _build_query_rewriter(config))


def format_docs_with_metadata(docs: list[Document]) -> str:
    formatted_chunks: list[str] = []
    for doc in docs:
        content = doc.page_content
        source = doc.metadata.get("source", "Unknown")
        raw_page = doc.metadata.get("page_number", doc.metadata.get("page"))
        if isinstance(raw_page, int):
            page = raw_page + 1
        elif raw_page is None:
            page = "N/A"
        else:
            page = raw_page
        formatted_chunks.append(f"--- Source: {source} (Page {page}) ---\n{content}")
    return "\n\n".join(formatted_chunks)

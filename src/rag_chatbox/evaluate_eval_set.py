from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

from .config import AppConfig, load_config
from .ingestion import load_documents, split_documents
from .rag_pipeline import build_chatbot
from .reader_distilbert import DistilBertOnnxReader, select_best_reader_answer
from .retrieval import build_retriever


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"\w+", (text or "").lower()))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    for token, count in pred_counts.items():
        overlap += min(count, ref_counts.get(token, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


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


def _extract_citations(answer: str) -> set[str]:
    _, source_lines = _split_answer_and_sources(answer)
    citations: set[str] = set()
    for raw_line in source_lines:
        cleaned = re.sub(r"^[\-\*\d\.\)\s]+", "", raw_line).strip()
        if cleaned:
            citations.add(_normalize_source_label(cleaned))
    return citations


def _normalize_source_label(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _page_label(source: str, page: int | str) -> str:
    return f"{source} (Page {page})"


def _parse_expected_source(item: Any) -> tuple[str | None, bool]:
    if isinstance(item, dict):
        source = item.get("source")
        page = item.get("page")
        if source is None or page is None:
            return None, False
        try:
            page_value: int | str = int(page)
        except (TypeError, ValueError):
            page_value = str(page)
        return _page_label(str(source), page_value), True

    if isinstance(item, str):
        match = re.match(r"^(?P<source>.+?)\s*\(page\s*(?P<page>\d+)\)\s*$", item, flags=re.IGNORECASE)
        if match:
            return _page_label(match.group("source").strip(), int(match.group("page"))), True
        raw = item.strip()
        return (raw or None), False

    return None, False


def _expected_source_labels(row: dict[str, Any]) -> tuple[set[str], bool]:
    labels: set[str] = set()
    checkable = True
    for item in row.get("expected_sources", []) or []:
        label, is_checkable = _parse_expected_source(item)
        if label:
            labels.add(_normalize_source_label(label))
        checkable = checkable and is_checkable
    return labels, checkable


def _doc_source_label(doc: Any) -> str:
    source = str(doc.metadata.get("source", "Unknown"))
    raw_page = doc.metadata.get("page_number", doc.metadata.get("page", "N/A"))
    if isinstance(raw_page, int):
        page = raw_page + 1
    else:
        page = raw_page
    return _normalize_source_label(_page_label(source, page))


def load_questions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _build_components(config: AppConfig, include_chatbot: bool) -> tuple[Any, Any, Any | None]:
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
    chatbot = build_chatbot(config, retriever=retriever, reader=reader) if include_chatbot else None
    return retriever, reader, chatbot


def evaluate_eval_set(
    config: AppConfig,
    eval_file: str | Path = "eval/questions.jsonl",
    *,
    mode: str = "rag",
    top_k: int = 5,
    limit: int = 0,
) -> dict[str, Any]:
    retriever, reader, chatbot = _build_components(config, include_chatbot=mode == "rag")
    questions = load_questions(Path(eval_file))
    questions = [row for row in questions if normalize_text(row.get("expected_answer", ""))]
    if limit > 0:
        questions = questions[:limit]

    total = 0
    retrieval_hits = 0
    retrieval_checked = 0
    reader_em = 0.0
    reader_f1 = 0.0
    rag_em = 0.0
    rag_f1 = 0.0
    citation_hits = 0
    citation_checked = 0
    rows: list[dict[str, Any]] = []

    for row in questions:
        total += 1
        question = row["question"]
        expected_answer = row["expected_answer"]
        expected_sources, source_checkable = _expected_source_labels(row)

        retrieved = retriever(question)[:top_k]
        retrieved_labels = {_doc_source_label(doc) for doc in retrieved}
        if source_checkable and expected_sources:
            retrieval_checked += 1
            retrieval_hits += int(bool(retrieved_labels & expected_sources))

        ranked_answers = reader.answer_on_documents(question=question, docs=retrieved)
        selected_reader_answer = select_best_reader_answer(ranked_answers, config.reader_min_span_score)
        if selected_reader_answer is not None:
            reader_prediction = selected_reader_answer.answer
        else:
            reader_prediction = "I don't know."

        reader_em += exact_match(reader_prediction, expected_answer)
        reader_f1 += token_f1(reader_prediction, expected_answer)

        rag_prediction = reader_prediction
        rag_citations: set[str] = set()
        if mode == "rag":
            assert chatbot is not None
            rag_prediction = chatbot.answer(question)
            rag_body, _ = _split_answer_and_sources(rag_prediction)
            rag_citations = _extract_citations(rag_prediction)
            if source_checkable and expected_sources:
                citation_checked += 1
                citation_hits += int(bool(rag_citations & expected_sources))

            rag_em += exact_match(rag_body, expected_answer)
            rag_f1 += token_f1(rag_body, expected_answer)

        rows.append(
            {
                "id": row.get("id"),
                "question": question,
                "expected_answer": expected_answer,
                "reader_prediction": reader_prediction,
                "rag_prediction": rag_prediction,
                "retrieved_sources": sorted(retrieved_labels),
                "expected_sources": sorted(expected_sources),
                "rag_citations": sorted(rag_citations),
            }
        )

    summary: dict[str, Any] = {
        "samples": total,
        "mode": mode,
        "top_k": top_k,
        "retrieval_hit_rate": retrieval_hits / retrieval_checked if retrieval_checked else 0.0,
        "reader_exact_match": reader_em / total if total else 0.0,
        "reader_f1": reader_f1 / total if total else 0.0,
    }
    if mode == "rag":
        summary.update(
            {
                "rag_exact_match": rag_em / total if total else 0.0,
                "rag_f1": rag_f1 / total if total else 0.0,
                "citation_hit_rate": citation_hits / citation_checked if citation_checked else 0.0,
            }
        )

    return {"summary": summary, "rows": rows}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the labeled RAG eval set")
    parser.add_argument("--eval-file", default="eval/questions.jsonl")
    parser.add_argument("--mode", choices=["reader", "rag"], default="rag")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args(argv)


def _resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    repo_candidate = Path(__file__).resolve().parents[2] / path
    if repo_candidate.exists():
        return repo_candidate
    return path


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    config = load_config()
    eval_file = _resolve_repo_path(args.eval_file)

    result = evaluate_eval_set(
        config=config,
        eval_file=eval_file,
        mode=args.mode,
        top_k=args.top_k,
        limit=args.limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

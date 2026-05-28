from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from rag_chatbox.config import load_config
from rag_chatbox.ingestion import load_documents, split_documents
from rag_chatbox.retrieval import build_retriever


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline indexer job for FAISS retrieval index")
    parser.add_argument("--print-summary", action="store_true", help="Print indexing summary as JSON")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    config = load_config()

    documents = load_documents(config)
    chunks = split_documents(documents, config)
    _ = build_retriever(documents, chunks, config)

    summary = {
        "data_dir": config.data_dir,
        "file_glob": config.file_glob,
        "faiss_index_dir": config.faiss_index_dir,
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "manifest_path": str(Path(config.faiss_index_dir) / "manifest.json"),
    }
    if args.print_summary:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

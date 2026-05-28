from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from .config import AppConfig

# Keep separators simple and deterministic for PDF text chunks.
TEXT_SEPARATORS = ["\n\n", "\n", " ", ""]


def _clean_text(text: str) -> str:
    cleaned = text.replace("\u00ad", "")
    cleaned = re.sub(r"-\n(?=\w)", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"(?m)^\s*\d+\s*$", "", cleaned)
    return cleaned.strip()


def load_documents(config: AppConfig) -> list[Document]:
    loader = DirectoryLoader(
        path=config.data_dir,
        glob=config.file_glob,
        show_progress=True,
        loader_cls=PyPDFLoader,
        use_multithreading=True,
    )
    documents = loader.load()
    if not config.clean_pdf_text:
        return documents

    cleaned_docs: list[Document] = []
    for doc in documents:
        cleaned_docs.append(
            Document(
                page_content=_clean_text(doc.page_content or ""),
                metadata=dict(doc.metadata),
            )
        )
    return cleaned_docs


def split_documents(documents: list[Document], config: AppConfig) -> list[Document]:
    if config.token_aware_chunking:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_model)
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            add_start_index=True,
            strip_whitespace=True,
            separators=TEXT_SEPARATORS,
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            add_start_index=True,
            strip_whitespace=True,
            separators=TEXT_SEPARATORS,
        )
    return splitter.split_documents(documents)

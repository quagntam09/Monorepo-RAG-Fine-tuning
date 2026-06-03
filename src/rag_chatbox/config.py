from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AppConfig:
    data_dir: str = "./paper"
    file_glob: str = "**/*.pdf"
    faiss_index_dir: str = "./.cache/faiss"
    embedding_model: str = "bge-m3"
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:0.5b"
    temperature: float = 0.2
    chunk_size: int = 1200
    chunk_overlap: int = 200
    top_k: int = 5
    fetch_k: int = 20
    score_threshold: float = 0.35
    fallback_to_top_k_when_no_threshold_hit: bool = True
    fallback_min_relevance_score: float = 0.0
    query_rewrite_enabled: bool = False
    query_rewrite_model: str = "qwen2.5:0.5b"
    query_rewrite_temperature: float = 0.0
    query_rewrite_max_variants: int = 3
    query_rewrite_cache_ttl_sec: float = 30.0
    query_rewrite_cache_max_size: int = 512
    retrieval_cache_ttl_sec: float = 15.0
    retrieval_cache_max_size: int = 256
    reader_artifact_dir: str = "./artifacts/readers/run_best"
    reader_max_length: int = 384
    reader_max_answer_length: int = 30
    reader_n_best_size: int = 20
    reader_min_span_score: float = 0.0
    reader_require_metadata: bool = True
    reader_service_url: str = "http://localhost:8081"
    reader_service_timeout_sec: float = 30.0
    reader_service_max_batch_size: int = 16
    reader_service_batch_timeout_ms: int = 20
    clean_pdf_text: bool = True
    token_aware_chunking: bool = False
    tokenizer_model: str = "distilbert-base-multilingual-cased"
    debug_trace: bool = False

    @property
    def reader_model_dir(self) -> str:
        return self.reader_artifact_dir


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_config_path(path: str | Path | None) -> Path:
    if path is None:
        return _repo_root() / "config" / "rag_config.yaml"
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return _repo_root() / resolved


def _read_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"RAG config file not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"RAG config must be a YAML mapping: {path}")

    valid_keys = {field.name for field in dataclasses.fields(AppConfig)}
    unknown_keys = sorted(set(data) - valid_keys)
    if unknown_keys:
        raise ValueError(f"Unknown RAG config keys in {path}: {unknown_keys}")

    return data


def _build_config(data: dict[str, Any]) -> AppConfig:
    config = AppConfig(**data)

    if config.chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if config.chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if config.chunk_overlap >= config.chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if config.top_k <= 0:
        raise ValueError("top_k must be > 0")
    if config.fetch_k <= 0:
        raise ValueError("fetch_k must be > 0")

    fetch_k = max(config.fetch_k, config.top_k)
    if fetch_k != config.fetch_k:
        config = dataclasses.replace(config, fetch_k=fetch_k)

    if config.query_rewrite_max_variants < 0:
        raise ValueError("query_rewrite_max_variants must be >= 0")
    if config.query_rewrite_cache_ttl_sec < 0:
        raise ValueError("query_rewrite_cache_ttl_sec must be >= 0")
    if config.retrieval_cache_ttl_sec < 0:
        raise ValueError("retrieval_cache_ttl_sec must be >= 0")
    if config.query_rewrite_cache_max_size <= 0:
        raise ValueError("query_rewrite_cache_max_size must be > 0")
    if config.retrieval_cache_max_size <= 0:
        raise ValueError("retrieval_cache_max_size must be > 0")
    if config.reader_service_timeout_sec <= 0:
        raise ValueError("reader_service_timeout_sec must be > 0")
    if config.reader_service_max_batch_size <= 0:
        raise ValueError("reader_service_max_batch_size must be > 0")
    if config.reader_service_batch_timeout_ms <= 0:
        raise ValueError("reader_service_batch_timeout_ms must be > 0")

    return config


def load_config(config_path: str | Path | None = None) -> AppConfig:
    path = _resolve_config_path(config_path)
    return _build_config(_read_yaml_config(path))

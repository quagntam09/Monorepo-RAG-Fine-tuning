from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    data_dir: str = "./paper"
    file_glob: str = "**/*.pdf"
    faiss_index_dir: str = "./.cache/faiss"
    embedding_model: str = "bge-m3"
    llm_model: str = "qwen2.5:0.5b"
    temperature: float = 0.2
    chunk_size: int = 1200
    chunk_overlap: int = 200
    top_k: int = 5
    fetch_k: int = 20
    score_threshold: float = 0.35
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
    clean_pdf_text: bool = True
    token_aware_chunking: bool = False
    tokenizer_model: str = "distilbert-base-multilingual-cased"
    debug_trace: bool = False

    @property
    def reader_model_dir(self) -> str:
        return self.reader_artifact_dir


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got: {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float, got: {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Environment variable {name} must be a boolean, got: {raw!r}")


def load_config() -> AppConfig:
    load_dotenv()

    chunk_size = _env_int("RAG_CHUNK_SIZE", 1200)
    chunk_overlap = _env_int("RAG_CHUNK_OVERLAP", 200)
    if chunk_size <= 0:
        raise ValueError("RAG_CHUNK_SIZE must be > 0")
    if chunk_overlap < 0:
        raise ValueError("RAG_CHUNK_OVERLAP must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")

    top_k = _env_int("RAG_TOP_K", 5)
    fetch_k = _env_int("RAG_FETCH_K", 20)
    if top_k <= 0:
        raise ValueError("RAG_TOP_K must be > 0")
    if fetch_k <= 0:
        raise ValueError("RAG_FETCH_K must be > 0")
    if fetch_k < top_k:
        fetch_k = top_k
    query_rewrite_max_variants = _env_int("RAG_QUERY_REWRITE_MAX_VARIANTS", 3)
    if query_rewrite_max_variants < 0:
        raise ValueError("RAG_QUERY_REWRITE_MAX_VARIANTS must be >= 0")
    query_rewrite_cache_ttl_sec = _env_float("RAG_QUERY_REWRITE_CACHE_TTL_SEC", 30.0)
    retrieval_cache_ttl_sec = _env_float("RAG_RETRIEVAL_CACHE_TTL_SEC", 15.0)
    query_rewrite_cache_max_size = _env_int("RAG_QUERY_REWRITE_CACHE_MAX_SIZE", 512)
    retrieval_cache_max_size = _env_int("RAG_RETRIEVAL_CACHE_MAX_SIZE", 256)
    if query_rewrite_cache_ttl_sec < 0:
        raise ValueError("RAG_QUERY_REWRITE_CACHE_TTL_SEC must be >= 0")
    if retrieval_cache_ttl_sec < 0:
        raise ValueError("RAG_RETRIEVAL_CACHE_TTL_SEC must be >= 0")
    if query_rewrite_cache_max_size <= 0:
        raise ValueError("RAG_QUERY_REWRITE_CACHE_MAX_SIZE must be > 0")
    if retrieval_cache_max_size <= 0:
        raise ValueError("RAG_RETRIEVAL_CACHE_MAX_SIZE must be > 0")

    return AppConfig(
        data_dir=os.getenv("RAG_DATA_DIR", "./paper"),
        file_glob=os.getenv("RAG_FILE_GLOB", "**/*.pdf"),
        faiss_index_dir=os.getenv("RAG_FAISS_INDEX_DIR", "./.cache/faiss"),
        embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "bge-m3"),
        llm_model=os.getenv("RAG_LLM_MODEL", "qwen2.5:0.5b"),
        temperature=_env_float("RAG_TEMPERATURE", 0.2),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        fetch_k=fetch_k,
        score_threshold=_env_float("RAG_SCORE_THRESHOLD", 0.35),
        query_rewrite_enabled=_env_bool("RAG_QUERY_REWRITE_ENABLED", False),
        query_rewrite_model=os.getenv("RAG_QUERY_REWRITE_MODEL", os.getenv("RAG_LLM_MODEL", "qwen2.5:0.5b")),
        query_rewrite_temperature=_env_float("RAG_QUERY_REWRITE_TEMPERATURE", 0.0),
        query_rewrite_max_variants=query_rewrite_max_variants,
        query_rewrite_cache_ttl_sec=query_rewrite_cache_ttl_sec,
        query_rewrite_cache_max_size=query_rewrite_cache_max_size,
        retrieval_cache_ttl_sec=retrieval_cache_ttl_sec,
        retrieval_cache_max_size=retrieval_cache_max_size,
        reader_artifact_dir=os.getenv(
            "RAG_READER_ARTIFACT_DIR",
            os.getenv("RAG_READER_MODEL_DIR", "./artifacts/readers/run_best"),
        ),
        reader_max_length=_env_int("RAG_READER_MAX_LENGTH", 384),
        reader_max_answer_length=_env_int("RAG_READER_MAX_ANSWER_LENGTH", 30),
        reader_n_best_size=_env_int("RAG_READER_N_BEST_SIZE", 20),
        reader_min_span_score=_env_float("RAG_READER_MIN_SPAN_SCORE", 0.0),
        reader_require_metadata=_env_bool("RAG_READER_REQUIRE_METADATA", True),
        clean_pdf_text=_env_bool("RAG_CLEAN_PDF_TEXT", True),
        token_aware_chunking=_env_bool("RAG_TOKEN_AWARE_CHUNKING", False),
        tokenizer_model=os.getenv("RAG_TOKENIZER_MODEL", "distilbert-base-multilingual-cased"),
        debug_trace=_env_bool("RAG_DEBUG_TRACE", False),
    )

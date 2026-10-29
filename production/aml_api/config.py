"""Runtime configuration with academic-leaderboard model guardrails."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


ACADEMIC_EMBEDDING_MODEL = "text-embedding-v4"
ACADEMIC_LLM_MODEL = "gpt-4o-mini"


def _integer(name: str, default: int, minimum: int = 1) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _boolean(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of: 1/0, true/false, yes/no, on/off")


def _float(name: str, default: float, *, minimum: float, maximum: float | None = None) -> float:
    value = float(os.getenv(name, str(default)))
    if value < minimum or (maximum is not None and value > maximum):
        suffix = f" and <= {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be >= {minimum}{suffix}")
    return value


@dataclass(frozen=True)
class Settings:
    storage_path: Path
    memory_system_key: str | None
    embedding_api_key: str
    embedding_base_url: str | None
    embedding_model: str
    llm_api_key: str
    llm_base_url: str | None
    llm_model: str
    caption_generation_mode: str
    reranker_model: str
    retrieve_pool: int
    use_bm25: bool
    bm25_k1: float
    bm25_b: float
    rerank_top_n: int
    return_limit: int
    neighbor_window: int
    base_first_quota: int
    add_processing_timeout_seconds: int

    @classmethod
    def from_env(cls) -> "Settings":
        embedding_model = os.getenv("AML_EMBEDDING_MODEL", ACADEMIC_EMBEDDING_MODEL)
        llm_model = os.getenv("AML_LLM_MODEL", ACADEMIC_LLM_MODEL)
        caption_generation_mode = os.getenv(
            "AML_CAPTION_GENERATION_MODE", "full"
        ).strip().lower()
        if embedding_model != ACADEMIC_EMBEDDING_MODEL:
            raise ValueError(
                f"Academic mode requires AML_EMBEDDING_MODEL={ACADEMIC_EMBEDDING_MODEL!r}; "
                f"got {embedding_model!r}."
            )
        if llm_model != ACADEMIC_LLM_MODEL:
            raise ValueError(
                f"Academic mode requires AML_LLM_MODEL={ACADEMIC_LLM_MODEL!r}; got {llm_model!r}."
            )
        if caption_generation_mode not in {"full", "complement"}:
            raise ValueError(
                "AML_CAPTION_GENERATION_MODE must be 'full' or 'complement'; "
                f"got {caption_generation_mode!r}."
            )

        common_key = os.getenv("OPENAI_API_KEY", "")
        embedding_key = os.getenv("AML_EMBEDDING_API_KEY", common_key)
        llm_key = os.getenv("AML_LLM_API_KEY", common_key)
        if not embedding_key:
            raise ValueError("Set AML_EMBEDDING_API_KEY (or OPENAI_API_KEY).")
        if not llm_key:
            raise ValueError("Set AML_LLM_API_KEY (or OPENAI_API_KEY).")

        default_store = Path(__file__).resolve().parents[1] / "aml_data" / "memory.sqlite3"
        return cls(
            storage_path=Path(os.getenv("AML_STORAGE_PATH", str(default_store))).resolve(),
            memory_system_key=os.getenv("AML_MEMORY_SYSTEM_KEY") or None,
            embedding_api_key=embedding_key,
            embedding_base_url=os.getenv("AML_EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None,
            embedding_model=embedding_model,
            llm_api_key=llm_key,
            llm_base_url=os.getenv("AML_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None,
            llm_model=llm_model,
            caption_generation_mode=caption_generation_mode,
            # The competition explicitly leaves rerankers unrestricted. By default we still use
            # gpt-4o-mini, keeping deployment simple and the entire online path auditable.
            reranker_model=os.getenv("AML_RERANKER_MODEL", ACADEMIC_LLM_MODEL),
            retrieve_pool=_integer("AML_RETRIEVE_POOL", 80),
            use_bm25=_boolean("AML_USE_BM25", False),
            bm25_k1=_float("AML_BM25_K1", 1.5, minimum=0.01),
            bm25_b=_float("AML_BM25_B", 0.75, minimum=0.0, maximum=1.0),
            rerank_top_n=_integer("AML_RERANK_TOP_N", 16, minimum=0),
            return_limit=min(100, _integer("AML_RETURN_LIMIT", 24)),
            neighbor_window=_integer("AML_NEIGHBOR_WINDOW", 1, minimum=0),
            # 0 keeps the historical block-oriented assembly. A positive value reserves
            # that many leading slots for ranked base hits before session neighbors.
            base_first_quota=_integer("AML_BASE_FIRST_QUOTA", 0, minimum=0),
            add_processing_timeout_seconds=_integer("AML_ADD_PROCESSING_TIMEOUT_SECONDS", 1800),
        )

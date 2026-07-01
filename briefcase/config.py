from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class ChunkingConfig:
    target_tokens: int = _env_int("BRIEFCASE_CHUNK_TARGET_TOKENS", 220)
    overlap_tokens: int = _env_int("BRIEFCASE_CHUNK_OVERLAP_TOKENS", 40)
    max_tokens: int = _env_int("BRIEFCASE_CHUNK_MAX_TOKENS", 320)
    parent_window: int = _env_int("BRIEFCASE_PARENT_WINDOW", 3)


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = _env_int("BRIEFCASE_TOP_K", 8)
    candidate_k: int = _env_int("BRIEFCASE_CANDIDATE_K", 40)
    rrf_k: int = _env_int("BRIEFCASE_RRF_K", 60)
    min_score: float = _env_float("BRIEFCASE_MIN_SCORE", 0.15)
    context_token_budget: int = _env_int("BRIEFCASE_CONTEXT_TOKENS", 3000)
    neighbor_expand: bool = _env("BRIEFCASE_NEIGHBOR_EXPAND", "1") == "1"


@dataclass(frozen=True)
class EmbeddingConfig:
    # "hashing" (default, no deps), "sentence-transformers", or "gemini"
    provider: str = _env("BRIEFCASE_EMBEDDINGS", "hashing")
    dim: int = _env_int("BRIEFCASE_EMBED_DIM", 768)
    st_model: str = _env("BRIEFCASE_ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    gemini_model: str = _env("BRIEFCASE_GEMINI_EMBED_MODEL", "text-embedding-004")


@dataclass(frozen=True)
class LLMConfig:
    # "auto" picks gemini -> ollama -> extractive fallback based on availability.
    provider: str = _env("BRIEFCASE_LLM", "auto")
    gemini_model: str = _env("BRIEFCASE_GEMINI_MODEL", "gemini-2.5-flash")
    ollama_model: str = _env("BRIEFCASE_OLLAMA_MODEL", "llama3.2")
    ollama_url: str = _env("BRIEFCASE_OLLAMA_URL", "http://localhost:11434")
    temperature: float = _env_float("BRIEFCASE_LLM_TEMPERATURE", 0.2)
    max_tokens: int = _env_int("BRIEFCASE_LLM_MAX_TOKENS", 1024)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(
        default_factory=lambda: Path(
            _env("BRIEFCASE_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))
        )
    )
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    # OCR for scanned / image-only PDFs and image files. "auto" uses OCR when a
    # backend is installed; "off" disables it.
    ocr: str = field(default_factory=lambda: _env("BRIEFCASE_OCR", "auto"))
    ocr_dpi: int = field(default_factory=lambda: _env_int("BRIEFCASE_OCR_DPI", 200))
    ocr_max_pages: int = field(default_factory=lambda: _env_int("BRIEFCASE_OCR_MAX_PAGES", 50))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "briefcase.db"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    return settings

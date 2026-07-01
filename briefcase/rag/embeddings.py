from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from typing import Protocol

import numpy as np

from briefcase.config import get_settings
from briefcase.text_utils import tokenize_words


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """A deterministic, dependency-free dense embedder.

    Uses the feature-hashing trick over word unigrams + bigrams with sub-linear
    term weighting, projected into a fixed-dimensional L2-normalised space.
    It has no semantic pre-training, but combined with BM25 in the hybrid
    retriever it gives solid lexical-plus-fuzzy recall, and it runs anywhere,
    offline, with zero model downloads.
    """

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or get_settings().embeddings.dim

    def _feature_index(self, feature: str) -> tuple[int, int]:
        h = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % self.dim
        sign = 1 if h[4] & 1 else -1
        return idx, sign

    def _vectorize(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        words = tokenize_words(text)
        if not words:
            return vec
        features: dict[str, float] = {}
        for w in words:
            features[w] = features.get(w, 0.0) + 1.0
        for a, b in zip(words, words[1:]):
            bigram = f"{a}_{b}"
            features[bigram] = features.get(bigram, 0.0) + 1.0
        for feature, count in features.items():
            idx, sign = self._feature_index(feature)
            vec[idx] += sign * (1.0 + math.log(count))
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vectorize(t) for t in texts])


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


class GeminiEmbedder:
    def __init__(self, model_name: str, dim: int) -> None:
        from google import genai

        from briefcase import settings_store

        key = settings_store.get_secret("gemini_api_key", "GEMINI_API_KEY", "GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set")
        self._client = genai.Client(api_key=key)
        self._model = model_name
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        resp = self._client.models.embed_content(model=self._model, contents=texts)
        vecs = np.asarray([e.values for e in resp.embeddings], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


@lru_cache
def get_embedder() -> Embedder:
    from briefcase import settings_store

    cfg = get_settings().embeddings
    provider = (settings_store.get("embeddings_provider") or cfg.provider).lower()
    try:
        if provider in ("sentence-transformers", "st"):
            return SentenceTransformerEmbedder(cfg.st_model)
        if provider == "gemini":
            return GeminiEmbedder(cfg.gemini_model, cfg.dim)
    except Exception:  # pragma: no cover - graceful fallback
        pass
    return HashingEmbedder(cfg.dim)

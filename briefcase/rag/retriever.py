from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from briefcase.config import get_settings
from briefcase.models import RetrievedChunk
from briefcase.rag.embeddings import get_embedder
from briefcase.repository import load_chunks
from briefcase.text_utils import content_tokens


class _BM25:
    """Compact BM25 Okapi over an in-memory corpus."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.n = len(corpus_tokens)
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avg_len = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.freqs: list[dict[str, int]] = []
        df: dict[str, int] = defaultdict(int)
        for tokens in corpus_tokens:
            counts: dict[str, int] = defaultdict(int)
            for t in tokens:
                counts[t] += 1
            self.freqs.append(counts)
            for term in counts:
                df[term] += 1
        self.idf = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in range(self.n):
                f = self.freqs[i].get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avg_len or 1))
                out[i] += idf * (f * (self.k1 + 1)) / denom
        return out


def _rrf(rankings: list[list[int]], k: int) -> dict[int, float]:
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] += 1.0 / (k + rank + 1)
    return fused


class HybridRetriever:
    """BM25 + dense-vector retrieval fused with Reciprocal Rank Fusion.

    Mirrors Pragya's retrieval strategy, scoped per notebook. Indices are built
    on demand from the notebook's chunks, fine for personal-scale corpora and
    keeps the storage layer a single SQLite file.
    """

    def __init__(self) -> None:
        self._cfg = get_settings().retrieval

    @property
    def _embedder(self):
        return get_embedder()

    def retrieve(
        self, notebook_id: str, query: str, *, source_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        chunks = load_chunks(notebook_id, source_ids)
        if not chunks:
            return []
        top_k = top_k or self._cfg.top_k

        # Dense (cosine) scores.
        has_vectors = all(c["vector"] is not None for c in chunks)
        query_vec = self._embedder.embed([query])[0]
        if has_vectors:
            matrix = np.vstack([c["vector"] for c in chunks])
            # If the embedder changed since these chunks were indexed, dimensions
            # won't match. Fall back to lexical-only rather than crash; re-ingest
            # to restore semantic search under the new embedder.
            if matrix.shape[1] == query_vec.shape[0]:
                vec_scores = matrix @ query_vec
            else:
                vec_scores = np.zeros(len(chunks), dtype=np.float32)
        else:
            vec_scores = np.zeros(len(chunks), dtype=np.float32)

        # Lexical (BM25) scores.
        corpus_tokens = [content_tokens(c["text"]) for c in chunks]
        bm25 = _BM25(corpus_tokens)
        bm25_scores = bm25.scores(content_tokens(query))

        candidate_k = min(self._cfg.candidate_k, len(chunks))
        vec_rank = list(np.argsort(-vec_scores)[:candidate_k])
        bm25_rank = list(np.argsort(-bm25_scores)[:candidate_k])
        fused = _rrf([vec_rank, bm25_rank], self._cfg.rrf_k)

        selected_by: dict[int, list[str]] = defaultdict(list)
        for i in vec_rank[: self._cfg.top_k * 2]:
            if vec_scores[i] > 0:
                selected_by[int(i)].append("vector")
        for i in bm25_rank[: self._cfg.top_k * 2]:
            if bm25_scores[i] > 0:
                selected_by[int(i)].append("bm25")

        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        results: list[RetrievedChunk] = []
        for idx, score in ordered[:top_k]:
            c = chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=c["chunk_id"],
                    source_id=c["source_id"],
                    source_title=c["source_title"],
                    ordinal=c["ordinal"],
                    text=c["text"],
                    vector_score=float(vec_scores[idx]),
                    bm25_score=float(bm25_scores[idx]),
                    fused_score=float(score),
                    selected_by=selected_by.get(idx, []),
                )
            )
        return results

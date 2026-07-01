from __future__ import annotations

import re

from briefcase.config import get_settings
from briefcase.models import Answer, Citation, RetrievedChunk
from briefcase.rag.llm import NoLLM, get_llm
from briefcase.rag.prompts import ANSWER_SYSTEM, build_answer_prompt
from briefcase.rag.retriever import HybridRetriever
from briefcase.text_utils import content_tokens, count_tokens, snippet, split_sentences

_MARKER = re.compile(r"\[(\d+)\]")
_REFUSAL = "I don't have enough in your sources to answer that."


def _short_llm_error(exc: Exception) -> str:
    """Turn a verbose provider error into one actionable sentence."""
    msg = str(exc)
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        return (
            "The model is rate-limited or out of quota (HTTP 429). Wait a moment, or "
            "switch model/engine in Settings (this key has no quota for some models)."
        )
    if "404" in msg or "NOT_FOUND" in msg:
        return "The selected model was not found (HTTP 404). Pick a valid model in Settings."
    if any(t in msg for t in ("401", "403", "PERMISSION_DENIED", "API key", "API_KEY")):
        return "The API key was rejected (auth error). Re-check the key in Settings."
    return f"The model call failed ({type(exc).__name__}). Showing source passages instead."


class AnswerGenerator:
    def __init__(self, retriever: HybridRetriever | None = None) -> None:
        self._retriever = retriever or HybridRetriever()
        self._cfg = get_settings().retrieval

    @property
    def _llm(self):
        # Fetched fresh (lru-cached) so a Settings change takes effect immediately.
        return get_llm()

    def answer(
        self, notebook_id: str, question: str, *, source_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> Answer:
        chunks = self._retriever.retrieve(
            notebook_id, question, source_ids=source_ids, top_k=top_k
        )
        if not chunks or not self._has_signal(chunks):
            return Answer(
                answer=_REFUSAL, refused=True, confidence=0.0,
                engine=self._llm.name, diagnostics={"retrieved": len(chunks)},
            )

        selected = self._fit_budget(chunks)
        confidence = self._confidence(selected)

        if isinstance(self._llm, NoLLM):
            return self._extractive_answer(question, selected, confidence)
        return self._llm_answer(question, selected, confidence)

    # ---- LLM-backed answering ----

    def _llm_answer(self, question: str, chunks: list[RetrievedChunk], confidence: float) -> Answer:
        passages = [(i + 1, c.source_title, c.text) for i, c in enumerate(chunks)]
        prompt = build_answer_prompt(question, passages)
        try:
            text = self._llm.complete(prompt, system=ANSWER_SYSTEM)
        except Exception as exc:  # pragma: no cover - network/runtime failure
            return self._extractive_answer(
                question, chunks, confidence, note=_short_llm_error(exc)
            )

        if not text or _REFUSAL.lower()[:20] in text.lower() and len(text) < 80:
            return Answer(answer=_REFUSAL, refused=True, confidence=0.0, engine=self._llm.name)

        citations = self._citations_from_markers(text, chunks)
        return Answer(
            answer=text, refused=False, confidence=confidence, citations=citations,
            engine=self._llm.name,
            diagnostics={"passages": len(chunks), "cited": len(citations)},
        )

    # ---- Extractive fallback (no LLM) ----

    def _extractive_answer(
        self, question: str, chunks: list[RetrievedChunk], confidence: float, note: str = "",
    ) -> Answer:
        q_terms = set(content_tokens(question))
        scored: list[tuple[float, int, str]] = []  # (score, chunk_index, sentence)
        for idx, chunk in enumerate(chunks):
            for sentence in split_sentences(chunk.text):
                terms = set(content_tokens(sentence))
                if not terms:
                    continue
                overlap = len(q_terms & terms)
                if overlap == 0:
                    continue
                score = overlap / (len(terms) ** 0.5)
                scored.append((score, idx, sentence.strip()))
        scored.sort(key=lambda t: t[0], reverse=True)

        picked: list[tuple[int, str]] = []
        seen_chunks: set[int] = set()
        for _, idx, sentence in scored:
            if sentence in [s for _, s in picked]:
                continue
            picked.append((idx, sentence))
            seen_chunks.add(idx)
            if len(picked) >= 5:
                break

        if not picked:
            # No lexical overlap, surface the top retrieved passage rather than refuse.
            picked = [(0, snippet(chunks[0].text, 400))]

        lines = ["Here is what your sources say:", ""]
        used_indices: list[int] = []
        for idx, sentence in picked:
            marker = idx + 1
            used_indices.append(idx)
            lines.append(f"- {sentence} [{marker}]")
        body = "\n".join(lines)
        if note:
            body += f"\n\n_{note}_"
        body += (
            "\n\n_Briefcase is running without a generative model, so this answer is "
            "extracted verbatim from your sources. Configure Gemini or Ollama for "
            "synthesized answers._"
        )

        citations = self._citations_for_indices(used_indices, chunks)
        return Answer(
            answer=body, refused=False, confidence=confidence, citations=citations,
            engine="extractive", diagnostics={"passages": len(chunks)},
        )

    # ---- helpers ----

    def _citations_from_markers(self, text: str, chunks: list[RetrievedChunk]) -> list[Citation]:
        markers = sorted({int(m) for m in _MARKER.findall(text)})
        indices = [m - 1 for m in markers if 1 <= m <= len(chunks)]
        return self._citations_for_indices(indices, chunks, renumber=False)

    def _citations_for_indices(
        self, indices: list[int], chunks: list[RetrievedChunk], renumber: bool = True,
    ) -> list[Citation]:
        seen: list[int] = []
        for i in indices:
            if i not in seen:
                seen.append(i)
        citations: list[Citation] = []
        for i in seen:
            c = chunks[i]
            citations.append(
                Citation(
                    marker=i + 1,
                    source_id=c.source_id,
                    source_title=c.source_title,
                    chunk_id=c.chunk_id,
                    ordinal=c.ordinal,
                    quote=snippet(c.text, 300),
                )
            )
        return citations

    def _fit_budget(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        budget = self._cfg.context_token_budget
        out: list[RetrievedChunk] = []
        used = 0
        for c in chunks:
            t = count_tokens(c.text)
            if used + t > budget and out:
                break
            out.append(c)
            used += t
        return out

    def _has_signal(self, chunks: list[RetrievedChunk]) -> bool:
        best_vec = max((c.vector_score for c in chunks), default=0.0)
        best_bm25 = max((c.bm25_score for c in chunks), default=0.0)
        return best_vec >= self._cfg.min_score or best_bm25 > 0.0

    def _confidence(self, chunks: list[RetrievedChunk]) -> float:
        if not chunks:
            return 0.0
        top = chunks[: min(3, len(chunks))]
        vec = sum(max(c.vector_score, 0.0) for c in top) / len(top)
        # Squash cosine (typically small for hashing embedder) into a friendly 0-1.
        conf = min(1.0, 0.35 + vec * 1.5)
        if any(c.bm25_score > 0 for c in top):
            conf = min(1.0, conf + 0.1)
        return round(conf, 3)

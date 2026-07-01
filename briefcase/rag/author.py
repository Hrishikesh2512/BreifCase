"""The Draft author: grounded write / rewrite / check over a notebook's sources.

Reuses the retriever, LLM, and citation machinery. Like the rest of Briefcase it
degrades gracefully: with no generative model, `write` still returns cited
passages assembled from the sources, and `check` works without any model at all.
"""

from __future__ import annotations

import re

from briefcase.models import Citation, ClaimCheck, DraftEdit, RetrievedChunk
from briefcase.rag.generator import _short_llm_error
from briefcase.rag.llm import NoLLM, get_llm
from briefcase.rag.prompts import (
    EDIT_SYSTEM,
    REWRITE_SYSTEM,
    VERIFY_SYSTEM,
    build_edit_prompt,
    build_rewrite_prompt,
    build_verify_prompt,
)
from briefcase.rag.retriever import HybridRetriever
from briefcase.text_utils import content_tokens, snippet, split_sentences

_MARKER = re.compile(r"\[(\d+)\]")
_EXTRACTIVE_NOTE = (
    "Written without a generative model, so this is assembled from your sources. "
    "Add Gemini or Ollama in Settings for synthesised drafting."
)


class DraftAuthor:
    def __init__(self, retriever: HybridRetriever | None = None) -> None:
        self._retriever = retriever or HybridRetriever()

    # ---- public commands ----

    def edit(
        self, notebook_id: str, instruction: str, existing_body: str,
        *, source_ids: list[str] | None = None,
    ) -> DraftEdit:
        """Revise the whole draft in place per the instruction, reading the sources
        (read-only) and the current draft. Returns the complete new body."""
        query = f"{instruction} {existing_body[:500]}".strip() or "overview"
        chunks = self._retriever.retrieve(notebook_id, query, source_ids=source_ids, top_k=8)
        llm = get_llm()

        if isinstance(llm, NoLLM):
            # No model: can only assemble from sources into an empty draft.
            if existing_body.strip():
                return DraftEdit(
                    body=existing_body,
                    note="Editing the draft needs a generative model. Add Gemini or Ollama in Settings.",
                )
            if not chunks:
                return DraftEdit(body=existing_body, note="No sources to draw from yet. Add sources first.")
            text, citations, engine, note = self._extractive_section(instruction, chunks)
            return DraftEdit(body=text, citations=citations, engine=engine, note=note)

        passages = [(i + 1, c.source_title, c.text) for i, c in enumerate(chunks)]
        try:
            new_body = llm.complete(build_edit_prompt(instruction, passages, existing_body), system=EDIT_SYSTEM)
        except Exception as exc:
            return DraftEdit(body=existing_body, note=_short_llm_error(exc))
        new_body = new_body.strip()
        if not new_body:
            return DraftEdit(body=existing_body, note="The model returned nothing; draft unchanged.")
        return DraftEdit(body=new_body, citations=self._citations(new_body, chunks), engine=llm.name)

    def rewrite(
        self, notebook_id: str, instruction: str, selection: str, existing_body: str,
        *, source_ids: list[str] | None = None,
    ) -> DraftEdit:
        selection = (selection or "").strip()
        if not selection:
            return DraftEdit(body=existing_body, note="Select some text in the draft to rewrite.")
        llm = get_llm()
        if isinstance(llm, NoLLM):
            return DraftEdit(body=existing_body, note="Rewriting needs a generative model. Add Gemini or Ollama in Settings.")

        query = f"{instruction} {selection}".strip()
        chunks = self._retriever.retrieve(notebook_id, query, source_ids=source_ids, top_k=6)
        passages = [(i + 1, c.source_title, c.text) for i, c in enumerate(chunks)]
        try:
            rewritten = llm.complete(build_rewrite_prompt(instruction, selection, passages), system=REWRITE_SYSTEM)
        except Exception as exc:
            return DraftEdit(body=existing_body, note=_short_llm_error(exc))

        rewritten = rewritten.strip()
        if selection in existing_body:
            new_body = existing_body.replace(selection, rewritten, 1)
        else:  # selection not found verbatim (e.g. whitespace differences): append.
            new_body = f"{existing_body.rstrip()}\n\n{rewritten}".strip()
        return DraftEdit(body=new_body, citations=self._citations(rewritten, chunks), engine=llm.name)

    def check(
        self, notebook_id: str, body: str, *, source_ids: list[str] | None = None,
    ) -> DraftEdit:
        """Fact-check each claim in the draft against the sources.

        Uses the LLM for real entailment (so a claim that a source *contradicts* is
        marked unsupported, not just topical-word overlap). Falls back to a weaker
        lexical heuristic only when no model is configured.
        """
        claims = [s.strip() for s in split_sentences(body) if len(content_tokens(s)) >= 3]
        if not claims:
            return DraftEdit(body=body, note="Nothing to check yet.", engine=get_llm().name)
        claims = claims[:40]

        # Evidence pool: top passages per claim, de-duplicated.
        pool: list = []
        seen: set[str] = set()
        for claim in claims:
            for c in self._retriever.retrieve(notebook_id, claim, source_ids=source_ids, top_k=3):
                if c.chunk_id not in seen:
                    seen.add(c.chunk_id)
                    pool.append(c)
        pool = pool[:24]

        llm = get_llm()
        if isinstance(llm, NoLLM):
            return self._lexical_check(claims, notebook_id, source_ids)
        if not pool:
            checks = [ClaimCheck(claim=snippet(c, 200), supported=False) for c in claims]
            return DraftEdit(body=body, checks=checks, engine=llm.name,
                             note="No enabled sources to check against.")

        passages = [(i + 1, c.source_title, c.text) for i, c in enumerate(pool)]
        try:
            raw = llm.complete(build_verify_prompt(claims, passages), system=VERIFY_SYSTEM)
            verdicts = self._parse_verdicts(raw, len(claims))
        except Exception:
            # Don't fall back to word-overlap here: it produces false "supported"
            # verdicts. Be honest that verification couldn't run.
            return DraftEdit(
                body=body, checks=[], engine=llm.name,
                note="Couldn't fact-check just now (the model was busy). Try again in a moment.",
            )

        checks: list[ClaimCheck] = []
        for i, claim in enumerate(claims):
            v = verdicts.get(i, {})
            supported = bool(v.get("supported"))
            p = v.get("passage")
            src = passages[p - 1][1] if (supported and isinstance(p, int) and 1 <= p <= len(passages)) else None
            checks.append(ClaimCheck(claim=snippet(claim, 200), supported=supported, supported_by=src))

        unsupported = sum(1 for c in checks if not c.supported)
        note = (
            "All claims are supported by your sources."
            if unsupported == 0
            else f"{unsupported} of {len(checks)} claims aren't supported by your sources (check the ✗ items)."
        )
        return DraftEdit(body=body, checks=checks, note=note, engine=llm.name)

    def _parse_verdicts(self, raw: str, n: int) -> dict[int, dict]:
        import json

        start, end = raw.find("["), raw.rfind("]")
        verdicts: dict[int, dict] = {}
        if start == -1 or end == -1:
            return verdicts
        try:
            for item in json.loads(raw[start : end + 1]):
                idx = int(item.get("claim", 0)) - 1
                if 0 <= idx < n:
                    verdicts[idx] = item
        except Exception:
            pass
        return verdicts

    def _lexical_check(
        self, claims: list[str], notebook_id: str, source_ids: list[str] | None
    ) -> DraftEdit:
        checks: list[ClaimCheck] = []
        for claim in claims:
            hits = self._retriever.retrieve(notebook_id, claim, source_ids=source_ids, top_k=1)
            best = hits[0] if hits else None
            supported = bool(best) and (best.bm25_score > 0 or best.vector_score >= 0.2)
            checks.append(ClaimCheck(
                claim=snippet(claim, 200), supported=supported,
                supported_by=best.source_title if (supported and best) else None,
            ))
        note = ("Approximate check (no model configured): matches wording, not facts. "
                "Add Gemini or Ollama in Settings for true fact-checking.")
        return DraftEdit(body="", checks=checks, note=note, engine="extractive")

    # ---- helpers ----

    def _citations(self, text: str, chunks: list[RetrievedChunk]) -> list[Citation]:
        markers = sorted({int(m) for m in _MARKER.findall(text)})
        out: list[Citation] = []
        for m in markers:
            if 1 <= m <= len(chunks):
                c = chunks[m - 1]
                out.append(Citation(
                    marker=m, source_id=c.source_id, source_title=c.source_title,
                    chunk_id=c.chunk_id, ordinal=c.ordinal, quote=snippet(c.text, 300),
                ))
        return out

    def _extractive_section(self, instruction: str, chunks: list[RetrievedChunk]):
        q_terms = set(content_tokens(instruction))
        scored: list[tuple[float, int, str]] = []
        for idx, chunk in enumerate(chunks):
            for sentence in split_sentences(chunk.text):
                terms = set(content_tokens(sentence))
                if not terms:
                    continue
                overlap = len(q_terms & terms) if q_terms else 1
                scored.append((overlap / (len(terms) ** 0.5), idx, sentence.strip()))
        scored.sort(key=lambda t: t[0], reverse=True)
        picked, used = [], []
        for _, idx, sentence in scored:
            if sentence not in [s for _, s in picked]:
                picked.append((idx, sentence))
                used.append(idx)
            if len(picked) >= 6:
                break
        if not picked:
            picked = [(0, snippet(chunks[0].text, 400))]
            used = [0]
        lines = [f"- {s} [{idx + 1}]" for idx, s in picked]
        text = "\n".join(lines)
        citations = self._citations(" ".join(f"[{i + 1}]" for i in sorted(set(used))), chunks)
        return text, citations, "extractive", _EXTRACTIVE_NOTE

from __future__ import annotations

import json

from briefcase.rag.llm import NoLLM, get_llm
from briefcase.rag.prompts import (
    QUESTIONS_SYSTEM,
    SUMMARY_SYSTEM,
    build_questions_prompt,
    build_summary_prompt,
)
from briefcase.repository import load_chunks
from briefcase.text_utils import split_sentences


def _broad_sample(notebook_id: str, source_ids: list[str] | None, limit: int = 16) -> list[dict]:
    """Sample chunks spread across sources so summaries reflect the whole corpus."""
    chunks = load_chunks(notebook_id, source_ids)
    if not chunks:
        return []
    by_source: dict[str, list[dict]] = {}
    for c in chunks:
        by_source.setdefault(c["source_id"], []).append(c)
    sample: list[dict] = []
    # Round-robin the earliest chunks of each source (usually the most descriptive).
    idx = 0
    while len(sample) < limit:
        added = False
        for group in by_source.values():
            if idx < len(group):
                sample.append(group[idx])
                added = True
                if len(sample) >= limit:
                    break
        if not added:
            break
        idx += 1
    return sample


def summarize(notebook_id: str, source_ids: list[str] | None = None) -> dict:
    sample = _broad_sample(notebook_id, source_ids)
    if not sample:
        return {"summary": "", "engine": "none"}
    passages = [c["text"] for c in sample]
    llm = get_llm()
    if not isinstance(llm, NoLLM):
        try:
            text = llm.complete(build_summary_prompt(passages), system=SUMMARY_SYSTEM)
            if text:
                return {"summary": text, "engine": llm.name}
        except Exception:
            pass
    # Extractive fallback: lead sentences from each sampled source.
    lines = ["**Overview (extracted from your sources)**", ""]
    seen_sources: set[str] = set()
    for c in sample:
        if c["source_id"] in seen_sources:
            continue
        seen_sources.add(c["source_id"])
        sentences = split_sentences(c["text"])
        if sentences:
            lines.append(f"- **{c['source_title']}**, {sentences[0]}")
    return {"summary": "\n".join(lines), "engine": "extractive"}


def suggested_questions(notebook_id: str, source_ids: list[str] | None = None) -> dict:
    sample = _broad_sample(notebook_id, source_ids, limit=12)
    if not sample:
        return {"questions": [], "engine": "none"}
    passages = [c["text"] for c in sample]
    llm = get_llm()
    if not isinstance(llm, NoLLM):
        try:
            raw = llm.complete(build_questions_prompt(passages), system=QUESTIONS_SYSTEM)
            start, end = raw.find("["), raw.rfind("]")
            if start != -1 and end != -1:
                questions = json.loads(raw[start : end + 1])
                questions = [str(q) for q in questions if str(q).strip()][:5]
                if questions:
                    return {"questions": questions, "engine": llm.name}
        except Exception:
            pass
    # Heuristic fallback: turn source titles into starter questions.
    titles: list[str] = []
    for c in sample:
        if c["source_title"] not in titles:
            titles.append(c["source_title"])
    questions = [f"What are the key points in \"{t}\"?" for t in titles[:4]]
    questions.append("What do these sources have in common?")
    return {"questions": questions[:5], "engine": "extractive"}

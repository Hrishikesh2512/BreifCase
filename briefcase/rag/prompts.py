from __future__ import annotations

ANSWER_SYSTEM = """You are Briefcase, a meticulous research assistant.
You answer ONLY using the numbered source passages provided by the user.
Rules:
- Ground every claim in the passages. After each claim, cite the passage(s) you used with bracketed numbers like [1] or [2][5].
- If the passages do not contain enough information to answer, reply with exactly: I don't have enough in your sources to answer that.
- Never invent facts, citations, or numbers that are not in the passages.
- Be concise and well-structured. Use short paragraphs or bullet points.
"""


def build_answer_prompt(question: str, passages: list[tuple[int, str, str]]) -> str:
    """passages: list of (marker, source_title, text)."""
    blocks = [
        f"[{marker}] (from \"{title}\")\n{text}" for marker, title, text in passages
    ]
    context = "\n\n".join(blocks)
    return (
        f"Source passages:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the passages above, with bracketed citations."
    )


SUMMARY_SYSTEM = """You are Briefcase. Summarize the provided source passages into a
clear briefing. Ground it strictly in the passages; do not add outside knowledge.
Produce: a 2-3 sentence overview, then 3-6 key-point bullets. Keep it tight."""


def build_summary_prompt(passages: list[str]) -> str:
    context = "\n\n".join(f"- {p}" for p in passages)
    return f"Source passages:\n\n{context}\n\nWrite the briefing now."


WRITE_SYSTEM = """You are Briefcase's writing assistant. Using ONLY the numbered source
passages, write clear, well-structured Markdown that fulfils the user's instruction.
Rules:
- Ground every claim in the passages and cite it with bracketed numbers like [1] or [2][5].
- Do not invent facts, names, or numbers that are not in the passages.
- Return only the new prose to add or the rewritten text. No preamble, no meta commentary.
- If the passages do not cover the request, write one short sentence saying so."""


def build_write_prompt(instruction: str, passages: list[tuple[int, str, str]], existing: str) -> str:
    blocks = [f"[{m}] (from \"{t}\")\n{txt}" for m, t, txt in passages]
    context = "\n\n".join(blocks)
    tail = f"\n\nCurrent draft (for continuity, do not repeat it):\n{existing[-1500:]}" if existing.strip() else ""
    return (
        f"Source passages:\n\n{context}{tail}\n\n"
        f"Instruction: {instruction}\n\nWrite the Markdown now, with bracketed citations."
    )


EDIT_SYSTEM = """You are Briefcase's editing assistant. You are given (a) numbered SOURCE
passages, which are read-only ground truth, and (b) the user's current DRAFT document.
Apply the user's instruction to the draft and return the COMPLETE, updated draft in Markdown.
Rules:
- Return the entire revised document, not just the changed part and not a description of changes.
- Edit the existing draft in place: keep what still fits, change or add what the instruction asks,
  remove what it says to remove. Do not simply append to the end.
- Ground any new factual claim in the source passages and cite it with bracketed numbers like [1].
- Do not invent facts, names, or numbers that are not in the passages.
- If the draft is empty, write it from scratch from the sources.
- Output only the document itself, with no preamble."""


def build_edit_prompt(instruction: str, passages: list[tuple[int, str, str]], draft: str) -> str:
    blocks = [f"[{m}] (from \"{t}\")\n{txt}" for m, t, txt in passages]
    context = "\n\n".join(blocks) if blocks else "(no sources enabled)"
    current = draft.strip() or "(the draft is currently empty)"
    return (
        f"SOURCE passages (read-only):\n\n{context}\n\n"
        f"Current DRAFT:\n\"\"\"\n{current}\n\"\"\"\n\n"
        f"Instruction: {instruction}\n\n"
        "Return the complete revised draft as Markdown."
    )


REWRITE_SYSTEM = """You are Briefcase's editing assistant. Rewrite the user's selected text
per their instruction, staying grounded ONLY in the numbered source passages. Keep or add
bracketed citations like [1]. Do not invent facts. Return only the rewritten text."""


def build_rewrite_prompt(
    instruction: str, selection: str, passages: list[tuple[int, str, str]]
) -> str:
    blocks = [f"[{m}] (from \"{t}\")\n{txt}" for m, t, txt in passages]
    context = "\n\n".join(blocks)
    return (
        f"Source passages:\n\n{context}\n\n"
        f"Selected text to rewrite:\n{selection}\n\n"
        f"Instruction: {instruction}\n\nReturn only the rewritten text."
    )


VERIFY_SYSTEM = """You are a strict fact-checker. You are given numbered CLAIMS and numbered
SOURCE passages. For each claim, decide whether the passages DIRECTLY support it.
- "supported" is true ONLY if the passages explicitly state the claim.
- If the passages contradict the claim, or simply do not contain it, "supported" is false,
  even when they discuss the same topic. (E.g. a claim naming the wrong person is NOT supported.)
Return ONLY a JSON array, one object per claim, in order:
[{"claim": <claim number>, "supported": <true|false>, "passage": <supporting passage number or null>}]
No prose, only the JSON array."""


def build_verify_prompt(claims: list[str], passages: list[tuple[int, str, str]]) -> str:
    claim_block = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    passage_block = "\n\n".join(f"[{m}] (from \"{t}\")\n{txt}" for m, t, txt in passages)
    if not passages:
        passage_block = "(no source passages available)"
    return (
        f"CLAIMS:\n{claim_block}\n\n"
        f"SOURCE passages:\n{passage_block}\n\n"
        "Return the JSON array of verdicts now."
    )


QUESTIONS_SYSTEM = """You are Briefcase. Based only on the provided passages, propose
insightful questions a reader could ask about this material. Return a JSON array of
5 short question strings and nothing else."""


def build_questions_prompt(passages: list[str]) -> str:
    context = "\n\n".join(f"- {p}" for p in passages)
    return f"Source passages:\n\n{context}\n\nReturn the JSON array of questions."

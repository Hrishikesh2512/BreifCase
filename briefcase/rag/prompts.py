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


QUESTIONS_SYSTEM = """You are Briefcase. Based only on the provided passages, propose
insightful questions a reader could ask about this material. Return a JSON array of
5 short question strings and nothing else."""


def build_questions_prompt(passages: list[str]) -> str:
    context = "\n\n".join(f"- {p}" for p in passages)
    return f"Source passages:\n\n{context}\n\nReturn the JSON array of questions."

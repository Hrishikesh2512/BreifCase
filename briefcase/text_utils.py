from __future__ import annotations

import hashlib
import re

_WHITESPACE = re.compile(r"[ \t]+")
_MULTINEWLINE = re.compile(r"\n{3,}")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(0-9\"'])")
_WORD = re.compile(r"[a-z0-9]+")

# A small English stopword list, enough to keep BM25 and extractive matching
# from being fooled by function words shared between unrelated sentences.
STOPWORDS = frozenset(
    """a an and are as at be but by for from has have he her his i in into is it its
    of on or she that the their them they this to was were what when where which who
    will with you your our we do does did not no so if then than these those over under
    about above below between out up down off again further once here there all any both
    each few more most other some such only own same too very can just how why""".split()
)


def count_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (~4 chars/token)."""
    return max(1, round(len(text) / 4))


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE.sub(" ", text)
    text = _MULTINEWLINE.sub("\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue
        parts.extend(s.strip() for s in _SENTENCE_SPLIT.split(block) if s.strip())
    return parts


def tokenize_words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokens with stopwords and single characters removed, used for lexical
    matching (BM25 / extractive overlap) so function words don't create noise."""
    return [t for t in _WORD.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_hash(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def snippet(text: str, limit: int = 240) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"

from __future__ import annotations

from dataclasses import dataclass

from briefcase.config import get_settings
from briefcase.text_utils import count_tokens, split_sentences


@dataclass
class RawChunk:
    text: str
    ordinal: int
    token_count: int


class Chunker:
    """Sentence-aware fixed-size chunking with overlap.

    Adapted from Pragya's chunker but dependency-free: groups sentences up to a
    target token budget, carrying a small overlap tail between chunks so context
    isn't severed at boundaries.
    """

    def __init__(self) -> None:
        self._cfg = get_settings().chunking

    def chunk(self, text: str) -> list[RawChunk]:
        sentences = split_sentences(text)
        if not sentences:
            return []

        groups: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0
        for sentence in sentences:
            tokens = count_tokens(sentence)
            # A single very long sentence: hard-split it.
            if tokens > self._cfg.max_tokens:
                if current:
                    groups.append(current)
                    current, current_tokens = [], 0
                groups.extend([w] for w in self._hard_split(sentence))
                continue
            if current_tokens + tokens > self._cfg.target_tokens and current:
                groups.append(current)
                current = self._overlap_tail(current) + [sentence]
                current_tokens = sum(count_tokens(s) for s in current)
            else:
                current.append(sentence)
                current_tokens += tokens
        if current:
            groups.append(current)

        chunks: list[RawChunk] = []
        for ordinal, group in enumerate(groups):
            body = " ".join(group)
            chunks.append(RawChunk(body, ordinal, count_tokens(body)))
        return chunks

    def _overlap_tail(self, sentences: list[str]) -> list[str]:
        tail: list[str] = []
        tokens = 0
        for sentence in reversed(sentences):
            t = count_tokens(sentence)
            if tokens + t > self._cfg.overlap_tokens:
                break
            tail.insert(0, sentence)
            tokens += t
        return tail

    def _hard_split(self, sentence: str) -> list[str]:
        words = sentence.split(" ")
        approx_words = max(1, self._cfg.target_tokens * 4 // 6)  # ~tokens -> words
        return [
            " ".join(words[i : i + approx_words])
            for i in range(0, len(words), approx_words)
        ]

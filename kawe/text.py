"""Sentence splitting shared by passage chunking and speech synthesis."""

from __future__ import annotations

import re
from typing import List

# Splits after ., ! or ? (optionally followed by a closing quote or bracket) and whitespace.
_SENTENCE_END = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"'”’)\]]))\s+")


def split_sentences(text: str) -> List[str]:
    text = " ".join(text.split())
    return [s for s in _SENTENCE_END.split(text) if s] if text else []


def split_text(text: str, max_chars: int) -> List[str]:
    """Splits text at sentence ends into pieces of at most max_chars characters."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text] if text else []
    chunks, current = [], ""
    for sentence in split_sentences(text):
        while len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks

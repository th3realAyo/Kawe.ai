"""Ask the document: grounded answers to a question, each traceable to the passages it used.

Prototype of the text version:
  1. Translate the question to English (if asked in Yoruba, Igbo or Hausa).
  2. Retrieve the closest passages with the embeddings already computed for the summary.
  3. Answer under a grounding prompt that allows only what the passages say; the model replies
     NOT_FOUND when they don't answer the question.
  4. Translate the answer back and return it with the passages it cites.

Questions and answers are never stored or logged; the caller logs only that a question was asked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from kawe.modal_client import ModalServiceError
from kawe.ranking import Chunk, EmbedFn

TOP_K = 3
# Below this cosine similarity (BGE-M3) no passage is treated as relevant. A first guess that
# the planned evaluation (questions with and without answers in the paper) should calibrate.
MIN_SIMILARITY = 0.35
MAX_QUESTION_CHARS = 1000
NOT_FOUND = "NOT_FOUND"
NOT_COVERED = "This document doesn't say."

AnswerFn = Callable[[str, List[str]], str]
TranslateFn = Callable[[str, str], str]

_CITATION = re.compile(r"\[(\d+)\]")


class QuestionError(Exception):
    """The question could not be answered because a step failed (not because the document is silent)."""


@dataclass
class RetrievedPassage:
    chunk: Chunk
    similarity: float
    number: int  # [n] the answer cites


@dataclass
class Answer:
    text: str  # in the asker's language, citation markers removed
    found: bool
    cited: List[RetrievedPassage] = field(default_factory=list)
    retrieved: List[RetrievedPassage] = field(default_factory=list)
    english_text: Optional[str] = None


@dataclass
class DocumentIndex:
    chunks: List[Chunk]
    vectors: np.ndarray  # L2-normalized, one row per chunk


def build_index(chunks: Sequence[Chunk], vectors: Optional[Sequence[Sequence[float]]] = None, embed_fn: Optional[EmbedFn] = None) -> DocumentIndex:
    """Uses the summary's embeddings when given; embeds the passages otherwise."""
    if not chunks:
        raise QuestionError("The document has no text to search.")
    if vectors is None:
        if embed_fn is None:
            from kawe.modal_client import embed as embed_fn
        try:
            vectors = embed_fn([c.text for c in chunks])
        except ModalServiceError as e:
            raise QuestionError(f"Could not index the document: {e}") from e
    return DocumentIndex(list(chunks), np.asarray(vectors, dtype=float))


def retrieve(index: DocumentIndex, question_vector: Sequence[float], top_k: int = TOP_K, min_similarity: float = MIN_SIMILARITY) -> List[RetrievedPassage]:
    """The top_k passages above min_similarity, numbered in document order."""
    similarity = index.vectors @ np.asarray(question_vector, dtype=float)
    best = [i for i in np.argsort(-similarity)[:top_k] if similarity[i] >= min_similarity]
    best.sort(key=lambda i: index.chunks[i].chunk_index)
    return [RetrievedPassage(index.chunks[i], float(similarity[i]), n) for n, i in enumerate(best, start=1)]


def parse_answer(raw: str, retrieved: List[RetrievedPassage]):
    """Returns (text without citation markers, cited passages, found)."""
    text = raw.strip()
    if not text or text.upper().startswith(NOT_FOUND):
        return "", [], False
    by_number = {p.number: p for p in retrieved}
    cited_numbers = sorted({int(n) for n in _CITATION.findall(text) if int(n) in by_number})
    clean = re.sub(r"\s*\[\d+\]", "", text)
    clean = re.sub(r"\s+([.,;:!?])", r"\1", clean).strip()
    # An answer that cites nothing is still shown, attributed to every passage it was given.
    cited = [by_number[n] for n in cited_numbers] or list(retrieved)
    return clean, cited, True


def _default_answer(question: str, passages: List[str]) -> str:
    from kawe.modal_client import answer

    return answer(question, passages)


def _default_translate(text: str, language: str) -> str:
    from kawe.modal_client import translate

    return translate(text, language)


def answer_question(
    question: str,
    index: DocumentIndex,
    language: str = "English",
    embed_fn: Optional[EmbedFn] = None,
    answer_fn: Optional[AnswerFn] = None,
    translate_fn: Optional[TranslateFn] = None,
) -> Answer:
    """Answers `question`, asked in `language`, from the document only."""
    question = " ".join(question.split())
    if not question:
        raise QuestionError("Type a question first.")
    if len(question) > MAX_QUESTION_CHARS:
        raise QuestionError(f"Questions can be at most {MAX_QUESTION_CHARS} characters.")
    answer_fn = answer_fn or _default_answer
    translate_fn = translate_fn or _default_translate
    if embed_fn is None:
        from kawe.modal_client import embed as embed_fn
    english = language.lower() == "english"

    try:
        english_question = question if english else translate_fn(question, "English")
        question_vector = embed_fn([english_question])[0]
        retrieved = retrieve(index, question_vector)
        if not retrieved:
            return Answer(_localized_not_covered(language, translate_fn), found=False)
        raw = answer_fn(english_question, [p.chunk.text for p in retrieved])
        text, cited, found = parse_answer(raw, retrieved)
        if not found:
            return Answer(_localized_not_covered(language, translate_fn), found=False, retrieved=retrieved)
        localized = text if english else translate_fn(text, language)
    except ModalServiceError as e:
        raise QuestionError(str(e)) from e
    return Answer(localized, found=True, cited=cited, retrieved=retrieved, english_text=text)


def _localized_not_covered(language: str, translate_fn: TranslateFn) -> str:
    if language.lower() == "english":
        return NOT_COVERED
    try:
        return translate_fn(NOT_COVERED, language)
    except ModalServiceError:
        return NOT_COVERED

"""Grounded summarization: rank passages, rewrite the top ones, keep each passage as a citation."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from kawe.modal_client import ModalServiceError
from kawe.ranking import DEFAULT_TOP_N, Chunk, EmbedFn, centrality_scores, select_top, split_into_chunks

log = logging.getLogger(__name__)

SummarizeFn = Callable[[str, str], str]
SummarizeBatchFn = Callable[[List[str], str], List[str]]

# langdetect codes -> names used in the model prompt
LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "pl": "Polish",
    "tr": "Turkish",
    "ru": "Russian",
    "nl": "Dutch",
    "cs": "Czech",
    "ar": "Arabic",
    "zh-cn": "Chinese",
    "ja": "Japanese",
    "hu": "Hungarian",
    "ko": "Korean",
    "hi": "Hindi",
}

_PREAMBLE = re.compile(
    r"^\s*(?:here\s+is|here's|below\s+is|sure[,!.]?\s*here\s+is)\b[^\n:]{0,100}?\bsummar\w*[^\n:]*:\s*",
    re.IGNORECASE,
)


class GroundedSummarizerError(Exception):
    """No summary could be produced for the document."""


@dataclass
class GroundedSection:
    summary_text: str
    location_label: str
    source_excerpt: str
    order: int
    chunk_index: int = -1
    score: Optional[float] = None


@dataclass
class GroundedSummaryResult:
    full_text: str
    sections: List[GroundedSection] = field(default_factory=list)
    language_code: str = "en"
    language_name: str = "English"
    skipped_chunks: int = 0
    total_chunks: int = 0
    timings: Dict[str, float] = field(default_factory=dict)
    # Every passage of the document and, when ranking ran, its embedding; reused for questions.
    chunks: List[Chunk] = field(default_factory=list)
    vectors: Optional[List[List[float]]] = None


def clean_summary(text: str) -> str:
    """Drops chat preambles such as "Here is a summary of the passage:" and wrapping quotes."""
    cleaned = _PREAMBLE.sub("", text, count=1).strip()
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1] and cleaned[0] in "\"'":
        cleaned = cleaned[1:-1].strip()
    return cleaned or text.strip()


def _default_summarize(text: str, language_name: str) -> str:
    from kawe.modal_client import summarize

    return summarize(text, language_name)


def summarize_chunk(
    chunk: Chunk,
    language_name: str,
    summarize_fn: Optional[SummarizeFn] = None,
    retries: int = 1,
) -> Optional[str]:
    """Rewrites one passage; returns None if it still fails after retrying."""
    summarize_fn = summarize_fn or _default_summarize
    for attempt in range(retries + 1):
        try:
            return clean_summary(summarize_fn(chunk.text, language_name))
        except ModalServiceError as e:
            if attempt < retries and e.retryable:
                time.sleep(1.5)
                continue
            log.warning("Skipping passage %s (%s) after %d attempt(s): %s", chunk.chunk_index, chunk.location_label, attempt + 1, e)
            return None
    return None


def _default_summarize_batch(texts: List[str], language_name: str) -> List[str]:
    from kawe.modal_client import summarize_batch

    return summarize_batch(texts, language_name)


def _batch_rewrite(chunks: Sequence[Chunk], language_name: str, batch_fn: SummarizeBatchFn) -> Optional[List[str]]:
    """All rewrites from one batched request, or None if the request fails."""
    try:
        return [clean_summary(text) for text in batch_fn([c.text for c in chunks], language_name)]
    except ModalServiceError as e:
        log.warning("Batched rewrite failed, rewriting passages one at a time: %s", e)
        return None


def rewrite_chunks(
    chunks: Sequence[Chunk],
    language_name: str,
    summarize_fn: Optional[SummarizeFn] = None,
    scores: Optional[Dict[int, float]] = None,
    summarize_batch_fn: Optional[SummarizeBatchFn] = None,
) -> Tuple[List[GroundedSection], int, List[float]]:
    """Returns (sections, number of skipped passages, seconds per passage).

    With summarize_batch_fn, every passage is rewritten in one request (seconds are then the
    batch time spread evenly); if that request fails, passages are rewritten one at a time.
    """
    batched: Optional[List[str]] = None
    if summarize_batch_fn is not None and chunks:
        start = time.perf_counter()
        batched = _batch_rewrite(chunks, language_name, summarize_batch_fn)
        per_chunk = (time.perf_counter() - start) / len(chunks)

    sections, skipped, seconds = [], 0, []
    for i, chunk in enumerate(chunks):
        if batched is not None:
            text = batched[i]
            seconds.append(per_chunk)
        else:
            start = time.perf_counter()
            text = summarize_chunk(chunk, language_name, summarize_fn)
            seconds.append(time.perf_counter() - start)
        if not text:
            skipped += 1
            continue
        sections.append(
            GroundedSection(
                summary_text=text,
                location_label=chunk.location_label,
                source_excerpt=chunk.text,
                order=chunk.order,
                chunk_index=chunk.chunk_index,
                score=(scores or {}).get(chunk.chunk_index),
            )
        )
    return sections, skipped, seconds


def generate_grounded_summary(
    segments: Sequence,
    language_code: str = "en",
    top_n: int = DEFAULT_TOP_N,
    embed_fn: Optional[EmbedFn] = None,
    summarize_fn: Optional[SummarizeFn] = None,
    summarize_batch_fn: Optional[SummarizeBatchFn] = None,
) -> GroundedSummaryResult:
    """Ranks passages, picks top_n with a redundancy penalty and rewrites each one.

    With neither summarize_fn nor summarize_batch_fn, the deployed endpoints are used: one
    batched request, falling back to one request per passage.
    """
    if summarize_fn is None and summarize_batch_fn is None:
        summarize_batch_fn = _default_summarize_batch
    language_name = LANGUAGE_NAMES.get(language_code, "English")
    chunks = split_into_chunks(segments)
    if not chunks:
        raise GroundedSummarizerError("No content available to summarize.")

    timings: Dict[str, float] = {}
    scores: Dict[int, float] = {}
    vectors = None
    if len(chunks) <= top_n:
        selected = sorted(chunks, key=lambda c: c.order)
    else:
        if embed_fn is None:
            from kawe.modal_client import embed as embed_fn
        start = time.perf_counter()
        try:
            vectors = embed_fn([c.text for c in chunks])
        except ModalServiceError as e:
            raise GroundedSummarizerError(f"Embedding failed: {e}") from e
        timings["embed"] = time.perf_counter() - start

        start = time.perf_counter()
        values = centrality_scores(vectors)
        selected = select_top(chunks, values, top_n, vectors)
        timings["rank"] = time.perf_counter() - start
        scores = {c.chunk_index: v for c, v in zip(chunks, values)}

    sections, skipped, seconds = rewrite_chunks(selected, language_name, summarize_fn, scores, summarize_batch_fn)
    timings["rewrite"] = sum(seconds)
    if not sections:
        raise GroundedSummarizerError("Every passage failed to summarize; the service may be unreachable.")

    return GroundedSummaryResult(
        full_text=" ".join(s.summary_text for s in sections),
        sections=sections,
        language_code=language_code,
        language_name=language_name,
        skipped_chunks=skipped,
        total_chunks=len(chunks),
        timings=timings,
        chunks=list(chunks),
        vectors=vectors,
    )

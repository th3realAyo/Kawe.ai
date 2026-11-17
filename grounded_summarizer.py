"""
grounded_summarizer.py

Top-level orchestration for the grounded RAG summarization pipeline:

    ExtractedSegment list (from doc_loader.py)
        -> split_into_chunks + rank_chunks (chunk_ranker.py)
        -> per-chunk abstractive rewrite via Modal's summarize endpoint
        -> assembled GroundedSummaryResult with citations back to source

This is the module main.py calls directly.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List

import requests

from chunk_ranker import DEFAULT_TOP_N, Chunk, rank_chunks, split_into_chunks


class GroundedSummarizerError(Exception):
    """Raised when the pipeline can't produce a summary at all (e.g. every
    chunk failed). Per-chunk failures are handled gracefully and don't
    raise this — see _summarize_one_chunk."""


# Maps langdetect's ISO codes to full language names, since the LLM prompt
# reads more reliably with "French" than with "fr". Covers the languages
# XTTS-v2 supports downstream, so summary language and TTS language stay
# in lockstep end-to-end.
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


@dataclass
class GroundedSection:
    """One piece of the final summary, traceable to exactly one source chunk."""

    summary_text: str
    location_label: str
    source_excerpt: str
    order: int


@dataclass
class GroundedSummaryResult:
    full_text: str
    sections: List[GroundedSection] = field(default_factory=list)
    language_code: str = "en"
    language_name: str = "English"
    skipped_chunks: int = 0  # chunks that failed even after retry


def _summarize_via_modal(text: str, language_name: str, max_new_tokens: int = 120) -> str:
    endpoint = os.environ["MODAL_SUMMARIZE_URL"]
    token = os.environ["RAG_AUTH_TOKEN"]

    response = requests.post(
        endpoint,
        json={"text": text, "language": language_name, "max_new_tokens": max_new_tokens},
        headers={"Authorization": f"Bearer {token}"},
        timeout=600,  # first-ever cold start can involve a large model download
    )
    if response.status_code != 200:
        raise GroundedSummarizerError(
            f"Summarize request failed ({response.status_code}): {response.text}"
        )
    return response.json()["summary"]


def _summarize_one_chunk(chunk: Chunk, language_name: str, retries: int = 1) -> str | None:
    """Summarizes a single chunk, retrying once on transient failure.
    Returns None (rather than raising) if it still fails, so one bad chunk
    doesn't take down the whole document's summary."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            return _summarize_via_modal(chunk.text, language_name)
        except (GroundedSummarizerError, requests.RequestException) as e:
            last_error = e
            if attempt < retries:
                time.sleep(1.5)  # brief backoff before retrying
    print(f"[grounded_summarizer] Skipping chunk (page {chunk.location_label}) after "
          f"{retries + 1} attempts: {last_error}")
    return None


def generate_grounded_summary(
    segments: List,  # List[ExtractedSegment] from doc_loader.py
    language_code: str = "en",
    top_n: int = DEFAULT_TOP_N,
) -> GroundedSummaryResult:
    language_name = LANGUAGE_NAMES.get(language_code, "English")

    all_chunks = split_into_chunks(segments)
    top_chunks = rank_chunks(all_chunks, top_n=top_n)

    if not top_chunks:
        raise GroundedSummarizerError("No content available to summarize.")

    sections: List[GroundedSection] = []
    skipped = 0

    for chunk in top_chunks:
        summary_text = _summarize_one_chunk(chunk, language_name)
        if summary_text is None:
            skipped += 1
            continue
        sections.append(
            GroundedSection(
                summary_text=summary_text,
                location_label=chunk.location_label,
                source_excerpt=chunk.text,
                order=chunk.order,
            )
        )

    if not sections:
        raise GroundedSummarizerError(
            "All chunks failed to summarize — the summarization service may be unreachable."
        )

    full_text = " ".join(section.summary_text for section in sections)

    return GroundedSummaryResult(
        full_text=full_text,
        sections=sections,
        language_code=language_code,
        language_name=language_name,
        skipped_chunks=skipped,
    )

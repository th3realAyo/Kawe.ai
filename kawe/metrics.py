"""Upload limits and summary statistics shown in the app."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
WORDS_PER_MINUTE = 150
TOP_N_MIN, TOP_N_MAX = 3, 40  # TOP_N_MAX stays within the batch endpoint's limit


@dataclass(frozen=True)
class SummaryMetrics:
    source_words: int
    summary_words: int
    compression_percentage: int
    estimated_listening_minutes: int


def validate_upload_size(size_bytes: int, max_bytes: int = MAX_UPLOAD_BYTES) -> None:
    if size_bytes < 0:
        raise ValueError("Upload size cannot be negative.")
    if size_bytes > max_bytes:
        raise ValueError(f"This file is too large. Upload a file up to {max_bytes // (1024 * 1024)} MB.")


def build_summary_metrics(source_words: int, summary_text: str) -> SummaryMetrics:
    if source_words < 0:
        raise ValueError("Source word count cannot be negative.")
    summary_words = len(summary_text.split())
    return SummaryMetrics(
        source_words=source_words,
        summary_words=summary_words,
        compression_percentage=round(summary_words / source_words * 100) if source_words else 0,
        estimated_listening_minutes=ceil(summary_words / WORDS_PER_MINUTE) if summary_words else 0,
    )


def summary_top_n(percentage: int, chunk_count: int) -> int:
    """Passages to rewrite: `percentage` of the document's passages, within [TOP_N_MIN, TOP_N_MAX].

    The result can exceed chunk_count for very short documents; every passage is then used.
    """
    return max(TOP_N_MIN, min(TOP_N_MAX, round(percentage / 100 * chunk_count)))

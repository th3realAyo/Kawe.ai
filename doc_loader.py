"""
doc_loader.py

Unified text extraction for PDF and Word (.docx) documents, producing a
list of ExtractedSegment objects that preserve enough location metadata
(page number for PDFs, a pseudo-page grouping for Word docs) to support
citation back to source later in the pipeline.

This module only extracts raw text — no chunking, cleaning, or scoring.
That happens downstream in chunk_ranker.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List


class UnsupportedFileTypeError(Exception):
    """Raised when a file extension isn't one of the supported document types."""


class DocumentExtractionError(Exception):
    """Raised when a supported file type fails to open or extract (corrupt,
    password-protected, empty, etc.)."""


@dataclass
class ExtractedSegment:
    """One location-addressable piece of a document.

    For PDFs this is one page. For Word docs, which have no fixed page
    boundaries at the file-format level, this is a group of paragraphs
    sized to approximate a page — good enough for citation purposes
    ("around paragraph group 4") even if it won't match the reader's
    on-screen page numbers exactly.
    """

    text: str
    location_label: str  # e.g. "Page 3" or "Section 4"
    order: int  # 0-indexed original document order


# Word docs don't have real pages in the file itself (pagination happens at
# render time), so we approximate one "page" as this many words — close
# enough for citation granularity without needing a layout engine.
WORDS_PER_PSEUDO_PAGE = 350


def extract_pdf(file_path: str) -> List[ExtractedSegment]:
    import pdfplumber

    segments: List[ExtractedSegment] = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = (page.extract_text() or "").strip()
                if text:
                    segments.append(
                        ExtractedSegment(
                            text=text,
                            location_label=f"Page {i + 1}",
                            order=i,
                        )
                    )
    except Exception as e:
        raise DocumentExtractionError(f"Failed to extract PDF: {e}") from e

    if not segments:
        raise DocumentExtractionError(
            "No extractable text found in PDF (it may be scanned/image-only)."
        )
    return segments


def extract_docx(file_path: str) -> List[ExtractedSegment]:
    import docx  # python-docx

    try:
        document = docx.Document(file_path)
        paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    except Exception as e:
        raise DocumentExtractionError(f"Failed to extract Word document: {e}") from e

    if not paragraphs:
        raise DocumentExtractionError("No extractable text found in Word document.")

    segments: List[ExtractedSegment] = []
    current_words: List[str] = []
    page_index = 0

    def flush():
        nonlocal current_words, page_index
        if current_words:
            segments.append(
                ExtractedSegment(
                    text=" ".join(current_words),
                    location_label=f"Section {page_index + 1}",
                    order=page_index,
                )
            )
            page_index += 1
            current_words = []

    for para in paragraphs:
        current_words.extend(para.split())
        if len(current_words) >= WORDS_PER_PSEUDO_PAGE:
            flush()
    flush()  # trailing partial section

    return segments


def extract_document(file_path: str) -> List[ExtractedSegment]:
    """Dispatches to the correct extractor based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_pdf(file_path)
    elif ext == ".docx":
        return extract_docx(file_path)
    elif ext == ".doc":
        raise UnsupportedFileTypeError(
            "Legacy .doc format isn't supported — please convert to .docx or PDF."
        )
    else:
        raise UnsupportedFileTypeError(f"Unsupported file type: {ext}")

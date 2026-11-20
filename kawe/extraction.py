"""Text extraction from PDF and Word documents."""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

WORDS_PER_PSEUDO_PAGE = 350
MIN_WORDS_BEFORE_HEADING_BREAK = 100
_HYPHEN_BREAK = "￾"  # PDFium marks hyphenated line breaks with this character


class UnsupportedFileTypeError(Exception):
    pass


class DocumentExtractionError(Exception):
    pass


@dataclass
class ExtractedSegment:
    text: str
    location_label: str  # "Page 3" for PDFs, 'Section 4 ("Methods")' for Word files
    order: int


def clean_page_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(_HYPHEN_BREAK + "\n", "").replace(_HYPHEN_BREAK, "").replace("­", "")
    lines = (" ".join(line.split()) for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def strip_running_lines(pages: List[str], min_pages: int = 4) -> List[str]:
    """Removes headers, footers and page numbers that repeat at the edges of most pages."""
    if len(pages) < min_pages:
        return pages

    def key(line: str) -> str:
        return re.sub(r"\d+", "#", line.lower()).strip()

    def is_edge(i: int, n: int) -> bool:
        return i < 2 or i >= n - 2

    split_pages = [page.split("\n") for page in pages]
    counts = Counter()
    for lines in split_pages:
        counts.update({key(line) for i, line in enumerate(lines) if is_edge(i, len(lines))})
    running = {k for k, n in counts.items() if n >= len(pages) / 2}

    return [
        "\n".join(line for i, line in enumerate(lines) if not (is_edge(i, len(lines)) and key(line) in running))
        for lines in split_pages
    ]


def read_pdf_pages(file_path: str) -> List[str]:
    """Raw text of each page, in content-stream order (keeps columns intact)."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(file_path)
    try:
        pages = []
        for page in pdf:
            textpage = page.get_textpage()
            pages.append(textpage.get_text_range())
            textpage.close()
            page.close()
        return pages
    finally:
        pdf.close()


def extract_pdf(file_path: str) -> List[ExtractedSegment]:
    try:
        raw_pages = read_pdf_pages(file_path)
    except Exception as e:
        raise DocumentExtractionError(f"Failed to extract PDF: {e}") from e

    pages = strip_running_lines([clean_page_text(p) for p in raw_pages])
    segments = [
        ExtractedSegment(text=text, location_label=f"Page {i + 1}", order=i)
        for i, text in enumerate(pages)
        if text.strip()
    ]
    if not segments:
        raise DocumentExtractionError("No extractable text found in PDF (it may be scanned or image-only).")
    return segments


def _docx_blocks(document):
    """Yields (text, heading) for each paragraph and table in body order; heading is True for headings."""
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            paragraph = Paragraph(child, document)
            text = paragraph.text.strip()
            if text:
                style = (paragraph.style.name if paragraph.style is not None else "") or ""
                yield text, style.startswith("Heading") or style == "Title"
        elif tag == "tbl":
            for row in Table(child, document).rows:
                cells = []
                for cell in row.cells:  # merged cells repeat, so keep each text once
                    text = " ".join(cell.text.split())
                    if text and text not in cells:
                        cells.append(text)
                if cells:
                    yield " | ".join(cells), False


def _section_label(number: int, heading: Optional[str]) -> str:
    if not heading:
        return f"Section {number}"
    short = heading if len(heading) <= 60 else heading[:57].rstrip() + "..."
    return f'Section {number} ("{short}")'


def extract_docx(file_path: str) -> List[ExtractedSegment]:
    """Groups paragraphs and table rows into ~350-word sections, since .docx files have no fixed pages.

    A heading starts a new section once the current one has MIN_WORDS_BEFORE_HEADING_BREAK words,
    and each section is labelled with the heading it falls under, so citations can be found.
    """
    import docx

    try:
        blocks = list(_docx_blocks(docx.Document(file_path)))
    except Exception as e:
        raise DocumentExtractionError(f"Failed to extract Word document: {e}") from e
    if not blocks:
        raise DocumentExtractionError("No extractable text found in Word document.")

    segments: List[ExtractedSegment] = []
    current: List[str] = []
    words = 0
    heading: Optional[str] = None  # heading the current section falls under
    latest_heading: Optional[str] = None

    def flush():
        nonlocal current, words, heading
        if current:
            segments.append(ExtractedSegment("\n".join(current), _section_label(len(segments) + 1, heading), len(segments)))
        current, words, heading = [], 0, latest_heading

    for text, is_heading in blocks:
        if is_heading:
            if words >= MIN_WORDS_BEFORE_HEADING_BREAK:
                latest_heading = text
                flush()
            else:
                latest_heading = text
                if not current:
                    heading = text
        current.append(text)
        words += len(text.split())
        if words >= WORDS_PER_PSEUDO_PAGE:
            flush()
    flush()
    return segments


def extract_document(file_path: str) -> List[ExtractedSegment]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_pdf(file_path)
    if ext == ".docx":
        return extract_docx(file_path)
    if ext == ".doc":
        raise UnsupportedFileTypeError("Legacy .doc files are not supported; convert to .docx or PDF.")
    raise UnsupportedFileTypeError(f"Unsupported file type: {ext}")

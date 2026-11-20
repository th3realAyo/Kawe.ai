"""Builds the test set from every PDF and Word file in eval/evaluation_papers.

Each paper's own abstract is the reference summary. The text Kàwé summarizes is
the body after the abstract, up to the reference list; the front matter, the
abstract itself, references and appendices are excluded. The body is extracted
with the app's own extractor, so the evaluation sees exactly what users see.

Also records, per paper, how the app's extractor (PDFium) compares with the
previous one (pdfplumber), for the extraction analysis.

Writes results/corpus.jsonl, results/extraction.jsonl and results/corpus_review.md.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

from common import PAPERS_DIR, RESULTS_DIR, markdown_table, write_jsonl

from kawe.extraction import extract_document

MIN_BODY_WORDS = 1500
MIN_ABSTRACT_WORDS = 60
MAX_ABSTRACT_WORDS = 600
LONG_TOKEN_CHARS = 25  # tokens longer than this are almost always words fused by lost spaces

ABSTRACT_START = re.compile(r"(?im)^[ \t]*(?:abstract|executive\s+summary)\b[ \t.:\-–—]*")
ABSTRACT_END = re.compile(
    r"(?im)^[ \t]*("
    r"key\s*words?|index\s+terms|jel\s+classification|"
    r"(?:1\.0|1\.?|i\.)?\s*introduction|"
    r"chapter\s+(?:one|1)\b|table\s+of\s+contents?|contents\s*$|"
    r"acknowledge?ments?|(?:1\.0|1\.)\s*background|background\s+(?:of|to)\s+the\s+study"
    r")\b"
    r"|^[ \t]*(?:[∗*†‡§¶]|contact\s+author|corresponding\s+author|arxiv:\d)"  # first-page footnotes
)
FOOTNOTE_LINE = re.compile(r"^\s*(?:[∗*†‡§¶]|arxiv:\d)", re.IGNORECASE)
REFERENCES = re.compile(r"(?im)^[ \t]*(?:\d+\.?\s*)?(references|bibliography|works\s+cited)[ \t:]*$")


def find_abstract(text: str) -> Optional[Tuple[int, int, int]]:
    """(heading start, abstract start, abstract end) character offsets, or None."""
    for match in ABSTRACT_START.finditer(text):
        line_end = text.find("\n", match.start())
        heading = text[match.start() : line_end if line_end != -1 else None]
        if re.search(r"\.{4,}|\s(\d{1,3}|[ivxlc]{1,6})\s*$", heading, re.IGNORECASE):
            continue  # table-of-contents entry
        start = match.end()
        end_match = ABSTRACT_END.search(text, start + 200)
        end = end_match.start() if end_match else len(text)
        words = text[start:end].split()
        if len(words) > MAX_ABSTRACT_WORDS:
            end = start + len(" ".join(text[start:].split()[:MAX_ABSTRACT_WORDS]))
            words = text[start:end].split()
        if len(words) >= MIN_ABSTRACT_WORDS:
            return match.start(), start, end
    return None


def cut_references(pages: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Drops the reference list and everything after it (appendices, checklists)."""
    for i in range(int(len(pages) * 0.25), len(pages)):
        match = REFERENCES.search(pages[i][1])
        if match:
            kept = pages[:i] + [(pages[i][0], pages[i][1][: match.start()])]
            return [(label, text) for label, text in kept if text.strip()]
    return pages


def split_paper(segments) -> dict:
    offsets, full = [], ""
    for seg in segments:
        offsets.append((len(full), seg.location_label))
        full += seg.text + "\n"
    span = find_abstract(full)
    if not span:
        raise ValueError("no Abstract heading found")
    heading_start, start, end = span

    pages = []
    for k, (page_start, label) in enumerate(offsets):
        page_end = offsets[k + 1][0] if k + 1 < len(offsets) else len(full)
        lo = max(page_start, end)  # keep only text after the abstract
        if lo < page_end:
            text = full[lo:page_end].strip()
            if text:
                pages.append((label, text))
    pages = cut_references(pages)
    if pages:  # first-page footnotes (author contacts, arXiv stamp) end up at the top of the body
        label, text = pages[0]
        text = "\n".join(line for line in text.split("\n") if not FOOTNOTE_LINE.match(line))
        pages = ([(label, text)] if text.strip() else []) + pages[1:]
    return {
        "front_matter": " ".join(full[:heading_start].split()),
        "reference": " ".join(full[start:end].split()),
        "pages": [{"label": label, "text": text} for label, text in pages],
    }


def extraction_stats(path: Path) -> dict:
    """Word counts and fused-token rates for PDFium (app) and pdfplumber (previous extractor)."""
    import pdfplumber

    from kawe.extraction import clean_page_text, read_pdf_pages

    def stats(texts: List[str], seconds: float) -> dict:
        tokens = " ".join(texts).split()
        long_tokens = sum(len(t) > LONG_TOKEN_CHARS for t in tokens)
        return {"words": len(tokens), "fused_tokens": long_tokens, "fused_rate": long_tokens / max(len(tokens), 1), "seconds": seconds}

    start = time.perf_counter()
    pdfium_pages = [clean_page_text(p) for p in read_pdf_pages(str(path))]
    pdfium = stats(pdfium_pages, time.perf_counter() - start)
    start = time.perf_counter()
    with pdfplumber.open(str(path)) as pdf:
        plumber_pages = [page.extract_text() or "" for page in pdf.pages]
    plumber = stats(plumber_pages, time.perf_counter() - start)
    return {"pages": len(pdfium_pages), "pdfium": pdfium, "pdfplumber": plumber}


def doc_id_for(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")[:48]


def main():
    files = sorted(p for p in PAPERS_DIR.iterdir() if p.suffix.lower() in {".pdf", ".docx"})
    if not files:
        raise SystemExit(f"No PDF or Word files in {PAPERS_DIR}")

    records, extraction, review, problems = [], [], [], []
    for path in files:
        try:
            start = time.perf_counter()
            segments = extract_document(str(path))
            extract_seconds = time.perf_counter() - start
            parts = split_paper(segments)
        except Exception as e:
            problems.append([path.name, str(e)])
            continue
        body_words = sum(len(p["text"].split()) for p in parts["pages"])
        if body_words < MIN_BODY_WORDS:
            problems.append([path.name, f"only {body_words} body words"])
            continue
        record = {
            "doc_id": doc_id_for(path),
            "file": path.name,
            "front_matter": parts["front_matter"][:400],
            "reference": parts["reference"],
            "pages": parts["pages"],
            "body_words": body_words,
            "source_pages": len(segments),
            "extract_seconds": round(extract_seconds, 3),
        }
        records.append(record)
        if path.suffix.lower() == ".pdf":
            extraction.append({"doc_id": record["doc_id"], **extraction_stats(path)})
        ref = record["reference"].split()
        review.append([
            record["doc_id"], len(ref), body_words, len(record["pages"]),
            " ".join(ref[:12]) + " …", "… " + " ".join(ref[-12:]),
            record["pages"][0]["text"][:60].replace("\n", " ") + " …",
        ])

    write_jsonl(RESULTS_DIR / "corpus.jsonl", records)
    write_jsonl(RESULTS_DIR / "extraction.jsonl", extraction)
    report = [
        f"# Test set: {len(records)} of {len(files)} files\n",
        markdown_table(["Paper", "Abstract words", "Body words", "Body pages", "Abstract starts", "Abstract ends", "Body starts"], review),
    ]
    if problems:
        report += ["\n## Problems\n", markdown_table(["File", "Problem"], problems)]
    (RESULTS_DIR / "corpus_review.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()

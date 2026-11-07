"""Step 1: turn your Nigerian documents into a test set.

Put the documents (text-based PDF or .docx, not scans) in
eval/nigerian_docs/ and list them in eval/nigerian_docs/manifest.csv
(see manifest_template.csv). Every document must contain its own summary
written by its authors: an "Executive Summary" (reports) or an "Abstract"
(theses, journal articles). That summary becomes the human reference; it
is removed from the text Kàwé summarizes, so Kàwé can't copy it.

The script finds the summary automatically from its heading. Always check
eval/results/corpus_review.md afterwards: if a summary was found wrongly,
fill in the summary_pages column of the manifest (e.g. "3-5") and run
again. Those pages are then used as the reference and left out of the body.

Usage:
    python eval/prepare_corpus.py
"""

from __future__ import annotations

import csv
import re
from typing import List, Optional, Tuple

from common import EVAL_DIR, RESULTS_DIR, markdown_table

DOCS_DIR = EVAL_DIR / "nigerian_docs"
MANIFEST = DOCS_DIR / "manifest.csv"
CORPUS = RESULTS_DIR / "corpus.jsonl"

MIN_BODY_WORDS = 1500
MIN_SUMMARY_WORDS = 60
MAX_SUMMARY_WORDS = 3000

SUMMARY_START = re.compile(r"(?im)^[ \t]*(executive\s+summary|abstract)\b[ \t.:\-–]*")
# Headings that usually come right after an abstract or executive summary.
SUMMARY_END = re.compile(
    r"(?im)^[ \t]*("
    r"key\s*words?|index\s+terms|"
    r"(?:1\.0|1\.|i\.|chapter\s+(?:one|1))?\s*introduction|"
    r"chapter\s+(?:one|1)\b|"
    r"table\s+of\s+contents|contents\s*$|"
    r"acknowledge?ments?|dedication|"
    r"(?:1\.0|1\.)\s*background|background\s+(?:of|to)\s+the\s+study|"
    r"list\s+of\s+(?:tables|figures|abbreviations|acronyms)"
    r")\b"
)
REFERENCES = re.compile(r"(?im)^[ \t]*(references|bibliography|works\s+cited)[ \t:]*$")


def read_pages(path) -> List[Tuple[str, str]]:
    """Returns (label, text) per page. PDFs use the app's own extractor. For
    Word files the paragraphs are grouped into the same ~350-word
    "Section N" pseudo-pages as the app, but line breaks between paragraphs
    are kept so headings like "ABSTRACT" can be found."""
    from doc_loader import WORDS_PER_PSEUDO_PAGE, extract_document

    if str(path).lower().endswith(".docx"):
        import docx

        paragraphs = [p.text.strip() for p in docx.Document(str(path)).paragraphs if p.text.strip()]
        pages, current, words = [], [], 0
        for para in paragraphs:
            current.append(para)
            words += len(para.split())
            if words >= WORDS_PER_PSEUDO_PAGE:
                pages.append((f"Section {len(pages) + 1}", "\n".join(current)))
                current, words = [], 0
        if current:
            pages.append((f"Section {len(pages) + 1}", "\n".join(current)))
        return pages
    return [(seg.location_label, seg.text) for seg in extract_document(str(path))]


def parse_pages(spec: str) -> Optional[set]:
    """'3-5' or '2,4' -> {3,4,5} / {2,4} (1-based)."""
    spec = (spec or "").strip()
    if not spec:
        return None
    pages = set()
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        pages.update(range(int(a), int(b or a) + 1))
    return pages


def find_summary(full_text: str) -> Optional[Tuple[int, int]]:
    """Character span of the abstract / executive summary body, or None."""
    for start_match in SUMMARY_START.finditer(full_text):
        start = start_match.end()
        # Skip table-of-contents entries like "Executive Summary ........ iv".
        line_end = full_text.find("\n", start_match.start())
        heading_line = full_text[start_match.start() : line_end if line_end != -1 else None]
        if re.search(r"\.{4,}|\s(\d{1,3}|[ivxlc]{1,6})\s*$", heading_line, re.IGNORECASE):
            continue
        end_match = SUMMARY_END.search(full_text, start + 200)
        end = end_match.start() if end_match else len(full_text)
        words = len(full_text[start:end].split())
        if words > MAX_SUMMARY_WORDS:  # no clear end: take a capped prefix
            end = start + len(" ".join(full_text[start:].split()[:MAX_SUMMARY_WORDS]))
        if len(full_text[start:end].split()) >= MIN_SUMMARY_WORDS:
            return start, end
    return None


def strip_references(pages: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Drops the reference list if it starts in the last 40% of the document."""
    cutoff = int(len(pages) * 0.6)
    for i in range(len(pages) - 1, cutoff - 1, -1):
        match = REFERENCES.search(pages[i][1])
        if match:
            kept = pages[:i] + [(pages[i][0], pages[i][1][: match.start()])]
            return [(label, text) for label, text in kept if text.strip()]
    return pages


def build_record(row: dict) -> dict:
    path = DOCS_DIR / row["file"]
    pages = read_pages(path)
    manual = parse_pages(row.get("summary_pages", ""))

    if manual:
        reference = " ".join(text for i, (_, text) in enumerate(pages, 1) if i in manual)
        body_pages = [(label, text) for i, (label, text) in enumerate(pages, 1) if i not in manual]
        method = f"manual (pages {row['summary_pages']})"
    else:
        # Join pages, remembering where each starts, so the summary can be cut
        # out of whichever page(s) it sits on.
        offsets, full = [], ""
        for label, text in pages:
            offsets.append((len(full), label))
            full += text + "\n"
        span = find_summary(full)
        if not span:
            raise ValueError("no 'Abstract' or 'Executive Summary' heading found; set summary_pages")
        start, end = span
        reference = full[start:end]
        heading_start = full.rfind("\n", 0, start) + 1  # remove the heading line too
        body_pages = []
        for k, (page_start, label) in enumerate(offsets):
            page_end = offsets[k + 1][0] if k + 1 < len(offsets) else len(full)
            text = full[page_start:page_end]
            cut_a, cut_b = max(heading_start, page_start), min(end, page_end)
            if cut_a < cut_b:
                text = text[: cut_a - page_start] + text[cut_b - page_start :]
            if text.strip():
                body_pages.append((label, text.strip()))
        method = "automatic"

    body_pages = strip_references(body_pages)
    reference = " ".join(reference.split())
    body_words = sum(len(text.split()) for _, text in body_pages)
    return {
        "doc_id": row["file"],
        "title": row.get("title", ""),
        "source": row.get("source", ""),
        "genre": row.get("genre", ""),
        "year": row.get("year", ""),
        "reference": reference,
        "reference_method": method,
        "pages": [{"label": label, "text": text} for label, text in body_pages],
        "body_words": body_words,
    }


def main():
    if not MANIFEST.exists():
        raise SystemExit(
            f"Create {MANIFEST} first (copy eval/manifest_template.csv) and put the documents in {DOCS_DIR}."
        )
    with MANIFEST.open(encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("file", "").strip()]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    records, review_rows, problems = [], [], []
    for row in rows:
        try:
            record = build_record(row)
        except Exception as e:  # report every problem at once, don't stop
            problems.append([row["file"], str(e)])
            continue
        if record["body_words"] < MIN_BODY_WORDS:
            problems.append([row["file"], f"only {record['body_words']} words of body text (need {MIN_BODY_WORDS}+)"])
            continue
        records.append(record)
        words = record["reference"].split()
        review_rows.append([
            row["file"],
            record["reference_method"],
            len(words),
            record["body_words"],
            len(record["pages"]),
            " ".join(words[:15]) + " …",
            "… " + " ".join(words[-15:]),
        ])

    import json

    with CORPUS.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    review = [
        f"# Corpus check: {len(records)} usable of {len(rows)} documents\n",
        "Read the first and last words of each reference summary. If one starts or ends in the "
        "wrong place, put its page numbers in the manifest's `summary_pages` column and run "
        "`python eval/prepare_corpus.py` again.\n",
        markdown_table(
            ["File", "Found by", "Summary words", "Body words", "Pages", "Summary starts", "Summary ends"],
            review_rows,
        ),
    ]
    if problems:
        review += ["\n\n## Problems\n", markdown_table(["File", "Problem"], problems)]
    (RESULTS_DIR / "corpus_review.md").write_text("\n".join(review) + "\n", encoding="utf-8")
    print("\n".join(review))
    print(f"\nSaved {len(records)} documents to {CORPUS}")


if __name__ == "__main__":
    main()

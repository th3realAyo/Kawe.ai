"""Step 2: generate summaries for the Nigerian test set.

Reads eval/results/corpus.jsonl (made by prepare_corpus.py). For each
document it produces three summaries of matched length:
  * kawe     - the app's grounded pipeline (embedding ranking + Llama rewrite)
  * textrank - the original university system (bag-of-words TextRank)
  * lead     - the first N words of the document (a standard simple baseline)

Kàwé works on the document's real pages, exactly as in the app. Kàwé decides
its own length from the summary percentage; the two baselines are then
matched to that length so the comparison is fair.

Outputs eval/results/summaries.jsonl. Scoring happens in score_summaries.py,
which needs no Modal calls and can be re-run freely.

Usage:
    python eval/run_summary_eval.py --percentage 30
"""

from __future__ import annotations

import argparse
import time

from common import RESULTS_DIR, append_jsonl, load_env, read_jsonl, require_env

CORPUS = RESULTS_DIR / "corpus.jsonl"


def to_segments(record):
    """The document's real pages (summary and references already removed)."""
    from doc_loader import ExtractedSegment

    return [
        ExtractedSegment(text=page["text"], location_label=page["label"], order=i)
        for i, page in enumerate(record["pages"])
    ]


def lead_summary(report: str, n_words: int) -> str:
    return " ".join(report.split()[:n_words])


def _sparse_pagerank(bags, alpha: float = 0.85, max_iter: int = 100, tol: float = 1.0e-6):
    """Weighted PageRank over the sentence cosine-similarity graph, computed
    without building the n x n matrix.

    Same maths as nx.pagerank(nx.from_numpy_array(S)) with S[i, j] the cosine
    of the bag-of-words vectors and S[i, i] = 0: S times a vector is computed
    as X (X^T v) minus the diagonal, where X holds the L2-normalized word
    counts. Dangling sentences (no shared words) spread their score evenly,
    and convergence uses networkx's test (L1 change < n * tol). Returns None
    if it fails to converge, like networkx raising
    PowerIterationFailedConvergence.
    """
    import numpy as np
    from scipy.sparse import csr_matrix, diags

    n = len(bags)
    vocab, rows, cols, vals = {}, [], [], []
    for i, bag in enumerate(bags):
        for word, count in bag.items():
            rows.append(i)
            cols.append(vocab.setdefault(word, len(vocab)))
            vals.append(float(count))
    X = csr_matrix((vals, (rows, cols)), shape=(n, max(len(vocab), 1)))
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    Xn = diags(np.divide(1.0, norms, out=np.zeros(n), where=norms > 0)) @ X
    self_sim = np.asarray(Xn.multiply(Xn).sum(axis=1)).ravel()  # 1, or 0 for empty

    def similarity_times(v):
        return Xn @ (Xn.T @ v) - self_sim * v

    degree = similarity_times(np.ones(n))
    dangling = degree <= 1e-12
    inv_degree = np.divide(1.0, degree, out=np.zeros(n), where=~dangling)
    x = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        last = x
        x = alpha * similarity_times(last * inv_degree) + (alpha * last[dangling].sum() + 1 - alpha) / n
        if np.abs(x - last).sum() < n * tol:
            return x
    return None


def textrank_summary(report: str, n_words: int) -> str:
    """The thesis system's TextRank: bag-of-words cosine similarity between
    sentences (stopwords removed), PageRank, then top sentences in document
    order. Same maths as textrank_summarizer.py, but computed with sparse
    matrices so 300-page reports take seconds instead of hours."""
    import re
    from collections import Counter

    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize

    stop = set(stopwords.words("english"))
    sentences = sent_tokenize(report)
    if not sentences:
        return ""
    bags = [
        Counter(w for w in word_tokenize(re.sub(r"[^\w\s]", "", s.lower())) if w not in stop)
        for s in sentences
    ]
    scores = _sparse_pagerank(bags)
    if scores is None:
        scores = [1.0] * len(sentences)

    chosen, total = [], 0
    for i in sorted(range(len(sentences)), key=lambda k: scores[k], reverse=True):
        if total >= n_words:
            break
        chosen.append(i)
        total += len(sentences[i].split())
    return " ".join(sentences[i] for i in sorted(chosen))


def warm_up() -> float:
    """One tiny request to each GPU service (embedder and summarizer run in
    separate containers) so the first document's timing doesn't include a
    cold start. Returns the combined cold-start time in seconds."""
    from chunk_ranker import _embed_via_modal
    from grounded_summarizer import _summarize_via_modal

    start = time.perf_counter()
    _embed_via_modal(["Kàwé turns long documents into short audio summaries."])
    _summarize_via_modal("Kàwé turns long documents into short audio summaries.", "English")
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--percentage", type=int, default=30, help="summary percentage, as on the app's slider")
    args = parser.parse_args()

    load_env()
    require_env("MODAL_EMBED_URL", "MODAL_SUMMARIZE_URL", "RAG_AUTH_TOKEN")

    import nltk

    for pkg in ("punkt", "punkt_tab", "stopwords"):
        nltk.download(pkg, quiet=True)

    from grounded_summarizer import generate_grounded_summary
    from product_metrics import summary_top_n

    out_path = RESULTS_DIR / "summaries.jsonl"
    done = {r["doc_index"] for r in read_jsonl(out_path)}

    print("Warming up the summarization model (first call can take a few minutes)...")
    cold_start = warm_up()
    print(f"  cold start: {cold_start:.1f}s")
    append_jsonl(RESULTS_DIR / "cold_start.jsonl", {"cold_start_seconds": cold_start})

    corpus = read_jsonl(CORPUS)
    if not corpus:
        raise SystemExit("No test set found. Run eval/prepare_corpus.py first.")

    for n, doc in enumerate(corpus, start=1):
        idx = doc["doc_id"]
        if idx in done:
            print(f"[{n}/{len(corpus)}] {idx}: already done, skipping")
            continue

        segments = to_segments(doc)
        report = " ".join(seg.text for seg in segments)
        reference = doc["reference"]
        top_n = summary_top_n(args.percentage, len(segments))

        start = time.perf_counter()
        result = generate_grounded_summary(segments, language_code="en", top_n=top_n)
        kawe_seconds = time.perf_counter() - start

        n_words = len(result.full_text.split())
        record = {
            "doc_index": idx,
            "title": doc.get("title", ""),
            "source": doc.get("source", ""),
            "genre": doc.get("genre", ""),
            "source_words": len(report.split()),
            "pages": len(segments),
            "top_n": top_n,
            "percentage": args.percentage,
            "reference": reference,
            "kawe": result.full_text,
            "kawe_sections": [
                {"summary": s.summary_text, "source": s.source_excerpt, "location": s.location_label}
                for s in result.sections
            ],
            "kawe_skipped_chunks": result.skipped_chunks,
            "kawe_seconds": round(kawe_seconds, 2),
            "top_n_capped": top_n == 15,
            "textrank": textrank_summary(report, n_words),
            "lead": lead_summary(report, n_words),
        }
        append_jsonl(out_path, record)
        print(
            f"[{n}/{len(corpus)}] {idx}: {record['source_words']} words -> "
            f"{n_words} words in {kawe_seconds:.1f}s ({len(result.sections)} sections)"
        )

    print("Done. Now run: python eval/score_summaries.py")


if __name__ == "__main__":
    main()

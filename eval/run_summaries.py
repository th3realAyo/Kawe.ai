"""Generates every summary used in the evaluation (needs the Modal endpoints).

Conditions, for each paper in results/corpus.jsonl:
  kawe          app pipeline (centrality ranking + passage-constrained rewrite), at every budget
  textrank      thesis TextRank, matched to Kàwé's length at every budget
  lead          first words of the body, matched to Kàwé's length at every budget
  kawe_random   ablation at the main budget: same rewriting, passages chosen at random
  kawe_first    ablation at the main budget: same rewriting, the first passages of the body
  llm_document  ablation at the main budget: whole body summarized in one prompt, no passage constraint

Passage rewrites use greedy decoding, so they are cached in results/passage_cache.jsonl and
reused across budgets and ablations. Timings are taken from the first, uncached run at the
main budget. Re-running skips finished work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time

from baselines import lead_summary, textrank_summary
from common import RESULTS_DIR, append_jsonl, ensure_nltk, load_env, read_jsonl, require_env

from kawe import modal_client
from kawe.extraction import ExtractedSegment
from kawe.metrics import summary_top_n
from kawe.ranking import split_into_chunks
from kawe.summarizer import generate_grounded_summary, rewrite_chunks

OUT = RESULTS_DIR / "summaries.jsonl"
CACHE = RESULTS_DIR / "passage_cache.jsonl"
EMBED_DIR = RESULTS_DIR / "embeddings"
LANGUAGE = "English"
DOCUMENT_INSTRUCTION = (
    "You are a precise summarization assistant. Summarize the following document in about "
    "{n} words, in English, as continuous prose. Cover its purpose, methods and main findings. "
    "Do not add information that is not in the document."
)


class PassageCache:
    """Caches raw model output per passage; counts calls that reached the model."""

    def __init__(self, path):
        self.path = path
        self.items = {r["key"]: r for r in read_jsonl(path)}
        self.misses = 0

    def summarize(self, text: str, language: str) -> str:
        key = hashlib.sha1(f"{language}\n{text}".encode("utf-8")).hexdigest()
        if key not in self.items:
            start = time.perf_counter()
            raw = modal_client.summarize(text, language)
            record = {"key": key, "raw": raw, "seconds": round(time.perf_counter() - start, 3)}
            append_jsonl(self.path, record)
            self.items[key] = record
            self.misses += 1
        return self.items[key]["raw"]


class Embeddings:
    """One embedding call per paper, stored so later budgets and re-runs reuse it."""

    def __init__(self, doc_id: str):
        self.path = EMBED_DIR / f"{doc_id}.json"
        self.called_api = False

    def __call__(self, texts):
        if self.path.exists():
            stored = json.loads(self.path.read_text(encoding="utf-8"))
            if stored["n"] == len(texts):
                return stored["vectors"]
        vectors = modal_client.embed(texts)
        self.called_api = True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"n": len(texts), "vectors": vectors}), encoding="utf-8")
        return vectors


def sections_payload(sections):
    return [
        {"summary": s.summary_text, "source": s.source_excerpt, "location": s.location_label, "chunk_index": s.chunk_index, "score": s.score}
        for s in sections
    ]


def warm_up() -> dict:
    """Times the first calls; the shared GPU container is cold after five idle minutes.

    Embedding and summarization share one container, so the first call pays the cold start.
    """
    start = time.perf_counter()
    modal_client.embed(["Warm-up request."])
    embed_seconds = time.perf_counter() - start
    start = time.perf_counter()
    modal_client.summarize("Warm-up request for the summarization model.", LANGUAGE)
    return {"embed_seconds": round(embed_seconds, 2), "summarize_seconds": round(time.perf_counter() - start, 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budgets", type=int, nargs="+", default=[10, 20, 30, 40, 50], help="summary percentages")
    parser.add_argument("--main-budget", type=int, default=30, help="budget used for ablations and timing")
    parser.add_argument("--skip-document-baseline", action="store_true", help="skip llm_document")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    budgets = [args.main_budget] + [b for b in args.budgets if b != args.main_budget]

    load_env()
    require_env("MODAL_EMBED_URL", "MODAL_SUMMARIZE_URL", "RAG_AUTH_TOKEN")
    if not args.skip_document_baseline:
        require_env("EVAL_AUTH_TOKEN")  # the whole-document baseline replaces the system prompt
    ensure_nltk()

    corpus = read_jsonl(RESULTS_DIR / "corpus.jsonl")
    if not corpus:
        raise SystemExit("No test set. Run eval/prepare_corpus.py first.")
    done = {(r["doc_id"], r["condition"], r["budget"]) for r in read_jsonl(OUT)}
    cache = PassageCache(CACHE)

    conditions = ["kawe", "textrank", "lead"]
    expected = {(d["doc_id"], c, b) for d in corpus for c in conditions for b in budgets}
    expected |= {(d["doc_id"], c, args.main_budget) for d in corpus for c in ("kawe_random", "kawe_first")}
    if not args.skip_document_baseline:
        expected |= {(d["doc_id"], "llm_document", args.main_budget) for d in corpus}
    if expected <= done:
        print("All summaries already exist.")
        return

    print("Warming up the GPU services...")
    cold = warm_up()
    append_jsonl(RESULTS_DIR / "cold_start.jsonl", cold)
    print(f"  first calls: embed {cold['embed_seconds']}s, summarize {cold['summarize_seconds']}s")

    for n, doc in enumerate(corpus, start=1):
        doc_id = doc["doc_id"]
        segments = [ExtractedSegment(p["text"], p["label"], i) for i, p in enumerate(doc["pages"])]
        body = " ".join(seg.text for seg in segments)
        chunks = split_into_chunks(segments)
        embeddings = Embeddings(doc_id)
        main_words = None

        for budget in budgets:
            top_n = summary_top_n(budget, len(chunks))
            if (doc_id, "kawe", budget) in done:
                record = next(r for r in read_jsonl(OUT) if (r["doc_id"], r["condition"], r["budget"]) == (doc_id, "kawe", budget))
                words = record["words"]
            else:
                misses = cache.misses
                start = time.perf_counter()
                result = generate_grounded_summary(segments, "en", top_n, embed_fn=embeddings, summarize_fn=cache.summarize)
                elapsed = time.perf_counter() - start
                fresh = cache.misses - misses == len(result.sections) + result.skipped_chunks
                timed = budget == args.main_budget and fresh and embeddings.called_api
                words = len(result.full_text.split())
                append_jsonl(OUT, {
                    "doc_id": doc_id, "condition": "kawe", "budget": budget, "top_n": top_n,
                    "summary": result.full_text, "words": words, "sections": sections_payload(result.sections),
                    "total_chunks": result.total_chunks, "skipped": result.skipped_chunks,
                    "timings": {**result.timings, "total": elapsed} if timed else None,
                })
                print(f"[{n}/{len(corpus)}] {doc_id} kawe@{budget}%: {top_n} passages, {words} words, {elapsed:.1f}s")
            for name, fn in (("textrank", textrank_summary), ("lead", lead_summary)):
                if (doc_id, name, budget) not in done:
                    summary = fn(body, words)
                    append_jsonl(OUT, {"doc_id": doc_id, "condition": name, "budget": budget, "summary": summary, "words": len(summary.split())})
            if budget == args.main_budget:
                main_words = words

        top_n = summary_top_n(args.main_budget, len(chunks))
        rng = random.Random(f"{args.seed}-{doc_id}")
        ablations = {
            "kawe_random": sorted(rng.sample(chunks, min(top_n, len(chunks))), key=lambda c: c.chunk_index),
            "kawe_first": chunks[:top_n],
        }
        for name, selected in ablations.items():
            if (doc_id, name, args.main_budget) in done:
                continue
            sections, skipped, _ = rewrite_chunks(selected, LANGUAGE, cache.summarize)
            summary = " ".join(s.summary_text for s in sections)
            append_jsonl(OUT, {
                "doc_id": doc_id, "condition": name, "budget": args.main_budget, "top_n": top_n,
                "summary": summary, "words": len(summary.split()), "sections": sections_payload(sections),
                "total_chunks": len(chunks), "skipped": skipped,
            })
            print(f"[{n}/{len(corpus)}] {doc_id} {name}: {len(sections)} passages")

        if not args.skip_document_baseline and (doc_id, "llm_document", args.main_budget) not in done:
            start = time.perf_counter()
            summary = modal_client.summarize(
                body, LANGUAGE, max_new_tokens=int(main_words * 1.7) + 64, instruction=DOCUMENT_INSTRUCTION.format(n=main_words)
            )
            elapsed = time.perf_counter() - start
            append_jsonl(OUT, {
                "doc_id": doc_id, "condition": "llm_document", "budget": args.main_budget, "summary": summary.strip(),
                "words": len(summary.split()), "target_words": main_words, "timings": {"total": elapsed},
            })
            print(f"[{n}/{len(corpus)}] {doc_id} llm_document: {len(summary.split())} words in {elapsed:.1f}s")

    print("Done. Next: python eval/score_summaries.py")


if __name__ == "__main__":
    main()

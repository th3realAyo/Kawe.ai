"""Scores results/summaries.jsonl.

Automatic measures (seconds, no GPU):
  * ROUGE-1/2/Lsum F1 against each paper's own abstract
  * abstractiveness: share of summary n-grams that do not occur in the body
  * provenance: where in the paper the summarized passages lie, compared with the passages
    that best match each abstract sentence ("oracle" positions)

Faithfulness (NLI on CPU, a few minutes; skip with --skip-nli). Each summary sentence is checked
against the passage it cites, against another section's passage (control), and against the
best of the five body passages most similar to it, which also applies to the whole-document baseline.

Writes scores.jsonl, positions.jsonl, grounding.jsonl, summary_report.md and summary_scores.json.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict

import numpy as np
from common import RESULTS_DIR, NLIJudge, ensure_nltk, markdown_table, paper_label, read_jsonl, write_jsonl
from stats import clustered_rate_ci, fmt_p, mean_ci, paired_test

from kawe.extraction import ExtractedSegment
from kawe.ranking import split_into_chunks
from kawe.summarizer import _PREAMBLE

SYSTEM_LABELS = {
    "kawe": "Kàwé",
    "textrank": "TextRank",
    "lead": "Lead",
    "kawe_random": "Kàwé, random passages",
    "kawe_first": "Kàwé, first passages",
    "llm_document": "LLM, whole document",
}
MAIN_SYSTEMS = ["kawe", "textrank", "lead"]
ABLATIONS = ["kawe", "kawe_random", "kawe_first", "llm_document"]
PASSAGE_SYSTEMS = ["kawe", "kawe_random", "kawe_first"]
THRESHOLD = 0.5
TOP_K = 5
TOKEN = re.compile(r"\w+", re.UNICODE)


def tokens(text: str):
    return TOKEN.findall(text.lower())


def ngrams(words, n):
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def novel_ngram_rate(summary: str, body_ngrams: dict, n: int) -> float:
    grams = ngrams(tokens(summary), n)
    return sum(g not in body_ngrams[n] for g in grams) / len(grams) if grams else math.nan


def sentences(text: str):
    from nltk.tokenize import sent_tokenize

    return [s for s in sent_tokenize(" ".join(text.split())) if len(s.split()) >= 3]


class TfIdf:
    """TF-IDF cosine retrieval over a paper's passages."""

    def __init__(self, passages):
        self.docs = [Counter(tokens(p)) for p in passages]
        df = Counter(t for d in self.docs for t in d)
        self.idf = {t: math.log((1 + len(self.docs)) / (1 + c)) + 1 for t, c in df.items()}
        self.vecs = [self._vec(d) for d in self.docs]

    def _vec(self, counts):
        v = {t: c * self.idf.get(t, 0.0) for t, c in counts.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def top(self, text: str, k: int):
        q = self._vec(Counter(tokens(text)))
        sims = [sum(q.get(t, 0.0) * w for t, w in v.items()) for v in self.vecs]
        return sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]


def paper_chunks(doc):
    segments = [ExtractedSegment(p["text"], p["label"], i) for i, p in enumerate(doc["pages"])]
    return split_into_chunks(segments)


def rouge_rows(records, corpus):
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
    rows = []
    body_grams = {}
    for doc_id, doc in corpus.items():
        words = tokens(" ".join(p["text"] for p in doc["pages"]))
        body_grams[doc_id] = {n: set(ngrams(words, n)) for n in (1, 2, 3)}
    for r in records:
        doc = corpus[r["doc_id"]]
        reference = "\n".join(sentences(doc["reference"]))
        candidate = "\n".join(sentences(r["summary"])) or r["summary"]
        s = scorer.score(reference, candidate)
        rows.append({
            "doc_id": r["doc_id"], "condition": r["condition"], "budget": r["budget"], "words": r["words"],
            "rouge1": s["rouge1"].fmeasure * 100, "rouge2": s["rouge2"].fmeasure * 100, "rougeLsum": s["rougeLsum"].fmeasure * 100,
            **{f"novel{n}": novel_ngram_rate(r["summary"], body_grams[r["doc_id"]], n) * 100 for n in (1, 2, 3)},
        })
    return rows


def position_rows(records, corpus, main_budget):
    from baselines import textrank_sentences
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    by_key = {(r["doc_id"], r["condition"], r["budget"]): r for r in records}
    rows = []
    for doc_id, doc in corpus.items():
        chunks = paper_chunks(doc)
        last = max(len(chunks) - 1, 1)
        oracle = []
        for sentence in sentences(doc["reference"]):
            recall = [scorer.score(c.text, sentence)["rouge1"].precision for c in chunks]
            oracle.append(int(np.argmax(recall)) / last)
        row = {"doc_id": doc_id, "oracle": oracle, "total_chunks": len(chunks)}
        for condition in PASSAGE_SYSTEMS:
            r = by_key.get((doc_id, condition, main_budget))
            if r:
                row[condition] = [s["chunk_index"] / last for s in r["sections"]]
        body = " ".join(p["text"] for p in doc["pages"])
        tr = by_key.get((doc_id, "textrank", main_budget))
        if tr:
            all_sentences = textrank_sentences(body)
            picked = [i for i, s in enumerate(all_sentences) if s in tr["summary"]]
            row["textrank"] = [i / max(len(all_sentences) - 1, 1) for i in picked]
        lead = by_key.get((doc_id, "lead", main_budget))
        if lead:
            row["lead_extent"] = lead["words"] / max(len(body.split()), 1)
        rows.append(row)
    return rows


def grounding_rows(records, corpus, main_budget, judge, seed):
    rng = random.Random(seed)
    jobs, rows = [], []
    for r in records:
        if r["condition"] not in ABLATIONS or r["budget"] != main_budget:
            continue
        chunks = paper_chunks(corpus[r["doc_id"]])
        index = TfIdf([c.text for c in chunks])
        units = [(i, s["summary"], s["source"]) for i, s in enumerate(r.get("sections") or [])]
        if not units:
            units = [(None, r["summary"], None)]
        sources = [u[2] for u in units if u[2]]
        for section, text, source in units:
            others = [src for j, src in enumerate(sources) if j != section]
            control = rng.choice(others) if others else None
            for sentence in sentences(text):
                row = {"doc_id": r["doc_id"], "condition": r["condition"], "section": section, "sentence": sentence}
                rows.append(row)
                k = len(rows) - 1
                if source:
                    jobs.append((k, "cited", source, sentence))
                if control:
                    jobs.append((k, "control", control, sentence))
                for c in index.top(sentence, TOP_K):
                    jobs.append((k, f"cand:{c}", chunks[c].text, sentence))

    print(f"NLI: {len(rows)} sentences, {len(jobs)} premise-hypothesis pairs", flush=True)
    results = judge.score([(p, h) for _, _, p, h in jobs])
    for (k, field, _, _), (entail, contra) in zip(jobs, results):
        row = rows[k]
        if field == "cited":
            row["cited_entail"], row["cited_contra"] = entail, contra
        elif field == "control":
            row["control_entail"] = entail
        elif entail > row.get("best_entail", -1.0):
            row["best_entail"], row["best_contra"], row["best_chunk"] = entail, contra, int(field.split(":")[1])
    return rows


def fmt_ci(mean, lo, hi, digits=1, pct=False):
    scale = 100 if pct else 1
    return f"{mean * scale:.{digits}f} [{lo * scale:.{digits}f}, {hi * scale:.{digits}f}]"


def build_report(scores, positions, grounding, records, corpus, main_budget):
    out, summary = [], {"n_papers": len(corpus), "main_budget": main_budget}

    def values(condition, metric, budget=main_budget):
        rows = sorted((s for s in scores if s["condition"] == condition and s["budget"] == budget), key=lambda s: s["doc_id"])
        return [s[metric] for s in rows]

    out.append(f"# Evaluation on {len(corpus)} papers (main budget {main_budget}%)\n")
    out.append("## ROUGE F1 against the authors' abstracts (mean, 95% bootstrap CI over papers)\n")
    table, summary["rouge"] = [], {}
    for c in ["kawe", "textrank", "lead", "kawe_random", "kawe_first", "llm_document"]:
        if not values(c, "rouge1"):
            continue
        cis = {m: mean_ci(values(c, m)) for m in ("rouge1", "rouge2", "rougeLsum")}
        summary["rouge"][c] = {m: list(v) for m, v in cis.items()} | {"words": statistics.mean(values(c, "words"))}
        table.append([SYSTEM_LABELS[c]] + [fmt_ci(*cis[m]) for m in ("rouge1", "rouge2", "rougeLsum")] + [f"{statistics.mean(values(c, 'words')):.0f}"])
    out.append(markdown_table(["System", "ROUGE-1", "ROUGE-2", "ROUGE-Lsum", "Words"], table))

    out.append("\n\n## Paired comparisons with Kàwé (difference, 95% CI, wins out of papers, Wilcoxon)\n")
    table, summary["paired"] = [], {}
    for c in ["textrank", "lead", "kawe_random", "kawe_first", "llm_document"]:
        if not values(c, "rouge1"):
            continue
        for m in ("rouge1", "rouge2", "rougeLsum"):
            t = paired_test(values("kawe", m), values(c, m))
            summary["paired"][f"{c}/{m}"] = t
            table.append([SYSTEM_LABELS[c], m, f"{t['mean_diff']:+.1f} [{t['ci_low']:+.1f}, {t['ci_high']:+.1f}]", f"{t['wins']}/{t['n']}", fmt_p(t["p"])])
    out.append(markdown_table(["Versus", "Metric", "Kàwé minus system", "Kàwé better", "p"], table))

    budgets = sorted({s["budget"] for s in scores if s["condition"] == "kawe"})
    out.append("\n\n## ROUGE-Lsum by summary budget\n")
    table, summary["budgets"] = [], {}
    for c in MAIN_SYSTEMS:
        row = [SYSTEM_LABELS[c]]
        for b in budgets:
            v = values(c, "rougeLsum", b)
            row.append(f"{statistics.mean(v):.1f}" if v else "–")
            summary["budgets"][f"{c}/{b}"] = {m: statistics.mean(values(c, m, b)) for m in ("rouge1", "rouge2", "rougeLsum", "words")} if v else None
        table.append(row)
    out.append(markdown_table(["System"] + [f"{b}%" for b in budgets], table))

    out.append("\n\n## Abstractiveness: novel n-grams relative to the body (%)\n")
    table, summary["novelty"] = [], {}
    for c in ["kawe", "kawe_random", "kawe_first", "llm_document", "textrank", "lead"]:
        if values(c, "novel2"):
            summary["novelty"][c] = {f"novel{n}": statistics.mean(values(c, f"novel{n}")) for n in (1, 2, 3)}
            table.append([SYSTEM_LABELS[c]] + [f"{summary['novelty'][c][f'novel{n}']:.1f}" for n in (1, 2, 3)])
    out.append(markdown_table(["System", "Unigrams", "Bigrams", "Trigrams"], table))

    out.append("\n\n## Provenance: distance from each abstract sentence's best-matching passage to the nearest summarized passage\n")
    table, summary["provenance"] = [], {}
    for c in PASSAGE_SYSTEMS + ["textrank"]:
        dists = [min(abs(o - p) for p in row[c]) for row in positions if row.get(c) for o in row["oracle"]]
        early = [p < 0.2 for row in positions for p in row.get(c, [])]
        if dists:
            summary["provenance"][c] = {"mean_distance": statistics.mean(dists), "share_first_fifth": sum(early) / len(early)}
            table.append([SYSTEM_LABELS[c], f"{statistics.mean(dists):.3f}", f"{100 * sum(early) / len(early):.0f}%"])
    oracle_early = [o < 0.2 for row in positions for o in row["oracle"]]
    table.append(["Abstract (oracle)", "0", f"{100 * sum(oracle_early) / max(len(oracle_early), 1):.0f}%"])
    out.append(markdown_table(["Selection", "Mean distance (fraction of paper)", "Selected from first 20%"], table))

    if grounding:
        out.append(f"\n\n## Faithfulness (NLI, threshold {THRESHOLD}; pooled rate, 95% CI resampling papers)\n")
        table, summary["grounding"] = [], {}
        for c in ABLATIONS:
            rows = [g for g in grounding if g["condition"] == c]
            if not rows:
                continue
            entry = {"sentences": len(rows)}
            cells = [SYSTEM_LABELS[c], len(rows)]
            for field, name in (("cited_entail", "cited"), ("cited_contra", "contradicted"), ("control_entail", "control"), ("best_entail", "best")):
                groups = defaultdict(list)
                for g in rows:
                    if field in g:
                        groups[g["doc_id"]].append(g[field] >= THRESHOLD)
                if groups:
                    entry[name] = list(clustered_rate_ci(groups))
                    cells.append(fmt_ci(*entry[name], pct=True))
                else:
                    cells.append("–")
            summary["grounding"][c] = entry
            table.append(cells)
        out.append(markdown_table(["System", "Sentences", "Supported by cited passage", "Contradicted by cited passage", "Supported by another section (control)", "Supported by best of 5 retrieved passages"], table))

        out.append("\n\n## Supported sentences per paper (Kàwé, cited passage)\n")
        per_doc = defaultdict(list)
        for g in grounding:
            if g["condition"] == "kawe" and "cited_entail" in g:
                per_doc[g["doc_id"]].append(g["cited_entail"] >= THRESHOLD)
        summary["grounding_per_paper"] = {d: sum(v) / len(v) for d, v in per_doc.items()}
        out.append(markdown_table(["Paper", "Sentences", "Supported"], [[paper_label(d), len(v), f"{100 * sum(v) / len(v):.0f}%"] for d, v in sorted(per_doc.items())]))

    timed = [r for r in records if r["condition"] == "kawe" and r.get("timings")]
    if timed:
        out.append("\n\n## Kàwé processing time at the main budget (warm GPU, seconds)\n")
        stages = ["embed", "rank", "rewrite", "total"]
        table = [[paper_label(r["doc_id"]), r["top_n"]] + [f"{r['timings'].get(s, 0):.1f}" for s in stages] for r in sorted(timed, key=lambda r: r["doc_id"])]
        table.append(["Mean", f"{statistics.mean(r['top_n'] for r in timed):.1f}"] + [f"{statistics.mean(r['timings'].get(s, 0) for r in timed):.1f}" for s in stages])
        out.append(markdown_table(["Paper", "Passages", "Embed", "Rank", "Rewrite", "Total"], table))
        summary["timing"] = {s: statistics.mean(r["timings"].get(s, 0) for r in timed) for s in stages}
        doc_runs = [r for r in records if r["condition"] == "llm_document" and r.get("timings")]
        if doc_runs:
            summary["timing"]["llm_document"] = statistics.mean(r["timings"]["total"] for r in doc_runs)
        cold = read_jsonl(RESULTS_DIR / "cold_start.jsonl")
        if cold:
            summary["cold_start"] = cold
            out.append("\nFirst-call (cold) times: " + "; ".join(f"embed {c['embed_seconds']}s, summarize {c['summarize_seconds']}s" for c in cold))

    cache = read_jsonl(RESULTS_DIR / "passage_cache.jsonl")
    if cache:
        hits = sum(bool(_PREAMBLE.match(c["raw"])) for c in cache)
        summary["preamble"] = {"outputs": len(cache), "with_preamble": hits}
        out.append(f"\n\nPassage rewrites that began with a chat preamble (removed by clean_summary): {hits} of {len(cache)}.")
    return out, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-nli", action="store_true", help="skip the faithfulness check (no model download)")
    parser.add_argument("--main-budget", type=int, default=30)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    ensure_nltk()

    records = read_jsonl(RESULTS_DIR / "summaries.jsonl")
    if not records:
        raise SystemExit("No summaries. Run eval/run_summaries.py first.")
    corpus = {d["doc_id"]: d for d in read_jsonl(RESULTS_DIR / "corpus.jsonl")}

    scores = rouge_rows(records, corpus)
    positions = position_rows(records, corpus, args.main_budget)
    write_jsonl(RESULTS_DIR / "scores.jsonl", scores)
    write_jsonl(RESULTS_DIR / "positions.jsonl", positions)

    grounding = read_jsonl(RESULTS_DIR / "grounding.jsonl")
    if not args.skip_nli:
        print("Loading the NLI model (about 0.7 GB on first use)...", flush=True)
        grounding = grounding_rows(records, corpus, args.main_budget, NLIJudge(), args.seed)
        write_jsonl(RESULTS_DIR / "grounding.jsonl", grounding)

    report, summary = build_report(scores, positions, grounding, records, corpus, args.main_budget)
    (RESULTS_DIR / "summary_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (RESULTS_DIR / "summary_scores.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()

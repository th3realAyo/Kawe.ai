"""Step 2 of the summary evaluation: score eval/results/summaries.jsonl.

Two kinds of score:

1. ROUGE-1/2/Lsum F1 against each document's own abstract or executive
   summary (written by its authors), for Kàwé
   and both baselines. ROUGE measures word overlap with the reference.

2. Grounding (Kàwé only). Every Kàwé summary section claims to come from one
   source passage. A natural-language-inference (NLI) model checks each
   summary sentence against that passage:
     - supported:     the passage entails the sentence
     - contradicted:  the passage contradicts the sentence
   As a control, the same sentence is also checked against a *different*
   section's passage from the same summary. If citations are meaningful, support should
   be much higher for the cited passage than for the random one.

No Modal calls; runs on CPU (the NLI step takes a few minutes).

Usage:
    python eval/score_summaries.py
"""

from __future__ import annotations

import argparse
import json
import random
import statistics

from common import RESULTS_DIR, markdown_table, read_jsonl

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
SYSTEMS = ["kawe", "textrank", "lead"]
SYSTEM_LABELS = {"kawe": "Kàwé (grounded)", "textrank": "TextRank (thesis system)", "lead": "Lead"}


def rouge_scores(records):
    from nltk.tokenize import sent_tokenize
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
    per_system = {s: {"rouge1": [], "rouge2": [], "rougeLsum": [], "words": []} for s in SYSTEMS}
    for r in records:
        reference = "\n".join(sent_tokenize(r["reference"]))
        for system in SYSTEMS:
            candidate = "\n".join(sent_tokenize(r[system]))
            scores = scorer.score(reference, candidate)
            for key in ("rouge1", "rouge2", "rougeLsum"):
                per_system[system][key].append(scores[key].fmeasure * 100)
            per_system[system]["words"].append(len(r[system].split()))
    return per_system


class NLIJudge:
    def __init__(self, model_name: str = NLI_MODEL):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.entail_idx = labels["entailment"]
        self.contra_idx = labels["contradiction"]

    def probs(self, premise: str, hypothesis: str):
        inputs = self.tokenizer(premise, hypothesis, truncation="only_first", max_length=512, return_tensors="pt")
        with self.torch.no_grad():
            p = self.torch.softmax(self.model(**inputs).logits[0], dim=-1)
        return float(p[self.entail_idx]), float(p[self.contra_idx])


def grounding_scores(records, judge, seed: int, threshold: float = 0.5):
    from nltk.tokenize import sent_tokenize

    rng = random.Random(seed)
    rows = []
    for r in records:
        sources = [s["source"] for s in r["kawe_sections"]]
        for i, section in enumerate(r["kawe_sections"]):
            others = [src for j, src in enumerate(sources) if j != i]
            control = rng.choice(others) if others else None
            for sentence in sent_tokenize(section["summary"]):
                if len(sentence.split()) < 3:
                    continue
                entail, contra = judge.probs(section["source"], sentence)
                row = {
                    "doc_index": r["doc_index"],
                    "sentence": sentence,
                    "cited_entail": entail,
                    "cited_contradict": contra,
                }
                if control:
                    row["control_entail"], _ = judge.probs(control, sentence)
                rows.append(row)

    supported = [row["cited_entail"] >= threshold for row in rows]
    contradicted = [row["cited_contradict"] >= threshold for row in rows]
    control = [row["control_entail"] >= threshold for row in rows if "control_entail" in row]
    return rows, {
        "sentences": len(rows),
        "supported_by_cited_pct": 100 * sum(supported) / max(len(rows), 1),
        "contradicted_by_cited_pct": 100 * sum(contradicted) / max(len(rows), 1),
        "supported_by_random_pct": 100 * sum(control) / max(len(control), 1),
    }


def mean_sd(values):
    if not values:
        return "–"
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.mean(values):.1f} ± {sd:.1f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-nli", action="store_true", help="only compute ROUGE")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    import nltk

    for pkg in ("punkt", "punkt_tab"):
        nltk.download(pkg, quiet=True)

    records = read_jsonl(RESULTS_DIR / "summaries.jsonl")
    if not records:
        raise SystemExit("No summaries found. Run eval/run_summary_eval.py first.")

    rouge = rouge_scores(records)
    report = [f"# Summary evaluation ({len(records)} Nigerian documents)\n"]
    report.append("## Table 1: ROUGE F1 against human reference summaries (mean ± SD)\n")
    report.append(
        markdown_table(
            ["System", "ROUGE-1", "ROUGE-2", "ROUGE-Lsum", "Words"],
            [
                [SYSTEM_LABELS[s]] + [mean_sd(rouge[s][k]) for k in ("rouge1", "rouge2", "rougeLsum", "words")]
                for s in SYSTEMS
            ],
        )
    )

    genres = sorted({r.get("genre") or "unspecified" for r in records})
    if len(genres) > 1:
        report.append("\n\n## Table 1b: ROUGE-Lsum by document type\n")
        rows = []
        for genre in genres:
            subset = [r for r in records if (r.get("genre") or "unspecified") == genre]
            g = rouge_scores(subset)
            rows.append([genre, len(subset)] + [mean_sd(g[s]["rougeLsum"]) for s in SYSTEMS])
        report.append(markdown_table(["Document type", "Documents"] + [SYSTEM_LABELS[s] for s in SYSTEMS], rows))

    seconds = [r["kawe_seconds"] for r in records]
    cold = read_jsonl(RESULTS_DIR / "cold_start.jsonl")
    report.append("\n\n## Table 2: Kàwé processing time\n")
    report.append(
        markdown_table(
            ["Measure", "Value"],
            [
                ["Source length (words)", mean_sd([r["source_words"] for r in records])],
                ["Passages rewritten per document", mean_sd([r["top_n"] for r in records])],
                ["Summary time, warm GPU (s)", mean_sd(seconds)],
                ["Median summary time (s)", f"{statistics.median(seconds):.1f}"],
                ["GPU cold start (s)", mean_sd([c["cold_start_seconds"] for c in cold])],
                ["Passages that failed and were skipped", sum(r["kawe_skipped_chunks"] for r in records)],
                ["Documents where the 15-passage cap applied", sum(r.get("top_n_capped", r["top_n"] == 15) for r in records)],
            ],
        )
    )

    summary = {"n_docs": len(records), "rouge": {s: {k: statistics.mean(v) for k, v in rouge[s].items()} for s in SYSTEMS}}

    if not args.skip_nli:
        print("Checking grounding with the NLI model (downloads ~0.7 GB the first time)...")
        rows, grounding = grounding_scores(records, NLIJudge(), args.seed)
        with (RESULTS_DIR / "grounding_sentences.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        report.append("\n\n## Table 3: Grounding of Kàwé summary sentences (NLI check)\n")
        report.append(
            markdown_table(
                ["Measure", "Value"],
                [
                    ["Summary sentences checked", grounding["sentences"]],
                    ["Supported by the cited passage", f"{grounding['supported_by_cited_pct']:.1f}%"],
                    ["Contradicted by the cited passage", f"{grounding['contradicted_by_cited_pct']:.1f}%"],
                    ["Supported by another section's passage (control)", f"{grounding['supported_by_random_pct']:.1f}%"],
                ],
            )
        )
        summary["grounding"] = grounding

    (RESULTS_DIR / "summary_scores.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (RESULTS_DIR / "summary_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"\nSaved to {RESULTS_DIR / 'summary_report.md'}")


if __name__ == "__main__":
    main()

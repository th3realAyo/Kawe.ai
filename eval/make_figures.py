"""Draws the evaluation figures (PDF for LaTeX, PNG for Word) into results/figures/.

Each figure is skipped when its input files are missing, so this can run after any stage.
"""

from __future__ import annotations

import textwrap
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import FIGURES_DIR, RESULTS_DIR, paper_label, read_jsonl
from stats import mean_ci

# Colour follows the system everywhere (palette validated for colour-vision deficiency).
COLORS = {
    "kawe": "#2a78d6",
    "textrank": "#eb6834",
    "lead": "#1baf7a",
    "kawe_random": "#eda100",
    "kawe_first": "#e87ba4",
    "llm_document": "#008300",
    "nllb": "#1baf7a",
    "llama": "#2a78d6",
}
LABELS = {
    "kawe": "Kàwé",
    "textrank": "TextRank",
    "lead": "Lead",
    "kawe_random": "Kàwé, random passages",
    "kawe_first": "Kàwé, first passages",
    "llm_document": "LLM, whole document",
    "llama": "Llama-3.1-8B (Kàwé)",
    "nllb": "NLLB-200",
}
INK, INK_2, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e2dc", "#bdbab2"
SUPPORTED, CONTRADICTED = "#2a78d6", "#e34948"
THRESHOLD = 0.5
MAIN = 30

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "legend.frameon": False,
    "figure.dpi": 150,
})


def save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def load():
    scores = read_jsonl(RESULTS_DIR / "scores.jsonl")
    summaries = read_jsonl(RESULTS_DIR / "summaries.jsonl")
    return {
        "scores": scores,
        "summaries": summaries,
        "grounding": read_jsonl(RESULTS_DIR / "grounding.jsonl"),
        "positions": read_jsonl(RESULTS_DIR / "positions.jsonl"),
        "extraction": read_jsonl(RESULTS_DIR / "extraction.jsonl"),
        "translations": read_jsonl(RESULTS_DIR / "translation_scored.jsonl"),
        "corpus": {c["doc_id"]: c for c in read_jsonl(RESULTS_DIR / "corpus.jsonl")},
        "papers": sorted({s["doc_id"] for s in scores}),
    }


def metric(scores, condition, name, budget=MAIN):
    rows = sorted((s for s in scores if s["condition"] == condition and s["budget"] == budget), key=lambda s: s["doc_id"])
    return [s[name] for s in rows]


def fig_rouge(d):
    systems = ["kawe", "textrank", "lead"]
    metrics = [("rouge1", "ROUGE-1"), ("rouge2", "ROUGE-2"), ("rougeLsum", "ROUGE-Lsum")]
    if not metric(d["scores"], "kawe", "rouge1"):
        return
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    width = 0.26
    rng = np.random.default_rng(0)
    for i, s in enumerate(systems):
        for j, (m, _) in enumerate(metrics):
            vals = metric(d["scores"], s, m)
            mean, lo, hi = mean_ci(vals)
            x = j + (i - 1) * (width + 0.02)
            ax.bar(x, mean, width, color=COLORS[s], label=LABELS[s] if j == 0 else None, zorder=2)
            ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], color=INK, lw=1, capsize=2.5, zorder=3)
            ax.scatter(x + rng.uniform(-0.07, 0.07, len(vals)), vals, s=9, facecolor="white", edgecolor=INK_2, lw=0.6, zorder=4)
    ax.set_xticks(range(len(metrics)), [name for _, name in metrics])
    ax.set_ylabel("F1 against the authors' abstract")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.text(1.0, -0.16, "Bars: mean · whiskers: 95% bootstrap CI · dots: individual papers", transform=ax.transAxes, ha="right", fontsize=7, color=INK_2)
    save(fig, "fig_rouge")


def fig_budget(d):
    systems = ["kawe", "textrank", "lead"]
    budgets = sorted({s["budget"] for s in d["scores"] if s["condition"] == "kawe"})
    if len(budgets) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.4), sharex=True)
    for ax, (m, name) in zip(axes, [("rouge1", "ROUGE-1"), ("rougeLsum", "ROUGE-Lsum")]):
        for s in systems:
            stats = [mean_ci(metric(d["scores"], s, m, b)) for b in budgets]
            means = [x[0] for x in stats]
            ax.fill_between(budgets, [x[1] for x in stats], [x[2] for x in stats], color=COLORS[s], alpha=0.12, lw=0)
            ax.plot(budgets, means, color=COLORS[s], lw=2, marker="o", ms=4, label=LABELS[s])
        ax.set_title(name, loc="left")
        ax.set_xlabel("Summary budget (% of document)")
        ax.set_xticks(budgets, [f"{b}%" for b in budgets])
    axes[0].set_ylabel("F1 (mean, 95% CI band)")
    fig.tight_layout()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.98))
    save(fig, "fig_budget")


def fig_per_paper(d):
    systems = ["kawe", "textrank", "lead"]
    if not metric(d["scores"], "kawe", "rougeLsum"):
        return
    papers = d["papers"]
    values = {s: metric(d["scores"], s, "rougeLsum") for s in systems}
    order = np.argsort(values["kawe"])
    fig, ax = plt.subplots(figsize=(6.3, 0.32 * len(papers) + 0.9))
    for row, k in enumerate(order):
        vals = [values[s][k] for s in systems]
        ax.plot([min(vals), max(vals)], [row, row], color=GRID, lw=2.5, zorder=1, solid_capstyle="round")
        for s in systems:
            ax.scatter(values[s][k], row, s=34, color=COLORS[s], edgecolor="white", lw=1, zorder=3, label=LABELS[s] if row == 0 else None)
    ax.set_yticks(range(len(papers)), [paper_label(papers[k]) for k in order])
    ax.set_xlabel("ROUGE-Lsum F1 at the 30% budget")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig_per_paper")


def fig_nli_distribution(d):
    rows = [g for g in d["grounding"] if g["condition"] == "kawe" and "cited_entail" in g and "control_entail" in g]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(3.1, 2.5))
    for key, label, color in (("cited_entail", "Cited passage", SUPPORTED), ("control_entail", "Another section (control)", NEUTRAL)):
        x = np.sort([g[key] for g in rows])
        ax.step(x, 1 - np.arange(len(x)) / len(x), where="post", color=color, lw=2, label=label)
    ax.axvline(THRESHOLD, color=INK_2, lw=0.8, ls="--")
    ax.text(THRESHOLD + 0.02, 0.95, "threshold", fontsize=7, color=INK_2, va="top")
    ax.set_xlabel("Entailment probability")
    ax.set_ylabel("Share of Kàwé sentences ≥ x")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=7)
    save(fig, "fig_nli_distribution")


def fig_faithfulness(d):
    conditions = [c for c in ["kawe", "kawe_random", "kawe_first", "llm_document"] if any(g["condition"] == c for g in d["grounding"])]
    if not conditions:
        return
    fig, ax = plt.subplots(figsize=(6.3, 0.42 * len(conditions) + 0.9))
    for row, c in enumerate(conditions):
        rows = [g for g in d["grounding"] if g["condition"] == c and "best_entail" in g]
        n = len(rows)
        sup = sum(g["best_entail"] >= THRESHOLD for g in rows) / n
        con = sum(g["best_entail"] < THRESHOLD and g["best_contra"] >= THRESHOLD for g in rows) / n
        neu = 1 - sup - con
        left = 0
        for share, color, label in ((sup, SUPPORTED, "Supported"), (neu, NEUTRAL, "Not supported"), (con, CONTRADICTED, "Contradicted")):
            ax.barh(row, share, left=left, color=color, height=0.62, edgecolor="white", lw=2, label=label if row == 0 else None)
            if share >= 0.06:
                ax.text(left + share / 2, row, f"{100 * share:.0f}%", ha="center", va="center", fontsize=7.5, color="white" if color != NEUTRAL else INK)
            left += share
        ax.text(1.01, row, f"n={n}", va="center", fontsize=7, color=INK_2)
    ax.set_yticks(range(len(conditions)), [LABELS[c] for c in conditions])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(visible=False)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_xlabel("Summary sentences, judged against the best of the 5 most similar passages in the paper")
    save(fig, "fig_faithfulness")


def fig_support_per_paper(d):
    per_doc = defaultdict(list)
    for g in d["grounding"]:
        if g["condition"] == "kawe" and "cited_entail" in g:
            per_doc[g["doc_id"]].append(g)
    if not per_doc:
        return
    papers = sorted(per_doc, key=lambda p: np.mean([g["cited_entail"] >= THRESHOLD for g in per_doc[p]]))
    fig, ax = plt.subplots(figsize=(6.3, 0.32 * len(papers) + 0.8))
    for row, p in enumerate(papers):
        rows = per_doc[p]
        cited = np.mean([g["cited_entail"] >= THRESHOLD for g in rows])
        control = np.mean([g["control_entail"] >= THRESHOLD for g in rows if "control_entail" in g])
        ax.barh(row, cited, color=SUPPORTED, height=0.6, label="Cited passage" if row == 0 else None)
        ax.scatter(control, row, marker="|", s=120, color=INK, lw=2, zorder=3, label="Control passage" if row == 0 else None)
        ax.text(cited + 0.01, row, f"{100 * cited:.0f}%  (n={len(rows)})", va="center", fontsize=7, color=INK_2)
    ax.set_yticks(range(len(papers)), [paper_label(p) for p in papers])
    ax.set_xlim(0, 1.18)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Kàwé sentences supported (NLI entailment ≥ 0.5)")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7)
    save(fig, "fig_support_per_paper")


def fig_provenance(d):
    rows = [r for r in d["positions"] if r.get("kawe")]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["doc_id"])
    fig, ax = plt.subplots(figsize=(6.3, 0.42 * len(rows) + 1.0))
    for i, r in enumerate(rows):
        if "lead_extent" in r:
            ax.barh(i, r["lead_extent"], height=0.7, color=COLORS["lead"], alpha=0.18, lw=0, label="Lead (covered span)" if i == 0 else None)
        ax.scatter(r["oracle"], [i + 0.18] * len(r["oracle"]), marker="v", s=22, facecolor="white", edgecolor=INK, lw=0.8, zorder=3, label="Abstract sentence's best passage" if i == 0 else None)
        ax.scatter(r["kawe"], [i - 0.12] * len(r["kawe"]), s=26, color=COLORS["kawe"], edgecolor="white", lw=0.8, zorder=4, label="Passage selected by Kàwé" if i == 0 else None)
        if r.get("kawe_first"):
            ax.scatter(r["kawe_first"], [i - 0.32] * len(r["kawe_first"]), s=10, color=COLORS["kawe_first"], zorder=3, label="First passages (ablation)" if i == 0 else None)
    ax.set_yticks(range(len(rows)), [paper_label(r["doc_id"]) for r in rows])
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Position in the paper body (0 = start, 1 = end of main text)")
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7.5)
    save(fig, "fig_provenance")


def fig_abstractive(d):
    novelty = {(s["doc_id"], s["condition"]): s["novel2"] for s in d["scores"] if s["budget"] == MAIN}
    systems = [c for c in ["kawe", "llm_document"] if any(g["condition"] == c for g in d["grounding"])]
    if not systems:
        return
    fig, ax = plt.subplots(figsize=(3.1, 2.6))
    for c in systems:
        xs, ys = [], []
        for p in d["papers"]:
            rows = [g for g in d["grounding"] if g["condition"] == c and g["doc_id"] == p and "best_entail" in g]
            if rows and (p, c) in novelty:
                xs.append(novelty[(p, c)])
                ys.append(np.mean([g["best_entail"] >= THRESHOLD for g in rows]))
        ax.scatter(xs, ys, s=40, color=COLORS[c], edgecolor="white", lw=1, label=LABELS[c], zorder=3)
        if xs:
            ax.scatter(np.mean(xs), np.mean(ys), s=110, marker="X", color=COLORS[c], edgecolor=INK, lw=0.8, zorder=4)
    ax.set_xlabel("Novel bigrams (% of summary bigrams)")
    ax.set_ylabel("Sentences supported")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=7)
    ax.text(1.0, 1.02, "X = mean over papers", transform=ax.transAxes, ha="right", fontsize=7, color=INK_2)
    save(fig, "fig_abstractive")


def fig_timing(d):
    timed = sorted((r for r in d["summaries"] if r["condition"] == "kawe" and r.get("timings")), key=lambda r: r["doc_id"])
    if not timed:
        return
    doc_times = {r["doc_id"]: r["timings"]["total"] for r in d["summaries"] if r["condition"] == "llm_document" and r.get("timings")}
    stages = [("embed", "Embed passages", "#9ec5f4"), ("rank", "Rank (PageRank)", "#5b9be6"), ("rewrite", "Rewrite top passages", "#1d5fae")]
    fig, ax = plt.subplots(figsize=(6.3, 0.36 * len(timed) + 0.9))
    for i, r in enumerate(timed):
        left = 0
        for key, label, color in stages:
            v = r["timings"].get(key, 0.0)
            ax.barh(i, v, left=left, color=color, height=0.6, edgecolor="white", lw=1.5, label=label if i == 0 else None)
            left += v
        ax.annotate(f"{left:.0f}s · {r['top_n']} passages", (left, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7, color=INK_2)
        if r["doc_id"] in doc_times:
            ax.scatter(doc_times[r["doc_id"]], i, marker="D", s=26, color=COLORS["llm_document"], zorder=4, label="LLM, whole document" if i == 0 else None)
    ax.set_yticks(range(len(timed)), [paper_label(r["doc_id"]) for r in timed])
    ax.invert_yaxis()
    ax.set_xlabel("Seconds (warm GPU)")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7)
    save(fig, "fig_timing")


def fig_extraction(d):
    rows = sorted(d["extraction"], key=lambda r: r["doc_id"])
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.3, 0.36 * len(rows) + 0.9))
    h = 0.36
    for i, r in enumerate(rows):
        for k, (key, label, color) in enumerate((("pdfplumber", "pdfplumber (previous)", COLORS["textrank"]), ("pdfium", "PDFium (current)", COLORS["kawe"]))):
            v = 100 * r[key]["fused_rate"]
            ax.barh(i + (k - 0.5) * h, v, height=h, color=color, edgecolor="white", lw=1, label=label if i == 0 else None)
            ax.text(v + 0.2, i + (k - 0.5) * h, f"{v:.1f}%", va="center", fontsize=6.5, color=INK_2)
    ax.set_yticks(range(len(rows)), [paper_label(r["doc_id"]) for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Tokens longer than 25 characters (words fused by lost spaces), % of tokens")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=7)
    save(fig, "fig_extraction")


def fig_translation(d):
    recs = d["translations"]
    if not recs:
        return
    langs = [l for l in ("yor", "ibo", "hau") if any(r["lang"] == l for r in recs)]
    systems = [s for s in ("llama", "nllb") if any(r["system"] == s for r in recs)]
    names = {"yor": "Yoruba", "ibo": "Igbo", "hau": "Hausa"}
    panels = [("chrf", "Round-trip chrF")] + ([("mutual_entail", "Meaning preserved (mutual NLI ≥ 0.5)")] if "mutual_entail" in recs[0] else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(6.3, 2.4))
    axes = np.atleast_1d(axes)
    w = 0.36
    for ax, (key, title) in zip(axes, panels):
        for k, s in enumerate(systems):
            for j, l in enumerate(langs):
                vals = [r[key] for r in recs if r["lang"] == l and r["system"] == s]
                if key == "mutual_entail":
                    vals = [float(v >= THRESHOLD) for v in vals]
                mean, lo, hi = mean_ci(vals)
                x = j + (k - 0.5) * (w + 0.03)
                ax.bar(x, mean, w, color=COLORS[s], label=LABELS[s] if j == 0 else None)
                ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], color=INK, lw=1, capsize=2.5)
        ax.set_xticks(range(len(langs)), [names[l] for l in langs])
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
        if key == "mutual_entail":
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
            ax.set_ylim(0, 1)
    axes[0].legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    save(fig, "fig_translation")


def fig_example(d):
    """A supported Kàwé sentence with its cited passage, beside the least supported whole-document sentence."""
    from kawe.extraction import ExtractedSegment
    from kawe.ranking import split_into_chunks

    kawe = [g for g in d["grounding"] if g["condition"] == "kawe" and g.get("cited_entail", 0) >= 0.9 and len(g["sentence"].split()) >= 12]
    doc = [g for g in d["grounding"] if g["condition"] == "llm_document" and "best_chunk" in g and len(g["sentence"].split()) >= 10]
    if not kawe or not doc:
        return
    k = max(kawe, key=lambda g: g["cited_entail"])
    u = min(doc, key=lambda g: g["best_entail"])
    by_key = {(r["doc_id"], r["condition"], r["budget"]): r for r in d["summaries"]}
    cited = by_key[(k["doc_id"], "kawe", MAIN)]["sections"][k["section"]]
    paper = d["corpus"][u["doc_id"]]
    chunks = split_into_chunks([ExtractedSegment(p["text"], p["label"], i) for i, p in enumerate(paper["pages"])])
    best = chunks[u["best_chunk"]]

    def wrap(text, width=50, limit=8):
        lines = textwrap.wrap(" ".join(text.split()), width)
        return "\n".join(lines[:limit] + (["[…]"] if len(lines) > limit else []))

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.6))
    panels = [
        (axes[0], "Kàwé: sentence and its citation", COLORS["kawe"], k["sentence"],
         f"Cited passage ({cited['location']}):", cited["source"],
         f"Entailment vs. cited passage: {k['cited_entail']:.2f}", paper_label(k["doc_id"])),
        (axes[1], "Whole-document LLM: weakest sentence", COLORS["llm_document"], u["sentence"],
         f"Best-matching passage ({best.location_label}):", best.text,
         f"Best entailment over 5 retrieved passages: {u['best_entail']:.2f}", paper_label(u["doc_id"])),
    ]
    for ax, title, color, sentence, evidence_title, evidence, verdict, label in panels:
        ax.axis("off")
        ax.text(0, 1.0, title, transform=ax.transAxes, fontsize=8, weight="bold", color=INK, va="bottom")
        ax.text(0, 0.95, f"Paper: {label}", transform=ax.transAxes, fontsize=6.8, color=INK_2, va="top")
        ax.text(0.02, 0.86, wrap(sentence, 48, 5), transform=ax.transAxes, va="top", fontsize=7, color=INK,
                bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=color, lw=1.5))
        ax.text(0.02, 0.52, evidence_title + "\n" + wrap(evidence, 52, 8), transform=ax.transAxes, va="top", fontsize=6.3,
                color=INK_2, bbox=dict(boxstyle="round,pad=0.45", fc="#f4f3ef", ec=GRID))
        ax.text(0.02, 0.0, verdict, transform=ax.transAxes, va="bottom", fontsize=7, color=color, weight="bold")
    fig.subplots_adjust(wspace=0.08)
    save(fig, "fig_example")


def main():
    d = load()
    for fn in (fig_rouge, fig_budget, fig_per_paper, fig_nli_distribution, fig_faithfulness, fig_support_per_paper,
               fig_provenance, fig_abstractive, fig_timing, fig_extraction, fig_translation, fig_example):
        try:
            fn(d)
        except Exception as e:  # one broken figure should not stop the rest
            print(f"skipped {fn.__name__}: {e}")


if __name__ == "__main__":
    main()

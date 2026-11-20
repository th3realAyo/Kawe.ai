"""Translates Kàwé summary sections into Yoruba, Igbo and Hausa and checks them by round trip.

Systems: Kàwé's translator (Llama-3.1-8B-Instruct on Modal) and NLLB-200 (600M, local CPU).
Each translation is translated back into English by the same system. Without reference
translations, the round trip is an automatic proxy for meaning preservation:
  * chrF between the back-translation and the original English section
  * mutual entailment (NLI) between the two, in both directions
Native-speaker adequacy and fluency ratings (make_listening_test.py) remain the primary measure.

Writes results/translations.jsonl, translation_report.md and translation_scores.json.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import defaultdict

from common import LANGUAGES, RESULTS_DIR, NLIJudge, append_jsonl, load_env, markdown_table, read_jsonl, require_env, write_jsonl
from stats import mean_ci

from kawe.modal_client import translate as modal_translate

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
SYSTEM_LABELS = {"llama": "Llama-3.1-8B (Kàwé)", "nllb": "NLLB-200 600M"}
OUT = RESULTS_DIR / "translations.jsonl"


class NLLBTranslator:
    def __init__(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL).eval()

    def translate(self, text: str, source_code: str, target_code: str) -> str:
        import torch

        self.tokenizer.src_lang = source_code
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            output = self.model.generate(
                **inputs, forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(target_code), max_new_tokens=400
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


def sample_sections(n: int, seed: int, budget: int):
    pool = [
        {"id": f"{r['doc_id']}#{i}", "text": s["summary"]}
        for r in read_jsonl(RESULTS_DIR / "summaries.jsonl")
        if r["condition"] == "kawe" and r["budget"] == budget
        for i, s in enumerate(r["sections"])
        if len(s["summary"].split()) >= 8
    ]
    if not pool:
        raise SystemExit("No Kàwé summaries. Run eval/run_summaries.py first.")
    random.Random(seed).shuffle(pool)
    return pool[:n]


def run_system(system: str, text: str, lang: dict, nllb):
    start = time.perf_counter()
    if system == "llama":
        hypothesis = modal_translate(text, lang["name"])
        forward = time.perf_counter() - start
        back = modal_translate(hypothesis, "English")
    else:
        hypothesis = nllb.translate(text, "eng_Latn", lang["nllb"])
        forward = time.perf_counter() - start
        back = nllb.translate(hypothesis, lang["nllb"], "eng_Latn")
    return hypothesis, back, forward


def copied_share(source: str, hypothesis: str) -> float:
    src = {w.lower().strip(".,;:") for w in source.split()}
    hyp = [w.lower().strip(".,;:") for w in hypothesis.split()]
    return sum(w in src for w in hyp) / max(len(hyp), 1)


def score(records, use_nli: bool):
    import sacrebleu

    for r in records:
        r["chrf"] = sacrebleu.sentence_chrf(r["back"], [r["source"]]).score
        r["copied"] = copied_share(r["source"], r["hypothesis"])
    if use_nli:
        judge = NLIJudge()
        forward = judge.score([(r["source"], r["back"]) for r in records])
        backward = judge.score([(r["back"], r["source"]) for r in records])
        for r, (f, _), (b, _) in zip(records, forward, backward):
            r["mutual_entail"] = min(f, b)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sections", type=int, default=30, help="summary sections per language")
    parser.add_argument("--langs", nargs="+", default=list(LANGUAGES), choices=list(LANGUAGES))
    parser.add_argument("--no-nllb", action="store_true")
    parser.add_argument("--skip-nli", action="store_true")
    parser.add_argument("--main-budget", type=int, default=30)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    load_env()
    require_env("MODAL_TRANSLATE_URL", "RAG_AUTH_TOKEN")
    systems = ["llama"] + ([] if args.no_nllb else ["nllb"])
    sections = sample_sections(args.sections, args.seed, args.main_budget)
    done = {(r["system"], r["lang"], r["id"]) for r in read_jsonl(OUT)}
    nllb = None

    for code in args.langs:
        for system in systems:
            for n, section in enumerate(sections, start=1):
                if (system, code, section["id"]) in done:
                    continue
                if system == "nllb" and nllb is None:
                    nllb = NLLBTranslator()
                hypothesis, back, seconds = run_system(system, section["text"], LANGUAGES[code], nllb)
                append_jsonl(OUT, {"system": system, "lang": code, "id": section["id"], "source": section["text"],
                                   "hypothesis": hypothesis, "back": back, "seconds": round(seconds, 2)})
                if n % 10 == 0:
                    print(f"{LANGUAGES[code]['name']} / {system}: {n}/{len(sections)}", flush=True)

    records = score(read_jsonl(OUT), not args.skip_nli)
    groups = defaultdict(list)
    for r in records:
        groups[(r["lang"], r["system"])].append(r)

    rows, summary = [], {}
    for (code, system), rs in sorted(groups.items()):
        chrf = mean_ci([r["chrf"] for r in rs])
        entry = {"n": len(rs), "chrf": list(chrf), "median_seconds": statistics.median(r["seconds"] for r in rs),
                 "copied": statistics.mean(r["copied"] for r in rs)}
        cells = [LANGUAGES[code]["name"], SYSTEM_LABELS[system], len(rs), f"{chrf[0]:.1f} [{chrf[1]:.1f}, {chrf[2]:.1f}]"]
        if "mutual_entail" in rs[0]:
            entry["preserved"] = sum(r["mutual_entail"] >= 0.5 for r in rs) / len(rs)
            cells.append(f"{100 * entry['preserved']:.0f}%")
        else:
            cells.append("–")
        cells += [f"{100 * entry['copied']:.0f}%", f"{entry['median_seconds']:.1f}"]
        summary[f"{system}/{code}"] = entry
        rows.append(cells)

    report = (
        "# Translation round trip (Kàwé summary sections)\n\n"
        + markdown_table(["Language", "System", "Sections", "Round-trip chrF (95% CI)", "Meaning preserved (mutual NLI)",
                          "Words copied from English", "Median seconds"], rows)
        + "\n\nRound-trip scores are a proxy; native-speaker ratings are the primary measure.\n"
    )
    (RESULTS_DIR / "translation_scores.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (RESULTS_DIR / "translation_report.md").write_text(report, encoding="utf-8")
    write_jsonl(RESULTS_DIR / "translation_scored.jsonl", records)
    print(report)


if __name__ == "__main__":
    main()

"""Step 4: translate Kàwé summaries of the Nigerian documents.

Takes summary sections that Kàwé produced for your Nigerian documents
(eval/results/summaries.jsonl) and translates them into Yoruba, Igbo and
Hausa with two systems:
  * llama - Kàwé's translator (Llama-3.1-8B-Instruct on Modal)
  * nllb  - NLLB-200, a dedicated translation model, as a comparison
            (runs on your CPU; downloads about 2.5 GB the first time)

There are no professional reference translations for these texts, so
quality is judged by native speakers: make_listening_test.py puts these
translations, blinded and shuffled, into the rating sheets. This script
also records speed and how often a translation came back empty or
unchanged (a sign the model didn't translate).

Outputs eval/results/translations.jsonl and translation_report.md.

Usage:
    python eval/run_translation_eval.py --sections 30
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time

from common import (
    LANGUAGES,
    RESULTS_DIR,
    append_jsonl,
    load_env,
    markdown_table,
    modal_translate,
    read_jsonl,
    require_env,
)

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
SYSTEM_LABELS = {"llama": "Llama-3.1-8B (Kàwé)", "nllb": "NLLB-200 600M"}


class NLLBTranslator:
    def __init__(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL, src_lang="eng_Latn")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL).eval()

    def translate(self, text: str, target_code: str) -> str:
        import torch

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(target_code),
                max_new_tokens=400,
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


def sample_sections(n: int, seed: int):
    """Picks n Kàwé summary sections, spread across documents."""
    summaries = read_jsonl(RESULTS_DIR / "summaries.jsonl")
    if not summaries:
        raise SystemExit("No summaries yet. Run eval/run_summary_eval.py first.")
    pool = [
        {"id": f"{doc['doc_index']}#{i}", "text": section["summary"]}
        for doc in summaries
        for i, section in enumerate(doc["kawe_sections"])
        if len(section["summary"].split()) >= 8
    ]
    random.Random(seed).shuffle(pool)
    return pool[:n]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sections", type=int, default=30, help="summary sections to translate per language")
    parser.add_argument("--langs", nargs="+", default=list(LANGUAGES), choices=list(LANGUAGES))
    parser.add_argument("--no-nllb", action="store_true", help="skip the NLLB comparison")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    load_env()
    require_env("MODAL_TRANSLATE_URL", "RAG_AUTH_TOKEN")

    out_path = RESULTS_DIR / "translations.jsonl"
    done = {(r["system"], r["lang"], r["id"]) for r in read_jsonl(out_path)}
    sections = sample_sections(args.sections, args.seed)
    systems = ["llama"] + ([] if args.no_nllb else ["nllb"])
    nllb = None

    for lang in args.langs:
        info = LANGUAGES[lang]
        for system in systems:
            for n, section in enumerate(sections, start=1):
                if (system, lang, section["id"]) in done:
                    continue
                start = time.perf_counter()
                if system == "llama":
                    hyp = modal_translate(section["text"], info["name"])
                else:
                    nllb = nllb or NLLBTranslator()
                    hyp = nllb.translate(section["text"], info["nllb"])
                append_jsonl(
                    out_path,
                    {
                        "system": system,
                        "lang": lang,
                        "id": section["id"],
                        "source": section["text"],
                        "hypothesis": hyp,
                        "seconds": round(time.perf_counter() - start, 2),
                    },
                )
                if n % 10 == 0:
                    print(f"{info['name']} / {system}: {n}/{len(sections)}")

    records = read_jsonl(out_path)
    rows, results = [], {}
    for lang in args.langs:
        for system in systems:
            rs = [r for r in records if r["system"] == system and r["lang"] == lang]
            if not rs:
                continue
            failed = sum(1 for r in rs if not r["hypothesis"].strip() or r["hypothesis"].strip() == r["source"].strip())
            ratio = statistics.mean(len(r["hypothesis"].split()) / max(len(r["source"].split()), 1) for r in rs)
            secs = statistics.median(r["seconds"] for r in rs)
            results[f"{system}/{lang}"] = {"n": len(rs), "empty_or_unchanged": failed, "length_ratio": ratio, "median_seconds": secs}
            rows.append([LANGUAGES[lang]["name"], SYSTEM_LABELS[system], len(rs), failed, f"{ratio:.2f}", f"{secs:.1f}"])

    report = (
        "# Translation run (Kàwé summaries of Nigerian documents)\n\n"
        "Quality is rated by native speakers (see ratings_report.md). This table only checks the runs.\n\n"
        + markdown_table(
            ["Language", "System", "Sections", "Empty or untranslated", "Length ratio (target/English words)", "Median seconds"],
            rows,
        )
    )
    (RESULTS_DIR / "translation_run.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (RESULTS_DIR / "translation_report.md").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

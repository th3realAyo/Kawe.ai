"""Summarizes the native-speaker ratings.

Raters return their sheets renamed with their initials, e.g.
    eval/listening_test/yor/audio_ratings_AO.csv
    eval/listening_test/yor/translation_ratings_AO.csv
This script joins them with key.csv and reports the mean opinion score
(MOS) with a 95% confidence interval for each voice and each translation
system.

Usage:
    python eval/analyze_ratings.py
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict

from common import EVAL_DIR, RESULTS_DIR, markdown_table

OUT_DIR = EVAL_DIR / "listening_test"
LANG_NAMES = {"eng": "English", "yor": "Yoruba", "ibo": "Igbo", "hau": "Hausa"}
SYSTEM_NAMES = {"llama": "Llama-3.1-8B (Kàwé)", "nllb": "NLLB-200 600M"}


def read_csv(path):
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def mos(values):
    """Mean with a normal-approximation 95% confidence interval."""
    if not values:
        return "–"
    mean = statistics.mean(values)
    if len(values) < 2:
        return f"{mean:.2f}"
    half = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return f"{mean:.2f} ± {half:.2f}"


def score(row, column):
    try:
        value = float(row[column])
    except (KeyError, ValueError):
        return None
    return value if 1 <= value <= 5 else None


def main():
    key = {(r["task"], r["lang"], r["item"]): r for r in read_csv(OUT_DIR / "key.csv")}

    audio = defaultdict(lambda: {"nat": [], "int": [], "raters": set()})
    trans = defaultdict(lambda: {"adq": [], "flu": [], "raters": set()})

    for lang_dir in sorted(p for p in OUT_DIR.iterdir() if p.is_dir()):
        lang = lang_dir.name
        for sheet in lang_dir.glob("audio_ratings_*.csv"):
            rater = sheet.stem.split("_")[-1]
            for row in read_csv(sheet):
                k = key.get(("audio", lang, row["clip"]))
                if not k:
                    continue
                group = audio[(lang, k["system"], k["voice"])]
                group["raters"].add(rater)
                for col, bucket in (("naturalness_1_to_5", "nat"), ("intelligibility_1_to_5", "int")):
                    v = score(row, col)
                    if v is not None:
                        group[bucket].append(v)
        for sheet in lang_dir.glob("translation_ratings_*.csv"):
            rater = sheet.stem.split("_")[-1]
            for row in read_csv(sheet):
                k = key.get(("translation", lang, row["item"]))
                if not k:
                    continue
                group = trans[(lang, k["system"])]
                group["raters"].add(rater)
                for col, bucket in (("adequacy_1_to_5", "adq"), ("fluency_1_to_5", "flu")):
                    v = score(row, col)
                    if v is not None:
                        group[bucket].append(v)

    if not audio and not trans:
        raise SystemExit("No filled rating sheets found (expected files like yor/audio_ratings_AO.csv).")

    report = ["# Native-speaker ratings (mean ± 95% CI, scale 1–5)\n", "## Table 5: Speech quality\n"]
    audio_rows, per_lang = [], defaultdict(lambda: {"nat": [], "int": [], "raters": set()})
    for (lang, engine, voice), g in sorted(audio.items()):
        audio_rows.append([LANG_NAMES[lang], engine, voice, len(g["raters"]), mos(g["nat"]), mos(g["int"])])
        per_lang[lang]["nat"] += g["nat"]
        per_lang[lang]["int"] += g["int"]
        per_lang[lang]["raters"] |= g["raters"]
    for lang, g in sorted(per_lang.items()):
        audio_rows.append([f"**{LANG_NAMES[lang]} (all voices)**", "", "", len(g["raters"]), mos(g["nat"]), mos(g["int"])])
    report.append(markdown_table(["Language", "Engine", "Voice", "Raters", "Naturalness", "Intelligibility"], audio_rows))

    if trans:
        report.append("\n\n## Table 6: Translation quality\n")
        report.append(
            markdown_table(
                ["Language", "System", "Raters", "Adequacy", "Fluency"],
                [
                    [LANG_NAMES[lang], SYSTEM_NAMES.get(system, system), len(g["raters"]), mos(g["adq"]), mos(g["flu"])]
                    for (lang, system), g in sorted(trans.items())
                ],
            )
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "ratings_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    raw = {
        "audio": {"/".join(k): {"naturalness": v["nat"], "intelligibility": v["int"]} for k, v in audio.items()},
        "translation": {"/".join(k): {"adequacy": v["adq"], "fluency": v["flu"]} for k, v in trans.items()},
    }
    (RESULTS_DIR / "ratings_raw.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()

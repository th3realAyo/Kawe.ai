"""Step 5: build the rating packs for native speakers.

For each language this creates a folder under eval/listening_test/ with:

  clips/                  audio clips with random names (raters can't tell
                          which voice made them). Up to --max-voices YarnGPT
                          voices per language, taken from the live voice list.
  audio_ratings.csv       one row per clip: naturalness and intelligibility, 1-5
  translation_ratings.csv (Yoruba/Igbo/Hausa) one row per translated summary
                          section: adequacy and fluency, 1-5. The two
                          translation systems are mixed and hidden.
  RATER_INSTRUCTIONS.md   what raters do

Where the spoken text comes from (all Nigerian):
  * Yoruba, Igbo, Hausa: sentences YOU collect from Nigerian-language
    sources and save one per line in eval/native_sentences/yor.txt,
    ibo.txt and hau.txt (see eval/README.md). Using text written by
    native speakers means the ratings measure the voices, not translation
    mistakes.
  * English: sentences taken automatically from your Nigerian documents.

The answer key goes to eval/listening_test/key.csv: keep it to yourself.

Usage:
    python eval/make_listening_test.py --sentences 8
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import uuid

from common import (
    EVAL_DIR,
    LANGUAGES,
    RESULTS_DIR,
    load_env,
    yarngpt_speak,
    read_jsonl,
    require_env,
)

OUT_DIR = EVAL_DIR / "listening_test"
NATIVE_DIR = EVAL_DIR / "native_sentences"

AUDIO_COLUMNS = ["clip", "naturalness_1_to_5", "intelligibility_1_to_5", "comments"]
TRANSLATION_COLUMNS = ["item", "english", "translation", "adequacy_1_to_5", "fluency_1_to_5", "comments"]


def write_sheet(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:  # -sig: opens cleanly in Excel
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)


def native_sentences(lang: str, n: int):
    path = NATIVE_DIR / f"{lang}.txt"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Add {n}+ {LANGUAGES[lang]['name']} sentences, one per line.")
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8-sig").splitlines()]
    sentences = [ln for ln in lines if ln and not ln.startswith("#")]
    if len(sentences) < n:
        raise SystemExit(f"{path} has {len(sentences)} sentences; need at least {n}.")
    return [{"id": f"{lang}-{i + 1}", "text": s} for i, s in enumerate(sentences[:n])]


def english_sentences(n: int, rng: random.Random):
    """Plain sentences of 8-30 words from the Nigerian documents' body text."""
    from nltk.tokenize import sent_tokenize

    pool = []
    for doc in read_jsonl(RESULTS_DIR / "corpus.jsonl"):
        for page in doc["pages"]:
            for s in sent_tokenize(" ".join(page["text"].split())):
                words = s.split()
                digits = sum(ch.isdigit() for ch in s)
                if 8 <= len(words) <= 30 and digits <= 4 and re.match(r"^[A-Z]", s) and s.endswith("."):
                    pool.append({"id": f"{doc['doc_id']}:{page['label']}", "text": s})
    if len(pool) < n:
        raise SystemExit("Not enough English sentences. Run eval/prepare_corpus.py first.")
    rng.shuffle(pool)
    return pool[:n]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sentences", type=int, default=8, help="sentences per voice")
    parser.add_argument("--translation-items", type=int, default=30, help="translations to rate per language (split equally between systems)")
    parser.add_argument("--skip-english", action="store_true", help="don't spend ElevenLabs credits")
    parser.add_argument("--max-voices", type=int, default=4, help="YarnGPT voices to test per language")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    load_env()
    require_env("YARNGPT_API_KEY")
    rng = random.Random(args.seed)

    import nltk

    for pkg in ("punkt", "punkt_tab"):
        nltk.download(pkg, quiet=True)

    # Check every input before spending any GPU time.
    texts = {lang: native_sentences(lang, args.sentences) for lang in LANGUAGES}
    from yarngpt_tts import list_voices, voices_for_language

    catalogue = list_voices()
    plan = []
    for lang in LANGUAGES:
        voices = voices_for_language(LANGUAGES[lang]["name"], catalogue)
        if len(voices) > args.max_voices:  # same random choice every run (seeded)
            voices = sorted(random.Random(f"{args.seed}-{lang}").sample(voices, args.max_voices))
        print(f"{LANGUAGES[lang]['name']}: testing voices {', '.join(voices)}")
        plan += [(lang, "yarngpt", voice) for voice in voices]
    if not args.skip_english:
        require_env("ELEVENLABS_API_KEY")
        from elevenlabs_tts import VOICE_NAME_TO_ID

        texts["eng"] = english_sentences(args.sentences, rng)
        plan += [("eng", "elevenlabs", voice) for voice in VOICE_NAME_TO_ID]

    by_lang = {}
    for lang, engine, voice in plan:
        by_lang.setdefault(lang, []).append((engine, voice))

    key_rows = []
    for lang, voices in by_lang.items():
        lang_dir = OUT_DIR / lang
        clip_dir = lang_dir / "clips"
        clip_dir.mkdir(parents=True, exist_ok=True)

        audio_rows = []
        for engine, voice in voices:
            for sentence in texts[lang]:
                clip_id = uuid.UUID(int=rng.getrandbits(128)).hex[:8]
                if engine == "yarngpt":
                    audio, ext = yarngpt_speak(sentence["text"], voice)
                else:
                    from elevenlabs_tts import synthesize_with_elevenlabs

                    audio, ext = synthesize_with_elevenlabs(sentence["text"], voice_name=voice), "mp3"
                (clip_dir / f"{clip_id}.{ext}").write_bytes(audio)
                audio_rows.append([f"{clip_id}.{ext}", "", "", ""])
                key_rows.append(["audio", lang, f"{clip_id}.{ext}", engine, voice, sentence["id"], sentence["text"]])
            print(f"{lang}: {voice} done")

        rng.shuffle(audio_rows)
        write_sheet(lang_dir / "audio_ratings.csv", AUDIO_COLUMNS, audio_rows)
        (lang_dir / "RATER_INSTRUCTIONS.md").write_text(
            (EVAL_DIR / "RATER_INSTRUCTIONS.md").read_text(encoding="utf-8"), encoding="utf-8"
        )

        if lang == "eng":
            continue
        translations = [r for r in read_jsonl(RESULTS_DIR / "translations.jsonl") if r["lang"] == lang]
        if not translations:
            print(f"{lang}: no translations yet; run run_translation_eval.py first to add a translation sheet")
            continue
        # Equal number of items from each translation system, then shuffled
        # together so raters can't tell which system produced which item.
        systems = sorted({r["system"] for r in translations})
        per_system = args.translation_items // len(systems)
        chosen = []
        for system in systems:
            pool = [r for r in translations if r["system"] == system]
            rng.shuffle(pool)
            chosen += pool[:per_system]
        rng.shuffle(chosen)
        translation_rows = []
        for r in chosen:
            item = uuid.UUID(int=rng.getrandbits(128)).hex[:8]
            translation_rows.append([item, r["source"], r["hypothesis"], "", "", ""])
            key_rows.append(["translation", lang, item, r["system"], "", r["id"], r["source"]])
        write_sheet(lang_dir / "translation_ratings.csv", TRANSLATION_COLUMNS, translation_rows)

    write_sheet(OUT_DIR / "key.csv", ["task", "lang", "item", "system", "voice", "source_id", "text"], key_rows)
    print(f"\nRating packs are in {OUT_DIR}. Send each language folder to its raters.")
    print(f"Keep {OUT_DIR / 'key.csv'} private: it reveals which clip is which.")


if __name__ == "__main__":
    main()

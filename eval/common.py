"""Shared helpers for the Kàwé evaluation scripts.

Every script writes its raw outputs as JSON Lines (one record per line) and
skips records that are already there, so an interrupted run can simply be
started again without paying for the same Modal calls twice.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable, List

import requests

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
RESULTS_DIR = EVAL_DIR / "results"

# Make the app's own modules (doc_loader, chunk_ranker, ...) importable so
# the evaluation exercises exactly the code the app runs.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Same names the app shows users; the Modal endpoints expect these.
LANGUAGES = {
    "yor": {"name": "Yoruba", "nllb": "yor_Latn"},
    "ibo": {"name": "Igbo", "nllb": "ibo_Latn"},
    "hau": {"name": "Hausa", "nllb": "hau_Latn"},
}


def load_env() -> None:
    """Loads the app's .env file so the Modal URLs and token are available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / ".env")


def require_env(*names: str) -> None:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        sys.exit(f"Missing settings in .env: {', '.join(missing)}")


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def markdown_table(headers: List[str], rows: Iterable[List]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def modal_translate(text: str, target_language: str, max_new_tokens: int = 512) -> str:
    """Same request the app makes (main.call_modal_translate)."""
    response = requests.post(
        os.environ["MODAL_TRANSLATE_URL"],
        json={"text": text, "target_language": target_language, "max_new_tokens": max_new_tokens},
        headers={"Authorization": f"Bearer {os.environ['RAG_AUTH_TOKEN']}"},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["translated_text"]


def yarngpt_speak(text: str, voice: str):
    """Same YarnGPT API call the app makes. Returns (audio_bytes, extension)."""
    from yarngpt_tts import synthesize

    audio, ext, _ = synthesize(text, voice)
    return audio, ext


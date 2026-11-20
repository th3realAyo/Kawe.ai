"""Paths and I/O helpers shared by the evaluation scripts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable, List

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
PAPERS_DIR = EVAL_DIR / "evaluation_papers"
RESULTS_DIR = EVAL_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LANGUAGES = {
    "yor": {"name": "Yoruba", "nllb": "yor_Latn"},
    "ibo": {"name": "Igbo", "nllb": "ibo_Latn"},
    "hau": {"name": "Hausa", "nllb": "hau_Latn"},
}
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# Short names for tables and figures, keyed by the start of the document id.
PAPER_LABELS = {
    "1785165118": "University education (NG)",
    "are-penalty": "Penalty shootouts",
    "eight-things": "Eight things about LLMs",
    "impact-of-ai": "AI tools and learning",
    "improving-s-p-500": "S&P 500 volatility",
    "neurips-2025": "LLM-as-a-judge",
    "sentiment-and-volatility": "Sentiment and volatility",
}


def paper_label(doc_id: str) -> str:
    return next((label for prefix, label in PAPER_LABELS.items() if doc_id.startswith(prefix)), doc_id)


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / ".env")


def require_env(*names: str) -> None:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        sys.exit(f"Missing settings in .env: {', '.join(missing)}")


def ensure_nltk() -> None:
    import nltk

    for package in ("punkt", "punkt_tab", "stopwords"):
        nltk.download(package, quiet=True)


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def markdown_table(headers: List[str], rows: Iterable[List]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


class NLIJudge:
    """Entailment and contradiction probabilities from an MNLI/FEVER/ANLI DeBERTa model."""

    def __init__(self, model_name: str = NLI_MODEL):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.entail_idx, self.contra_idx = labels["entailment"], labels["contradiction"]

    def score(self, pairs: List[tuple], batch_size: int = 8) -> List[tuple]:
        """[(premise, hypothesis)] -> [(p_entail, p_contradict)]."""
        out = []
        for k in range(0, len(pairs), batch_size):
            batch = pairs[k : k + batch_size]
            inputs = self.tokenizer(
                [p for p, _ in batch], [h for _, h in batch],
                truncation="only_first", max_length=512, padding=True, return_tensors="pt",
            )
            with self.torch.no_grad():
                probs = self.torch.softmax(self.model(**inputs).logits, dim=-1)
            out += [(float(r[self.entail_idx]), float(r[self.contra_idx])) for r in probs]
        return out

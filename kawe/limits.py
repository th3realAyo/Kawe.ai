"""Request limits and token checks for the inference endpoints.

Kept free of Modal and FastAPI imports so it can be unit-tested; modal_app.py ships this package
into the container and calls these functions.
"""

from __future__ import annotations

import hmac
from typing import List, Optional

APP, EVAL = "app", "eval"

MAX_EMBED_TEXTS = 2000
MAX_EMBED_TEXT_CHARS = 8000
MAX_PASSAGE_CHARS = 6000  # a 180-word passage is ~1,200 characters; leaves room for long words
MAX_BATCH_PASSAGES = 64
MAX_SUMMARY_TOKENS = 256
MAX_TRANSLATE_CHARS = 6000
MAX_TRANSLATE_TOKENS = 1024
# The evaluation's whole-document baseline sends a full paper and asks for a long summary.
EVAL_MAX_TEXT_CHARS = 300_000
EVAL_MAX_SUMMARY_TOKENS = 4096
MAX_LANGUAGE_CHARS = 40


class RequestRejected(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _matches(supplied: str, expected: Optional[str]) -> bool:
    return bool(expected) and hmac.compare_digest(supplied.encode(), expected.encode())


def caller_role(authorization: Optional[str], app_token: Optional[str], eval_token: Optional[str] = None) -> str:
    """Returns APP or EVAL for a valid bearer token; raises RequestRejected(401) otherwise."""
    if authorization and authorization.startswith("Bearer "):
        supplied = authorization[len("Bearer "):]
        if _matches(supplied, eval_token):
            return EVAL
        if _matches(supplied, app_token):
            return APP
    raise RequestRejected(401, "Invalid or missing auth token")


def _check(condition: bool, detail: str, status_code: int = 413) -> None:
    if not condition:
        raise RequestRejected(status_code, detail)


def check_embed(texts: List[str]) -> None:
    _check(0 < len(texts) <= MAX_EMBED_TEXTS, f"Send between 1 and {MAX_EMBED_TEXTS} texts.")
    _check(all(len(t) <= MAX_EMBED_TEXT_CHARS for t in texts), f"Each text must be at most {MAX_EMBED_TEXT_CHARS} characters.")


def check_summarize(text: str, max_new_tokens: int, language: str, instruction: Optional[str], role: str) -> None:
    _check(instruction is None or role == EVAL, "Custom instructions need the evaluation token.", status_code=403)
    max_chars = EVAL_MAX_TEXT_CHARS if role == EVAL else MAX_PASSAGE_CHARS
    max_tokens = EVAL_MAX_SUMMARY_TOKENS if role == EVAL else MAX_SUMMARY_TOKENS
    _check(0 < len(text) <= max_chars, f"Text must be between 1 and {max_chars} characters.")
    _check(0 < max_new_tokens <= max_tokens, f"max_new_tokens must be between 1 and {max_tokens}.", status_code=422)
    _check(0 < len(language) <= MAX_LANGUAGE_CHARS, "Invalid language.", status_code=422)


def check_summarize_batch(texts: List[str], max_new_tokens: int, language: str) -> None:
    _check(0 < len(texts) <= MAX_BATCH_PASSAGES, f"Send between 1 and {MAX_BATCH_PASSAGES} passages.")
    _check(all(0 < len(t) <= MAX_PASSAGE_CHARS for t in texts), f"Each passage must be 1 to {MAX_PASSAGE_CHARS} characters.")
    _check(0 < max_new_tokens <= MAX_SUMMARY_TOKENS, f"max_new_tokens must be between 1 and {MAX_SUMMARY_TOKENS}.", status_code=422)
    _check(0 < len(language) <= MAX_LANGUAGE_CHARS, "Invalid language.", status_code=422)


def check_translate(text: str, target_language: str, max_new_tokens: int) -> None:
    _check(0 < len(text) <= MAX_TRANSLATE_CHARS, f"Text must be between 1 and {MAX_TRANSLATE_CHARS} characters.")
    _check(0 < max_new_tokens <= MAX_TRANSLATE_TOKENS, f"max_new_tokens must be between 1 and {MAX_TRANSLATE_TOKENS}.", status_code=422)
    _check(0 < len(target_language) <= MAX_LANGUAGE_CHARS, "Invalid language.", status_code=422)

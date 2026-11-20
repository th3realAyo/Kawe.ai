"""Client for the inference endpoints deployed by modal_app.py."""

from __future__ import annotations

import os
from typing import List, Optional

import requests


class ModalServiceError(Exception):
    """An inference endpoint could not be reached or returned an error.

    `retryable` is False for errors a retry cannot fix (missing settings, bad token, rejected input).
    """

    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


def _error_detail(response) -> str:
    """The server's own error message, if it sent a plain one.

    The raw body is never used: validation errors echo the request, which would put document
    text into error messages and logs.
    """
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        return ""
    return f": {detail[:200]}" if isinstance(detail, str) else ""


def _post(url_variable: str, payload: dict, timeout: int, token_variable: str = "RAG_AUTH_TOKEN") -> dict:
    url = os.environ.get(url_variable)
    token = os.environ.get(token_variable)
    if not url or not token:
        raise ModalServiceError(f"{url_variable} and {token_variable} must be set.", retryable=False)
    try:
        response = requests.post(url, json=payload, headers={"Authorization": f"Bearer {token}"}, timeout=timeout)
    except requests.RequestException as e:
        raise ModalServiceError(f"Could not reach {url_variable}: {e}") from e
    if response.status_code != 200:
        retryable = response.status_code == 429 or response.status_code >= 500
        raise ModalServiceError(f"{url_variable} returned {response.status_code}{_error_detail(response)}", retryable=retryable)
    try:
        data = response.json()
    except ValueError as e:
        raise ModalServiceError(f"{url_variable} returned a response that is not JSON.") from e
    if not isinstance(data, dict):
        raise ModalServiceError(f"{url_variable} returned an unexpected response.")
    return data


def _field(data: dict, key: str, url_variable: str):
    if key not in data:
        raise ModalServiceError(f"{url_variable} response has no '{key}' field; is the deployed app up to date?", retryable=False)
    return data[key]


def embed(texts: List[str]) -> List[List[float]]:
    """L2-normalized passage embeddings."""
    return _field(_post("MODAL_EMBED_URL", {"texts": texts}, timeout=300), "embeddings", "MODAL_EMBED_URL")


def summarize(
    text: str,
    language: str = "English",
    max_new_tokens: int = 120,
    instruction: Optional[str] = None,
) -> str:
    """Passage-constrained rewrite.

    `instruction` replaces the default system prompt. The server accepts it only with the
    evaluation token, so such calls are sent with EVAL_AUTH_TOKEN instead of RAG_AUTH_TOKEN.
    """
    payload = {"text": text, "language": language, "max_new_tokens": max_new_tokens}
    token_variable = "RAG_AUTH_TOKEN"
    if instruction:
        payload["instruction"] = instruction
        token_variable = "EVAL_AUTH_TOKEN"
    data = _post("MODAL_SUMMARIZE_URL", payload, timeout=900, token_variable=token_variable)
    if instruction and not data.get("instruction_used"):
        raise ModalServiceError("The deployed summarizer ignores custom instructions; run `modal deploy modal_app.py`.", retryable=False)
    return _field(data, "summary", "MODAL_SUMMARIZE_URL")


def summarize_batch(texts: List[str], language: str = "English", max_new_tokens: int = 120) -> List[str]:
    """Rewrites several passages in one request; the server generates them as a batch."""
    payload = {"texts": texts, "language": language, "max_new_tokens": max_new_tokens}
    summaries = _field(_post("MODAL_SUMMARIZE_BATCH_URL", payload, timeout=900), "summaries", "MODAL_SUMMARIZE_BATCH_URL")
    if not isinstance(summaries, list) or len(summaries) != len(texts):
        raise ModalServiceError("MODAL_SUMMARIZE_BATCH_URL returned the wrong number of summaries.", retryable=False)
    return summaries


def translation_token_budget(text: str) -> int:
    """Output tokens to allow for a translation.

    Llama's tokenizer splits Yoruba, Igbo and Hausa (with their diacritics) into several tokens
    per word, so a fixed small limit cut translations off.
    """
    return max(128, min(1024, 6 * len(text.split()) + 64))


def translate(text: str, target_language: str, max_new_tokens: Optional[int] = None) -> str:
    payload = {
        "text": text,
        "target_language": target_language,
        "max_new_tokens": max_new_tokens or translation_token_budget(text),
    }
    return _field(_post("MODAL_TRANSLATE_URL", payload, timeout=300), "translated_text", "MODAL_TRANSLATE_URL")

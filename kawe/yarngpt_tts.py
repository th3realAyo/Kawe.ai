"""Client for the hosted YarnGPT text-to-speech API (https://yarngpt.ai/api-docs).

Synthesis is an asynchronous job: POST /api/v1/tts returns a job id, and
GET /api/v1/status/{job_id} is polled until it completes with a signed audio URL.
The API takes no language field, so text must already be in the target language.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Dict, List, Optional, Tuple

import requests

from kawe.text import split_text as _split_text

DEFAULT_BASE_URL = "https://api.yarngpt.ai"
MAX_CHARS = 15_000  # API limit per request
CHUNK_CHARS = 14_000

# Used only when the live voice list cannot be fetched.
FALLBACK_VOICES = [
    "umar", "osagie", "mary", "adaora", "zainab", "adam", "idera", "emma",
    "wura", "jude", "chinenye", "tayo", "regina", "femi", "nonso", "remi",
]

LANGUAGE_ALIASES = {
    "english": {"en", "eng", "english"},
    "yoruba": {"yo", "yor", "yoruba", "yorùbá"},
    "igbo": {"ig", "ibo", "igbo", "ìgbò"},
    "hausa": {"ha", "hau", "hausa"},
}


class YarnGPTError(Exception):
    """Synthesis failed; the message is safe to show to users."""


def _base_url() -> str:
    return os.environ.get("YARNGPT_BASE_URL", DEFAULT_BASE_URL)


def _api_key() -> str:
    key = os.environ.get("YARNGPT_API_KEY")
    if not key:
        raise YarnGPTError("YARNGPT_API_KEY is not set.")
    return key

def _headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {_api_key()}"}

def _error_message(response: requests.Response) -> str:
    """The API's error body is {"error": {"code", "message", "user_message", ...}}."""
    try:
        error = response.json().get("error", {})
        text = error.get("user_message") or error.get("message") or response.text
        code = error.get("code")
        return f"{text} ({code})" if code else text
    except ValueError:
        return f"HTTP {response.status_code}: {response.text[:200]}"

def _request(method: str, url: str, retries: int = 3, **kwargs) -> requests.Response:
    """Sends a request, waiting and retrying on rate limits and server errors."""
    for attempt in range(retries + 1):
        response = requests.request(method, url, timeout=60, **kwargs)
        if response.status_code == 429 or response.status_code >= 500:
            if attempt < retries:
                wait = float(response.headers.get("Retry-After", 2 ** attempt))
                time.sleep(min(wait, 30))
                continue
        if response.status_code >= 400:
            raise YarnGPTError(_error_message(response))
        return response
    raise YarnGPTError(_error_message(response))


def _parse_voice(entry) -> Optional[dict]:
    """Accepts a plain string or a dict with an id/name and optional languages."""
    if isinstance(entry, str):
        return {"id": entry, "languages": None, "default": False}
    if not isinstance(entry, dict):
        return None
    voice_id = entry.get("id") or entry.get("voice") or entry.get("name")
    if not voice_id:
        return None
    langs = entry.get("languages") or entry.get("language") or entry.get("supported_languages")
    if isinstance(langs, str):
        langs = [langs]
    if langs:
        langs = [str(lang).lower() if not isinstance(lang, dict) else str(lang.get("code") or lang.get("name", "")).lower() for lang in langs]
    return {"id": str(voice_id), "languages": langs or None, "default": bool(entry.get("default"))}


def list_voices() -> List[dict]:
    """Live voice catalogue, or FALLBACK_VOICES if it can't be fetched."""
    try:
        data = _request("GET", f"{_base_url()}/api/v1/voices", headers=_headers(), retries=1).json()
    except (YarnGPTError, requests.RequestException, ValueError):
        return [{"id": v, "languages": None, "default": False} for v in FALLBACK_VOICES]
    if isinstance(data, dict):
        data = data.get("voices") or data.get("data") or data.get("items") or []
    voices = [v for v in (_parse_voice(e) for e in data) if v]
    return voices or [{"id": v, "languages": None, "default": False} for v in FALLBACK_VOICES]


def voices_for_language(language: str, voices: Optional[List[dict]] = None) -> List[str]:
    """Voices that list the language; all voices if the catalogue has no language data."""
    voices = voices if voices is not None else list_voices()
    aliases = LANGUAGE_ALIASES.get(language.lower(), {language.lower()})
    if not any(v["languages"] for v in voices):
        return [v["id"] for v in voices]
    matching = [v["id"] for v in voices if v["languages"] and aliases & set(v["languages"])]
    return matching or [v["id"] for v in voices]


def split_text(text: str, max_chars: int = CHUNK_CHARS) -> List[str]:
    """Splits text at sentence ends into pieces under max_chars."""
    return _split_text(text, max_chars)


def _synthesize_chunk(text: str, voice: str, output_format: str, poll_seconds: float, timeout_seconds: float) -> Tuple[bytes, List[str]]:
    body = {"text": text, "voice": voice, "output_format": output_format}
    idempotency_key = hashlib.sha256(f"{voice}|{output_format}|{text}".encode()).hexdigest()[:64]
    job = _request(
        "POST",
        f"{_base_url()}/api/v1/tts",
        headers={**_headers(), "Idempotency-Key": idempotency_key},
        json=body,
    ).json()
    warnings = [w.get("message", str(w)) if isinstance(w, dict) else str(w) for w in job.get("warnings") or []]
    job_id = job["job_id"]

    deadline = time.monotonic() + timeout_seconds
    while True:
        status = _request("GET", f"{_base_url()}/api/v1/status/{job_id}", headers=_headers()).json()
        if status.get("status") == "completed":
            break
        if status.get("status") == "failed":
            raise YarnGPTError(status.get("user_message") or "YarnGPT could not generate this audio.")
        if time.monotonic() > deadline:
            raise YarnGPTError(f"YarnGPT job {job_id} did not finish within {timeout_seconds:.0f} seconds.")
        time.sleep(poll_seconds)

    # The signed URL is the credential; the API key is not sent to the storage host.
    audio = requests.get(status["audio_url"], timeout=120)
    if audio.status_code >= 400 or not audio.content:
        raise YarnGPTError(f"Could not download the audio (HTTP {audio.status_code}).")
    return audio.content, warnings


def synthesize(
    text: str,
    voice: str,
    poll_seconds: float = 2.0,
    timeout_seconds: float = 600.0,
) -> Tuple[bytes, str, List[str]]:
    """Returns (audio_bytes, extension, warnings): one MP3, or joined WAV pieces for long text."""
    chunks = split_text(text)
    if not chunks:
        raise YarnGPTError("There is no text to speak.")
    if len(chunks) == 1:
        audio, warnings = _synthesize_chunk(chunks[0], voice, "mp3", poll_seconds, timeout_seconds)
        return audio, "mp3", warnings

    from kawe.audio import join_wav_clips

    clips, warnings = [], []
    for chunk in chunks:
        audio, chunk_warnings = _synthesize_chunk(chunk, voice, "wav", poll_seconds, timeout_seconds)
        clips.append(audio)
        warnings += [w for w in chunk_warnings if w not in warnings]
    return join_wav_clips(clips), "wav", warnings

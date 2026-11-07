from __future__ import annotations

import hashlib
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import requests

BASE_URL = os.environ.get("YARNGPT_BASE_URL", "https://api.yarngpt.ai")
MAX_CHARS = 15_000  # API limit per request
CHUNK_CHARS = 14_000  # leave headroom below the limit

# Voices available when this was written; used only if the live list fails.
FALLBACK_VOICES = [
    "umar", "osagie", "mary", "adaora", "zainab", "adam", "idera", "emma",
    "wura", "jude", "chinenye", "tayo", "regina", "femi", "nonso", "remi",
]

# Language names/codes a voice entry might use, per Kàwé audio language.
LANGUAGE_ALIASES = {
    "english": {"en", "eng", "english"},
    "yoruba": {"yo", "yor", "yoruba", "yorùbá"},
    "igbo": {"ig", "ibo", "igbo", "ìgbò"},
    "hausa": {"ha", "hau", "hausa"},
}


class YarnGPTError(Exception):
    """Raised when YarnGPT synthesis fails; the message is safe to show users."""


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


# ---------------------------------------------------------------------------
# Voices
# ---------------------------------------------------------------------------


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
        data = _request("GET", f"{BASE_URL}/api/v1/voices", headers=_headers(), retries=1).json()
    except (YarnGPTError, requests.RequestException, ValueError):
        return [{"id": v, "languages": None, "default": False} for v in FALLBACK_VOICES]
    if isinstance(data, dict):
        data = data.get("voices") or data.get("data") or data.get("items") or []
    voices = [v for v in (_parse_voice(e) for e in data) if v]
    return voices or [{"id": v, "languages": None, "default": False} for v in FALLBACK_VOICES]


def voices_for_language(language: str, voices: Optional[List[dict]] = None) -> List[str]:
    """Voice ids that list the language. If the catalogue gives no language
    information at all, every voice is returned."""
    voices = voices if voices is not None else list_voices()
    aliases = LANGUAGE_ALIASES.get(language.lower(), {language.lower()})
    if not any(v["languages"] for v in voices):
        return [v["id"] for v in voices]
    matching = [v["id"] for v in voices if v["languages"] and aliases & set(v["languages"])]
    return matching or [v["id"] for v in voices]


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


def split_text(text: str, max_chars: int = CHUNK_CHARS) -> List[str]:
    """Splits text at sentence ends into pieces under max_chars."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text] if text else []
    chunks, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        while len(sentence) > max_chars:  # a single enormous "sentence"
            chunks.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def _synthesize_chunk(text: str, voice: str, output_format: str, poll_seconds: float, timeout_seconds: float) -> Tuple[bytes, List[str]]:
    body = {"text": text, "voice": voice, "output_format": output_format}
    # Same text + voice + format -> same key, so a retry never pays twice.
    idempotency_key = hashlib.sha256(f"{voice}|{output_format}|{text}".encode()).hexdigest()[:64]
    job = _request(
        "POST",
        f"{BASE_URL}/api/v1/tts",
        headers={**_headers(), "Idempotency-Key": idempotency_key},
        json=body,
    ).json()
    warnings = [w.get("message", str(w)) if isinstance(w, dict) else str(w) for w in job.get("warnings") or []]
    job_id = job["job_id"]

    deadline = time.monotonic() + timeout_seconds
    while True:
        status = _request("GET", f"{BASE_URL}/api/v1/status/{job_id}", headers=_headers()).json()
        if status.get("status") == "completed":
            break
        if status.get("status") == "failed":
            raise YarnGPTError(status.get("user_message") or "YarnGPT could not generate this audio.")
        if time.monotonic() > deadline:
            raise YarnGPTError(f"YarnGPT job {job_id} did not finish within {timeout_seconds:.0f} seconds.")
        time.sleep(poll_seconds)

    # The signed link is the credential; don't send the API key to the storage host.
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
    """Speaks text with a YarnGPT voice.

    Returns (audio_bytes, file_extension, warnings). Text within the API limit
    comes back as one MP3. Longer text is sent in pieces as WAV and joined.
    """
    chunks = split_text(text)
    if not chunks:
        raise YarnGPTError("There is no text to speak.")
    if len(chunks) == 1:
        audio, warnings = _synthesize_chunk(chunks[0], voice, "mp3", poll_seconds, timeout_seconds)
        return audio, "mp3", warnings

    from audio_utils import join_wav_clips

    clips, warnings = [], []
    for chunk in chunks:
        audio, chunk_warnings = _synthesize_chunk(chunk, voice, "wav", poll_seconds, timeout_seconds)
        clips.append(audio)
        warnings += [w for w in chunk_warnings if w not in warnings]
    return join_wav_clips(clips), "wav", warnings

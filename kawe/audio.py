"""Joining audio clips synthesized piece by piece."""

from __future__ import annotations

import io
from typing import List

import numpy as np
import soundfile as sf

PAUSE_SECONDS = 0.4


def join_wav_clips(clips: List[bytes], pause_seconds: float = PAUSE_SECONDS) -> bytes:
    """Concatenates WAV clips of one sample rate into a 16-bit WAV with short pauses between them."""
    if not clips:
        raise ValueError("No audio clips to join.")

    pieces, sample_rate = [], None
    for clip in clips:
        data, rate = sf.read(io.BytesIO(clip), dtype="float32")
        if sample_rate is None:
            sample_rate = rate
        elif rate != sample_rate:
            raise ValueError(f"Clips have different sample rates ({sample_rate} vs {rate}).")
        if pieces:
            pieces.append(np.zeros((int(sample_rate * pause_seconds),) + data.shape[1:], dtype="float32"))
        pieces.append(data)

    buffer = io.BytesIO()
    sf.write(buffer, np.concatenate(pieces), sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()

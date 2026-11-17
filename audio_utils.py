"""Helpers for assembling audio produced section by section.

YarnGPT stops generating after a fixed number of audio tokens (roughly 50
seconds of speech), so a long summary sent in one request is silently cut
off. Kàwé therefore synthesizes each summary section separately and joins
the clips here.
"""

from __future__ import annotations

import io
from typing import List

import numpy as np
import soundfile as sf

PAUSE_BETWEEN_SECTIONS_SECONDS = 0.4


def join_wav_clips(clips: List[bytes], pause_seconds: float = PAUSE_BETWEEN_SECTIONS_SECONDS) -> bytes:
    """Joins WAV clips (same sample rate) into one 16-bit WAV, with a short
    pause between clips. Accepts float or integer WAVs, which matters because
    torchaudio writes 32-bit float WAVs that Python's wave module can't read."""
    if not clips:
        raise ValueError("No audio clips to join.")

    pieces = []
    sample_rate = None
    for clip in clips:
        data, rate = sf.read(io.BytesIO(clip), dtype="float32")
        if sample_rate is None:
            sample_rate = rate
        elif rate != sample_rate:
            raise ValueError(f"Clips have different sample rates ({sample_rate} vs {rate}).")
        if pieces:
            silence_shape = (int(sample_rate * pause_seconds),) + data.shape[1:]
            pieces.append(np.zeros(silence_shape, dtype="float32"))
        pieces.append(data)

    buffer = io.BytesIO()
    sf.write(buffer, np.concatenate(pieces), sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()

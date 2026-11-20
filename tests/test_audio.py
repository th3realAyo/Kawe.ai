import io
import unittest

import numpy as np
import soundfile as sf

from kawe.audio import join_wav_clips


def make_clip(seconds: float, rate: int = 24000, subtype: str = "FLOAT") -> bytes:
    buffer = io.BytesIO()
    samples = np.full(int(seconds * rate), 0.1, dtype="float32")
    sf.write(buffer, samples, rate, format="WAV", subtype=subtype)
    return buffer.getvalue()


class JoinWavClipsTests(unittest.TestCase):
    def test_joins_float_clips_with_pause(self):
        joined = join_wav_clips([make_clip(1.0), make_clip(2.0)], pause_seconds=0.5)
        data, rate = sf.read(io.BytesIO(joined))
        self.assertEqual(rate, 24000)
        self.assertEqual(len(data), int(3.5 * 24000))

    def test_single_clip_is_kept(self):
        data, _ = sf.read(io.BytesIO(join_wav_clips([make_clip(1.0, subtype="PCM_16")])))
        self.assertEqual(len(data), 24000)

    def test_rejects_mismatched_sample_rates(self):
        with self.assertRaises(ValueError):
            join_wav_clips([make_clip(1.0, rate=24000), make_clip(1.0, rate=16000)])

    def test_rejects_empty_list(self):
        with self.assertRaises(ValueError):
            join_wav_clips([])


if __name__ == "__main__":
    unittest.main()

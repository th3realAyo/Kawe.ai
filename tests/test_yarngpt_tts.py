"""YarnGPT client tests against a local fake API."""

import io
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import soundfile as sf


def wav_bytes(seconds=0.5, rate=24000):
    buffer = io.BytesIO()
    sf.write(buffer, np.full(int(rate * seconds), 0.1, dtype="float32"), rate, format="WAV")
    return buffer.getvalue()


class FakeYarnGPT(BaseHTTPRequestHandler):
    jobs = {}
    voices_payload = {"voices": [{"id": "idera", "languages": ["en", "yo"], "default": True}, {"id": "nonso", "languages": ["ig", "en"]}]}
    fail_next = False
    seen_auth_on_audio = []

    def log_message(self, *args):
        pass

    def _json(self, code, payload, headers=None):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/audio/"):
            FakeYarnGPT.seen_auth_on_audio.append(self.headers.get("Authorization"))
            fmt = self.path.rsplit(".", 1)[-1]
            data = wav_bytes() if fmt == "wav" else b"ID3fake-mp3-bytes"
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.headers.get("Authorization") != "Bearer test-key":
            return self._json(401, {"error": {"code": "INVALID_API_KEY", "user_message": "Bad key"}})
        if self.path == "/api/v1/voices":
            return self._json(200, FakeYarnGPT.voices_payload)
        if self.path.startswith("/api/v1/status/"):
            job_id = self.path.rsplit("/", 1)[-1]
            job = FakeYarnGPT.jobs[job_id]
            job["polls"] += 1
            if job["fail"]:
                return self._json(200, {"status": "failed", "user_message": "Voice unavailable"})
            if job["polls"] < 2:
                return self._json(200, {"status": "processing", "percentage": 50})
            port = self.server.server_port
            return self._json(200, {"status": "completed", "percentage": 100, "audio_url": f"http://127.0.0.1:{port}/audio/{job_id}.{job['fmt']}"})
        self._json(404, {"error": {"code": "NOT_FOUND"}})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert self.headers.get("Idempotency-Key"), "missing Idempotency-Key"
        assert len(body["text"]) <= 15000
        job_id = f"job{len(FakeYarnGPT.jobs)}"
        FakeYarnGPT.jobs[job_id] = {"polls": 0, "fmt": body["output_format"], "fail": FakeYarnGPT.fail_next, "voice": body["voice"]}
        warnings = [{"code": "VOICE_LANGUAGE_UNLISTED", "message": "Voice may not speak this language"}] if body["voice"] == "nonso" else []
        self._json(202, {"job_id": job_id, "status": "queued", "warnings": warnings})


class YarnGPTClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeYarnGPT)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        os.environ["YARNGPT_BASE_URL"] = f"http://127.0.0.1:{cls.server.server_port}"
        os.environ["YARNGPT_API_KEY"] = "test-key"
        os.environ["NO_PROXY"] = "127.0.0.1"
        from kawe import yarngpt_tts

        cls.yt = yarngpt_tts

    def setUp(self):
        FakeYarnGPT.fail_next = False

    def test_short_text_returns_mp3(self):
        audio, ext, warnings = self.yt.synthesize("Ẹ káàárọ̀. Báwo ni?", "idera", poll_seconds=0.01)
        self.assertEqual(ext, "mp3")
        self.assertTrue(audio.startswith(b"ID3"))
        self.assertEqual(warnings, [])
        self.assertIsNone(FakeYarnGPT.seen_auth_on_audio[-1], "API key must not go to the audio host")

    def test_long_text_is_split_and_joined_as_wav(self):
        text = "Eyi ni gbolohun kan ti o gun die. " * 900  # ~30,000 characters
        audio, ext, _ = self.yt.synthesize(text, "idera", poll_seconds=0.01)
        self.assertEqual(ext, "wav")
        data, rate = sf.read(io.BytesIO(audio))
        self.assertGreater(len(data) / rate, 1.0)  # 3 clips of 0.5 s plus pauses

    def test_warnings_are_returned(self):
        _, _, warnings = self.yt.synthesize("Hello there.", "nonso", poll_seconds=0.01)
        self.assertEqual(warnings, ["Voice may not speak this language"])

    def test_failed_job_raises_user_message(self):
        FakeYarnGPT.fail_next = True
        with self.assertRaisesRegex(self.yt.YarnGPTError, "Voice unavailable"):
            self.yt.synthesize("Hello again.", "idera", poll_seconds=0.01)

    def test_bad_key_raises(self):
        os.environ["YARNGPT_API_KEY"] = "wrong"
        try:
            with self.assertRaisesRegex(self.yt.YarnGPTError, "INVALID_API_KEY"):
                self.yt.synthesize("Hi.", "idera", poll_seconds=0.01)
        finally:
            os.environ["YARNGPT_API_KEY"] = "test-key"

    def test_voices_filtered_by_language(self):
        self.assertEqual(self.yt.voices_for_language("Yoruba"), ["idera"])
        self.assertEqual(self.yt.voices_for_language("Igbo"), ["nonso"])
        # No voice lists Hausa -> show all rather than nothing
        self.assertEqual(self.yt.voices_for_language("Hausa"), ["idera", "nonso"])

    def test_voice_list_without_language_info_returns_all(self):
        old = FakeYarnGPT.voices_payload
        FakeYarnGPT.voices_payload = ["umar", "wura"]
        try:
            self.assertEqual(self.yt.voices_for_language("Yoruba"), ["umar", "wura"])
        finally:
            FakeYarnGPT.voices_payload = old

    def test_fallback_voices_when_api_unreachable(self):
        os.environ["YARNGPT_API_KEY"] = "wrong"
        try:
            self.assertEqual(self.yt.voices_for_language("Yoruba"), self.yt.FALLBACK_VOICES)
        finally:
            os.environ["YARNGPT_API_KEY"] = "test-key"

    def test_split_text_respects_limit(self):
        chunks = self.yt.split_text("Short sentence. " * 2000, max_chars=1000)
        self.assertTrue(all(len(c) <= 1000 for c in chunks))
        self.assertEqual(" ".join(chunks), " ".join(("Short sentence. " * 2000).split()))


if __name__ == "__main__":
    unittest.main()

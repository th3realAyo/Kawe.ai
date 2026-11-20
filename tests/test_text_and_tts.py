import unittest
from unittest import mock

from kawe import elevenlabs_tts
from kawe.text import split_sentences, split_text


class TextTests(unittest.TestCase):
    def test_sentences_keep_closing_quotes(self):
        self.assertEqual(split_sentences('He said "stop." Then he left!  Why?'), ['He said "stop."', "Then he left!", "Why?"])

    def test_split_text_respects_limit_and_sentences(self):
        text = "Short sentence. " * 200
        pieces = split_text(text, 100)
        self.assertTrue(all(len(p) <= 100 for p in pieces))
        self.assertTrue(all(p.endswith(".") for p in pieces))
        self.assertEqual(" ".join(pieces), text.strip())

    def test_over_long_sentence_is_cut(self):
        pieces = split_text("a" * 250 + ". Next.", 100)
        self.assertEqual([len(p) for p in pieces], [100, 100, 57])


class ElevenLabsTests(unittest.TestCase):
    def test_long_text_is_sent_in_pieces_and_joined(self):
        client = mock.MagicMock()
        client.text_to_speech.convert.side_effect = lambda **kw: iter([b"mp3-", str(len(kw["text"])).encode()])
        text = "A sentence of moderate length. " * 600  # ~18,600 characters
        with mock.patch.dict("os.environ", {"ELEVENLABS_API_KEY": "k"}), mock.patch("elevenlabs.client.ElevenLabs", return_value=client):
            audio = elevenlabs_tts.synthesize_with_elevenlabs(text)
        calls = client.text_to_speech.convert.call_args_list
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(len(c.kwargs["text"]) <= elevenlabs_tts.CHUNK_CHARS for c in calls))
        self.assertEqual(audio.count(b"mp3-"), 3)

    def test_errors_are_wrapped(self):
        client = mock.MagicMock()
        client.text_to_speech.convert.side_effect = RuntimeError("quota")
        with mock.patch.dict("os.environ", {"ELEVENLABS_API_KEY": "k"}), mock.patch("elevenlabs.client.ElevenLabs", return_value=client):
            with self.assertRaises(elevenlabs_tts.ElevenLabsTTSError):
                elevenlabs_tts.synthesize_with_elevenlabs("Hello.")


if __name__ == "__main__":
    unittest.main()

"""English speech synthesis with ElevenLabs."""

import os 

VOICE_NAME_TO_ID = {
    "Daniel": "onwK4e9ZLuTAKqWW03F9",
    "Chris": "iP95p4xoKVk53GoZ742B",
    "Bella": "hpp4J3VqNfWAUOO0d1Us",
    "Lilly": "pFZP5JQG7iQjIQuC4Bku",
    "Liam": "TX3LPaxmHKxFdv7VOQHJ",
}
MODEL_ID = "eleven_multilingual_v2"
CHUNK_CHARS = 9_000  # the model accepts at most 10,000 characters per request


class ElevenLabsTTSError(Exception):
    pass


def synthesize_with_elevenlabs(text: str, voice_name: str = "Daniel") -> bytes:
    """MP3 audio; text over CHUNK_CHARS is synthesized in sentence-aligned pieces and joined."""
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise ElevenLabsTTSError("ELEVENLABS_API_KEY is not set.")
    voice_id = VOICE_NAME_TO_ID.get(voice_name)
    if not voice_id:
        raise ElevenLabsTTSError(f"Unknown voice '{voice_name}'. Options: {', '.join(VOICE_NAME_TO_ID)}")

    from elevenlabs.client import ElevenLabs

    from kawe.text import split_text

    pieces = split_text(text, CHUNK_CHARS)
    if not pieces:
        raise ElevenLabsTTSError("There is no text to speak.")
    audio = b""
    try:
        client = ElevenLabs(api_key=api_key)
        for piece in pieces:
            # MP3 is a sequence of independent frames, so same-format pieces can be concatenated.
            stream = client.text_to_speech.convert(text=piece, voice_id=voice_id, model_id=MODEL_ID, output_format="mp3_44100_128")
            audio += b"".join(stream)
    except Exception as e:
        raise ElevenLabsTTSError(str(e)) from e
    return audio

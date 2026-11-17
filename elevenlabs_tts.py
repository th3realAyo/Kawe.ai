import os

from elevenlabs.client import ElevenLabs


class ElevenLabsTTSError(Exception):
    """Raised when ElevenLabs synthesis fails for any reason."""
VOICE_NAME_TO_ID = {
    "Daniel": "onwK4e9ZLuTAKqWW03F9",
    "Chris": "iP95p4xoKVk53GoZ742B",
    "Bella": "hpp4J3VqNfWAUOO0d1Us",
    "Lilly": "pFZP5JQG7iQjIQuC4Bku",
    "Liam": "TX3LPaxmHKxFdv7VOQHJ",
}

def synthesize_with_elevenlabs(text: str, voice_name: str = "Rachel") -> bytes:
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise ElevenLabsTTSError("ELEVENLABS_API_KEY is not set.")

    voice_id = VOICE_NAME_TO_ID.get(voice_name)
    if not voice_id:
        raise ElevenLabsTTSError(
            f"Unknown voice name '{voice_name}'. Valid options: {list(VOICE_NAME_TO_ID)}"
        )

    try:
        client = ElevenLabs(api_key=api_key)
        audio_stream = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        return b"".join(audio_stream)
    except Exception as e:
        raise ElevenLabsTTSError(str(e)) from e

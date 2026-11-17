# Kàwé

**Read smart. Listen better.**

Kàwé turns long documents into clear, listenable summaries. Upload a document, choose how much of it you want to keep, select an audio language and voice, then listen or download the result.

It is designed for people who need to understand documents quickly without reading every page.

## What Kàwé does

- Extracts text from PDF and Word documents
- Produces concise, source-grounded summaries
- Lets users control the desired summary length
- Supports audio generation in selected languages
- Offers a choice of available TTS voices
- Lets users listen to or download generated audio

## Flow

1. Upload a document
2. Choose a summary length
3. Select a summary or audio language
4. Choose a voice
5. Generate, listen to, and download the audio summary

## Technology

Kàwé uses Python and Streamlit for the app experience, document extraction tools for PDF and Word files, semantic ranking and language models for summarization, and text-to-speech providers such as YarnGPT and ElevenLabs for audio generation.

## Local development

From the project folder:

```bash
python -m venv .venv
```

On Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run main.py
```

The current local interface is being expanded to include the full Kàwé workflow: configurable summary length, document language handling, and provider/voice selection.

## Configuration

Keep API credentials in environment variables or your deployment platform's secret manager. Do not commit them to the repository.

```env
# Modal endpoints (printed by `modal deploy modal_app.py`)
MODAL_EMBED_URL=
MODAL_SUMMARIZE_URL=
MODAL_TRANSLATE_URL=
RAG_AUTH_TOKEN=          # shared secret, must match the Modal secret

# Text-to-speech
ELEVENLABS_API_KEY=      # English voices
YARNGPT_API_KEY=         # Yoruba, Igbo and Hausa voices, and English backup (yarngpt.ai)

# User accounts and usage logging (Supabase)
SUPABASE_URL=
SUPABASE_SERVICE_KEY=
DB_ENCRYPTION_KEY=       # generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Create the database tables by running `db-schema.sql` in the Supabase SQL editor.

## Status

Kàwé is actively being developed. The product direction is a secure document-to-summary-to-audio experience with flexible language and voice choices.

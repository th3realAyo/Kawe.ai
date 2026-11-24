# Kàwé

Kàwé turns long PDF and Word documents into short summaries in which every section is linked to the passage it came from, and reads the summary aloud in English, Yoruba, Igbo or Hausa.

## How it works

1. **Extraction** (`kawe/extraction.py`): page text via PDFium, with running headers and page numbers removed. Word files (paragraphs and tables) are grouped into ~350-word sections that break at headings and are labelled with them, e.g. `Section 3 ("Methods")`. Scanned, image-only PDFs are not supported (no OCR).
2. **Ranking** (`kawe/ranking.py`): the text is split into passages of whole sentences (at most 180 words; short tails are merged), embedded with the multilingual BGE-M3, and scored by PageRank centrality on the cosine-similarity graph. Passages are then picked by maximal marginal relevance, so near-duplicate central passages are not all chosen.
3. **Grounded rewriting** (`kawe/summarizer.py`): each selected passage is rewritten on its own by Llama-3.1-8B-Instruct (bfloat16, greedy decoding) under a prompt that forbids content not in the passage; the passage is kept as the citation. All passages go to the GPU in one batched request.
4. **Translation and speech**: sections are translated one at a time by the same model; English is voiced by ElevenLabs, Yoruba, Igbo and Hausa by YarnGPT.

5. **Ask the document (prototype, `kawe/qa.py`)**: after a summary, a question in English, Yoruba, Igbo or Hausa is translated to English, matched against the summary's passage embeddings (top 3 above a similarity floor), and answered under a prompt that allows only what those passages say, citing them as [1], [2]. If nothing relevant is found, or the model replies NOT_FOUND, the app says "This document doesn't say." The answer is translated back and can be played aloud. Questions and answers are never stored; only the fact that a question was asked, and whether it was answered, is logged. The similarity floor (`MIN_SIMILARITY`) is a first guess still to be calibrated.

Model inference runs in one Modal GPU container (`modal_app.py`); the Streamlit app (`main.py`) calls its endpoints. Users sign in with an email address that is stored only as an irreversible keyed hash (see `kawe/db.py` and `policies/privacy-policy.md`).

## Project layout

```
main.py              Streamlit app
modal_app.py         GPU endpoints: embed, summarize, translate
kawe/                application package
  extraction.py      PDF and Word text extraction
  ranking.py         passage splitting and centrality ranking
  summarizer.py      grounded summarization pipeline
  modal_client.py    HTTP client for the Modal endpoints
  qa.py              ask-the-document prototype: retrieval and grounded answers
  limits.py          request limits and token checks enforced by the endpoints
  text.py            sentence splitting
  elevenlabs_tts.py  English speech
  yarngpt_tts.py     Yoruba, Igbo and Hausa speech
  audio.py           joining audio clips
  export.py          Word downloads
  metrics.py         upload limits and summary statistics
  accounts.py, db.py email sign-in, account deletion, usage logging (Supabase)
eval/                evaluation scripts (see eval/README.md)
tests/               unit tests
schema.sql           Supabase tables
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt   # app + test dependencies (requirements.txt is the app alone)
```

Deploy the inference endpoints (needs `pip install modal` and the Modal secrets `hf-token` with `HF_TOKEN`, and `rag-auth-token` with `RAG_AUTH_TOKEN` and, for the evaluation, `EVAL_AUTH_TOKEN`):

```powershell
modal deploy modal_app.py
```

The deploy prints five endpoint URLs. Create the database tables by running `schema.sql` in the Supabase SQL editor (it also enables row level security), then fill in `.env`:

```env
MODAL_EMBED_URL=             # .../inference-embed-endpoint...
MODAL_SUMMARIZE_URL=         # .../inference-summarize-endpoint...
MODAL_SUMMARIZE_BATCH_URL=   # .../inference-summarize-batch-endpoint...
MODAL_TRANSLATE_URL=         # .../inference-translate-endpoint...
MODAL_ANSWER_URL=            # .../inference-answer-endpoint... (ask-the-document prototype)
RAG_AUTH_TOKEN=
EVAL_AUTH_TOKEN=             # only needed to run the evaluation's whole-document baseline
ELEVENLABS_API_KEY=
YARNGPT_API_KEY=
SUPABASE_URL=
SUPABASE_SERVICE_KEY=
EMAIL_HASH_KEY=              # python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Keep `EMAIL_HASH_KEY` secret and never change it: it is what makes stored email hashes irreversible, and a new key means no existing user can sign in again.

Run the app:

```powershell
streamlit run main.py
```

## Tests

```powershell
python -m pytest tests
```

GitHub Actions runs the same tests on every push (`.github/workflows/tests.yml`).

## Evaluation

See `eval/README.md`.

## License

MIT; see `LICENSE`. The models it calls have their own licences: Llama 3.1 is under the Llama 3.1 Community License, BGE-M3 under MIT, and ElevenLabs and YarnGPT under their providers' terms.

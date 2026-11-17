"""
modal_app.py

Modal application hosting the GPU-backed ML inference for the grounded
RAG summarization pipeline.

Exposes three authenticated HTTP endpoints, called by the Streamlit app
(hosted separately on a free Hugging Face Space):

    POST /embed        -> embeds a list of text chunks (BAAI/bge-base-en-v1.5)
    POST /summarize     -> grounded, per-chunk abstractive rewrite
                            (Llama-3.1-8B-Instruct, 4-bit quantized)

    POST /translate     -> summary translation, same Llama model

Speech synthesis is not hosted here: English uses ElevenLabs and
Yoruba/Igbo/Hausa use the hosted YarnGPT API (see yarngpt_tts.py).

Deploy with:
    modal deploy modal_app.py

Required Modal secrets (create once via `modal secret create ...`):
    hf-token        -> HF_TOKEN            (Hugging Face token with access
                                             to the gated meta-llama/Llama-3.1-8B-Instruct repo)
    rag-auth-token  -> RAG_AUTH_TOKEN      (shared secret the Streamlit app
                                             must send in the Authorization header)
"""

import os
from typing import List

import modal
from fastapi import Header

# --------------------------------------------------------------------------
# App + image definition
# --------------------------------------------------------------------------

app = modal.App("pdf-rag-summarizer")
rag_secret = modal.Secret.from_name("rag-auth-token")


# Persists downloaded model weights across container restarts. Without
# this, every cold start re-downloads Llama-3.1-8B (~16GB) from Hugging
# Face from scratch, which is what was causing every chunk to time out.
hf_cache_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)

# Single image shared by both GPU classes. Kept in one image for simplicity;
# split into two images later if cold-start size becomes an issue.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.44.2",
        "accelerate==0.33.0",
        "bitsandbytes==0.43.3",
        "sentence-transformers==3.0.1",
        "fastapi[standard]==0.115.0",
        "pydantic==2.9.2",
    )
    .env({"HF_HOME": "/cache/huggingface"})
)

# --------------------------------------------------------------------------
# Auth helper
# --------------------------------------------------------------------------
# Simple shared-secret check so the public Modal endpoint can't be called by
# anyone other than the Streamlit app. Not full OAuth-grade auth, but
# sufficient for a single-frontend internal API.


def _check_auth(authorization: str | None) -> None:
    from fastapi import HTTPException

    expected = os.environ["RAG_AUTH_TOKEN"]
    if not authorization or authorization.removeprefix("Bearer ") != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing auth token")


# --------------------------------------------------------------------------
# Request/response schemas
# --------------------------------------------------------------------------

from pydantic import BaseModel


class EmbedRequest(BaseModel):
    texts: List[str]


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]


class SummarizeRequest(BaseModel):
    text: str
    max_new_tokens: int = 120
    language: str = "English"  # e.g. "English", "French" — matches detected doc language


class SummarizeResponse(BaseModel):
    summary: str


class TranslateRequest(BaseModel):
    text: str
    target_language: str  # full name, e.g. "English", "Yoruba"
    max_new_tokens: int = 300


class TranslateResponse(BaseModel):
    translated_text: str


# --------------------------------------------------------------------------
# Embedding model: BAAI/bge-base-en-v1.5
# --------------------------------------------------------------------------


@app.cls(
    image=image,
    gpu="A10G",
    scaledown_window=120,
    volumes={"/cache": hf_cache_volume},
    timeout=600,
    secrets=[rag_secret]
)
class Embedder:
    @modal.enter()
    def load_model(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")

    @modal.method()
    def embed(self, texts: List[str]) -> List[List[float]]:
        # bge models recommend a query/passage prefix convention; for pure
        # passage-importance scoring (no separate query), embedding the raw
        # passage text directly is the right call here.
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    @modal.fastapi_endpoint(method="POST")
    def embed_endpoint(self, request: EmbedRequest, authorization: str = Header(None)):
        _check_auth(authorization)
        vectors = self.embed.local(request.texts)
        return EmbedResponse(embeddings=vectors)


# --------------------------------------------------------------------------
# Abstractive rewriting model: Llama-3.1-8B-Instruct (4-bit quantized)
# --------------------------------------------------------------------------

GROUNDING_SYSTEM_PROMPT = (
    "You are a precise summarization assistant. You will be given a single "
    "passage extracted from a larger document. Summarize ONLY the "
    "information contained in that passage, in 1-3 concise sentences. "
    "Do not add facts, context, or interpretation that is not explicitly "
    "present in the passage. Do not speculate. Respond in {language}."
)

# Reuses the same loaded Llama-3.1-8B-Instruct model as summarization —
# translation is just a different prompt, not a different model, so this
# adds zero extra GPU memory or cold-start cost on top of the Summarizer
# container that's already running.
TRANSLATION_SYSTEM_PROMPT = (
    "You are a precise translation assistant. Translate the following text "
    "into {language}. Preserve the original meaning exactly — do not "
    "summarize, expand, or add commentary. Respond with ONLY the "
    "translated text, nothing else."
)


@app.cls(
    image=image,
    gpu="A10G",
    scaledown_window=120,
    secrets=[modal.Secret.from_name("hf-token"), rag_secret],
    volumes={"/cache": hf_cache_volume},
    timeout=900,  # generous headroom for a cold first-ever download of ~16GB
)
class Summarizer:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        model_id = "meta-llama/Llama-3.1-8B-Instruct"
        hf_token = os.environ["HF_TOKEN"]

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
            token=hf_token,
        )

    @modal.method()
    def summarize(self, text: str, max_new_tokens: int = 120, language: str = "English") -> str:
        import torch

        messages = [
            {"role": "system", "content": GROUNDING_SYSTEM_PROMPT.format(language=language)},
            {"role": "user", "content": text},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # deterministic, favors grounding over creativity
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    @modal.fastapi_endpoint(method="POST")
    def summarize_endpoint(self, request: SummarizeRequest, authorization: str = Header(None)):
        _check_auth(authorization)
        summary = self.summarize.local(
            request.text, request.max_new_tokens, request.language
        )
        return SummarizeResponse(summary=summary)

    @modal.method()
    def translate(self, text: str, target_language: str, max_new_tokens: int = 300) -> str:
        import torch

        messages = [
            {
                "role": "system",
                "content": TRANSLATION_SYSTEM_PROMPT.format(language=target_language),
            },
            {"role": "user", "content": text},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    @modal.fastapi_endpoint(method="POST")
    def translate_endpoint(self, request: TranslateRequest, authorization: str = Header(None)):
        _check_auth(authorization)
        translated = self.translate.local(
            request.text, request.target_language, request.max_new_tokens
        )
        return TranslateResponse(translated_text=translated)


# Yoruba, Igbo and Hausa speech now comes from the hosted YarnGPT API
# (yarngpt_tts.py), so no speech model is deployed here.

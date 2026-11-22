"""GPU inference endpoints for Kàwé, deployed with `modal deploy modal_app.py`.

One container on one A10G serves every endpoint, so a document pays for one cold start:

POST embed_endpoint            bge-m3 passage embeddings (multilingual)
POST summarize_endpoint        passage-constrained rewrite of one passage
POST summarize_batch_endpoint  the same rewrite for many passages, generated as a batch
POST translate_endpoint        translation
POST answer_endpoint           grounded answer to a question from given passages (prototype)

The language model is Llama-3.1-8B-Instruct in bfloat16 (~16 GB, fits the A10G's 24 GB).

Required Modal secrets:
  hf-token        HF_TOKEN, access to the gated Llama repo
  rag-auth-token  RAG_AUTH_TOKEN, the bearer token the app sends; optionally also
                  EVAL_AUTH_TOKEN, a second token that alone may replace the system prompt
                  and send whole documents (used by the evaluation's baselines)
"""

import os
from typing import List, Optional

import modal
from fastapi import Header
from pydantic import BaseModel

EMBED_MODEL = "BAAI/bge-m3"
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
GENERATION_BATCH_SIZE = 8

GROUNDING_SYSTEM_PROMPT = (
    "You are a precise summarization assistant. You will be given a single "
    "passage extracted from a larger document. Summarize ONLY the "
    "information contained in that passage, in 1-3 concise sentences. "
    "Do not add facts, context, or interpretation that is not explicitly "
    "present in the passage. Do not speculate. Respond in {language}."
)
TRANSLATION_SYSTEM_PROMPT = (
    "You are a precise translation assistant. Translate the following text "
    "into {language}. Preserve the original meaning exactly — do not "
    "summarize, expand, or add commentary. Respond with ONLY the "
    "translated text, nothing else."
)

ANSWER_SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered passages below, which come from one document. "
    "If the passages do not contain the answer, reply with exactly NOT_FOUND and nothing else. "
    "Otherwise answer in 1-3 short, plain English sentences and put the number of each passage "
    "you used in square brackets after the sentence it supports, for example [2]. "
    "Do not use outside knowledge, do not guess, and do not mention 'the passages'."
)

app = modal.App("kawe")
hf_cache_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)  # keeps weights across cold starts

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.44.2",
        "accelerate==0.33.0",
        "sentence-transformers==3.0.1",
        "fastapi[standard]==0.115.0",
        "pydantic==2.9.2",
    )
    .env({"HF_HOME": "/cache/huggingface"})
    .add_local_python_source("kawe")  # for kawe.limits; the package's other modules are imported lazily
)


class EmbedRequest(BaseModel):
    texts: List[str]


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]


class SummarizeRequest(BaseModel):
    text: str
    max_new_tokens: int = 120
    language: str = "English"
    instruction: Optional[str] = None  # replaces the grounding prompt; evaluation token only


class SummarizeResponse(BaseModel):
    summary: str
    instruction_used: bool = False


class SummarizeBatchRequest(BaseModel):
    texts: List[str]
    max_new_tokens: int = 120
    language: str = "English"


class SummarizeBatchResponse(BaseModel):
    summaries: List[str]


class AnswerRequest(BaseModel):
    question: str
    passages: List[str]
    max_new_tokens: int = 200


class AnswerResponse(BaseModel):
    answer: str


class TranslateRequest(BaseModel):
    text: str
    target_language: str
    max_new_tokens: int = 512


class TranslateResponse(BaseModel):
    translated_text: str


def _authorize(authorization: Optional[str]) -> str:
    from fastapi import HTTPException

    from kawe.limits import RequestRejected, caller_role

    try:
        return caller_role(authorization, os.environ.get("RAG_AUTH_TOKEN"), os.environ.get("EVAL_AUTH_TOKEN"))
    except RequestRejected as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


def _validate(check, *args) -> None:
    from fastapi import HTTPException

    from kawe.limits import RequestRejected

    try:
        check(*args)
    except RequestRejected as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


@app.cls(
    image=image,
    gpu="A10G",
    scaledown_window=300,
    secrets=[modal.Secret.from_name("hf-token"), modal.Secret.from_name("rag-auth-token")],
    volumes={"/cache": hf_cache_volume},
    timeout=900,
)
class Inference:
    @modal.enter()
    def load_models(self):
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.embedder = SentenceTransformer(EMBED_MODEL, device="cuda", model_kwargs={"torch_dtype": torch.float16})
        self.embedder.max_seq_length = 512  # passages are at most ~300 tokens

        token = os.environ["HF_TOKEN"]
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, token=token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # decoder-only batched generation
        self.model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, torch_dtype=torch.bfloat16, device_map="cuda", token=token)
        self.model.eval()

    def _generate(self, system_prompt: str, texts: List[str], max_new_tokens: int) -> List[str]:
        """Greedy decoding with the chat template, in batches of GENERATION_BATCH_SIZE."""
        import torch

        outputs: List[str] = []
        for start in range(0, len(texts), GENERATION_BATCH_SIZE):
            prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for text in texts[start : start + GENERATION_BATCH_SIZE]
            ]
            # The chat template already starts with <|begin_of_text|>.
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            prompt_length = inputs["input_ids"].shape[-1]
            outputs += [self.tokenizer.decode(ids[prompt_length:], skip_special_tokens=True).strip() for ids in output_ids]
        return outputs

    @modal.method()
    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.encode(texts, batch_size=32, normalize_embeddings=True).tolist()

    @modal.method()
    def summarize(self, text: str, max_new_tokens: int = 120, language: str = "English", instruction: Optional[str] = None) -> str:
        system_prompt = instruction or GROUNDING_SYSTEM_PROMPT.format(language=language)
        return self._generate(system_prompt, [text], max_new_tokens)[0]

    @modal.method()
    def summarize_batch(self, texts: List[str], max_new_tokens: int = 120, language: str = "English") -> List[str]:
        return self._generate(GROUNDING_SYSTEM_PROMPT.format(language=language), texts, max_new_tokens)

    @modal.method()
    def translate(self, text: str, target_language: str, max_new_tokens: int = 512) -> str:
        return self._generate(TRANSLATION_SYSTEM_PROMPT.format(language=target_language), [text], max_new_tokens)[0]

    @modal.method()
    def answer(self, question: str, passages: List[str], max_new_tokens: int = 200) -> str:
        numbered = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1))
        prompt = f"Passages:\n{numbered}\n\nQuestion: {question}"
        return self._generate(ANSWER_SYSTEM_PROMPT, [prompt], max_new_tokens)[0]

    @modal.fastapi_endpoint(method="POST")
    def answer_endpoint(self, request: AnswerRequest, authorization: str = Header(None)):
        from kawe.limits import check_answer

        _authorize(authorization)
        _validate(check_answer, request.question, request.passages, request.max_new_tokens)
        return AnswerResponse(answer=self.answer.local(request.question, request.passages, request.max_new_tokens))

    @modal.fastapi_endpoint(method="POST")
    def embed_endpoint(self, request: EmbedRequest, authorization: str = Header(None)):
        from kawe.limits import check_embed

        _authorize(authorization)
        _validate(check_embed, request.texts)
        return EmbedResponse(embeddings=self.embed.local(request.texts))

    @modal.fastapi_endpoint(method="POST")
    def summarize_endpoint(self, request: SummarizeRequest, authorization: str = Header(None)):
        from kawe.limits import check_summarize

        role = _authorize(authorization)
        _validate(check_summarize, request.text, request.max_new_tokens, request.language, request.instruction, role)
        summary = self.summarize.local(request.text, request.max_new_tokens, request.language, request.instruction)
        return SummarizeResponse(summary=summary, instruction_used=bool(request.instruction))

    @modal.fastapi_endpoint(method="POST")
    def summarize_batch_endpoint(self, request: SummarizeBatchRequest, authorization: str = Header(None)):
        from kawe.limits import check_summarize_batch

        _authorize(authorization)
        _validate(check_summarize_batch, request.texts, request.max_new_tokens, request.language)
        return SummarizeBatchResponse(summaries=self.summarize_batch.local(request.texts, request.max_new_tokens, request.language))

    @modal.fastapi_endpoint(method="POST")
    def translate_endpoint(self, request: TranslateRequest, authorization: str = Header(None)):
        from kawe.limits import check_translate

        _authorize(authorization)
        _validate(check_translate, request.text, request.target_language, request.max_new_tokens)
        return TranslateResponse(translated_text=self.translate.local(request.text, request.target_language, request.max_new_tokens))

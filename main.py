"""Streamlit front end: upload, grounded summary, translation and audio."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from typing import List, Optional

import requests
import streamlit as st
from dotenv import load_dotenv
from langdetect import DetectorFactory, detect

from kawe.accounts import account_sidebar, log_document_outcome, require_signup
from kawe.elevenlabs_tts import (
    VOICE_NAME_TO_ID,
    ElevenLabsTTSError,
    synthesize_with_elevenlabs,
)
from kawe.export import build_summary_docx, build_translation_docx
from kawe.extraction import (
    DocumentExtractionError,
    UnsupportedFileTypeError,
    extract_document,
)
from kawe.metrics import (
    TOP_N_MAX,
    build_summary_metrics,
    summary_top_n,
    validate_upload_size,
)
from kawe.modal_client import ModalServiceError, translate
from kawe.ranking import split_into_chunks
from kawe.summarizer import (
    LANGUAGE_NAMES,
    GroundedSummarizerError,
    generate_grounded_summary,
)
from kawe.yarngpt_tts import YarnGPTError, voices_for_language
from kawe.yarngpt_tts import synthesize as synthesize_with_yarngpt

load_dotenv()
DetectorFactory.seed = 0  # langdetect is random without a fixed seed
log = logging.getLogger(__name__)

AUDIO_LANGUAGES = ["English", "Yoruba", "Igbo", "Hausa"]
ELEVENLABS_VOICES = list(VOICE_NAME_TO_ID)
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
SESSION_KEYS = ["segments", "chunk_count", "language_code", "summary_result", "summary_id", "confirmed", "audio", "translation"]

@dataclass
class DocumentContext:
    user_id: str
    file_name: str
    document_type: str
    length_words: int
    percentage: int
    language_name: str

    @property
    def stem(self) -> str:
        return os.path.splitext(self.file_name)[0]

    def log(self, success: bool, error_type: Optional[str] = None, **fields) -> None:
        """error_type is a short category; error details go to the server log, not the database."""
        log_document_outcome(
            self.user_id,
            document_type=self.document_type,
            document_length_words=self.length_words,
            summarization_percentage=self.percentage,
            processing_success=success,
            error_type=error_type,
            **fields,
        )


@st.cache_data(ttl=3600, show_spinner=False)
def yarngpt_voice_options(language: str) -> List[str]:
    return voices_for_language(language)


def english_backup_voice() -> str:
    voices = yarngpt_voice_options("English")
    return "idera" if "idera" in voices else voices[0]


def detect_language(segments) -> str:
    """Language of a sample taken from the start, middle and end of the document."""
    picks = sorted({0, len(segments) // 4, len(segments) // 2, 3 * len(segments) // 4, len(segments) - 1})
    sample = " ".join(" ".join(segments[i].text.split()[:200]) for i in picks)
    try:
        return detect(sample)
    except Exception:
        return "en"


def extract_uploaded(uploaded_file):
    """Extracts the upload once per file; returns (segments, language_code) or None on failure."""
    file_id = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    if st.session_state.get("processed_file_id") != file_id:
        for key in SESSION_KEYS:
            st.session_state.pop(key, None)
        st.session_state.processed_file_id = file_id

    if "segments" not in st.session_state:
        with st.spinner("Extracting document text..."):
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
            try:
                segments = extract_document(tmp.name)
            except (UnsupportedFileTypeError, DocumentExtractionError) as e:
                st.error(str(e))
                return None
            finally:
                os.unlink(tmp.name)
        st.session_state.segments = segments
        st.session_state.chunk_count = len(split_into_chunks(segments))
        st.session_state.language_code = detect_language(segments)
    return st.session_state.segments, st.session_state.language_code


def summarize_step(ctx: DocumentContext, segments, language_code: str, top_n: int) -> None:
    with st.spinner("Ranking passages and writing the grounded summary (the first run can take a minute)..."):
        try:
            result = generate_grounded_summary(segments, language_code=language_code, top_n=top_n)
        except GroundedSummarizerError as e:
            st.error(f"Summarization failed: {e}")
            log.warning("Summarization failed: %s", e)
            ctx.log(False, "summarization")
            return
    ctx.log(True, summary_language=ctx.language_name)
    st.session_state.summary_result = result
    st.session_state.summary_id = uuid.uuid4().hex
    st.session_state.confirmed = False


def review_step(ctx: DocumentContext, result) -> None:
    st.subheader("2. Review your grounded summary")
    if result.skipped_chunks:
        st.caption(f"Note: {result.skipped_chunks} section(s) couldn't be summarized and were skipped.")
    st.write(result.full_text)

    metrics = build_summary_metrics(ctx.length_words, result.full_text)
    columns = st.columns(3)
    columns[0].metric("Source length", f"{metrics.source_words:,} words")
    columns[1].metric("Summary length", f"{metrics.summary_words:,} words")
    columns[2].metric(
        "Estimated listening time",
        f"{metrics.estimated_listening_minutes} min" if metrics.estimated_listening_minutes else "Under 1 min",
        help="Estimated at roughly 150 words per minute.",
    )
    st.caption(f"This summary is about {metrics.compression_percentage}% of the source document.")

    st.markdown("**Citations — see exactly where each part came from:**")
    for i, section in enumerate(result.sections, start=1):
        with st.expander(f"Section {i} — source: {section.location_label}"):
            st.markdown(f"**Summary:** {section.summary_text}")
            st.markdown("**Original excerpt:**")
            st.caption(section.source_excerpt)

    st.download_button(
        "Download summary (Word)",
        data=build_summary_docx(ctx.file_name, ctx.language_name, result),
        file_name=f"summary_{ctx.stem}.docx",
        mime=DOCX_MIME,
    )


def get_translation(ctx: DocumentContext, result, language: str, key, log_fields: dict) -> Optional[List[str]]:
    """Translated summary sections, computed once per summary and language."""
    cached = st.session_state.get("translation")
    if cached and cached["key"] == key:
        return cached["sections"]
    texts = [s.summary_text for s in result.sections if s.summary_text.strip()]
    with st.spinner(f"Translating summary into {language}..."):
        try:
            # One request per section: a single request would stop at max_new_tokens.
            translated = [translate(text, language) for text in texts]
        except ModalServiceError as e:
            st.error(f"Translation failed: {e}")
            log.warning("Translation failed: %s", e)
            ctx.log(False, "translation", **log_fields)
            return None
    st.session_state.translation = {"key": key, "sections": translated}
    return translated


def synthesize(text: str, language: str, voice: str):
    """Returns (audio_bytes, extension, warnings)."""
    if language != "English":
        return synthesize_with_yarngpt(text, voice)
    try:
        return synthesize_with_elevenlabs(text, voice_name=voice), "mp3", []
    except ElevenLabsTTSError as e:
        backup = english_backup_voice()
        st.warning(f"ElevenLabs failed ({e}). Using the YarnGPT voice '{backup}' instead.")
        return synthesize_with_yarngpt(text, backup)


def listen_step(ctx: DocumentContext, result) -> None:
    st.subheader("4. Translate and listen")
    language = st.selectbox("Language", AUDIO_LANGUAGES)
    needs_translation = language.lower() != ctx.language_name.lower()
    voice = st.selectbox("Voice", ELEVENLABS_VOICES if language == "English" else yarngpt_voice_options(language))

    summary_id = st.session_state.get("summary_id")
    translation_key = (summary_id, language)
    audio_key = (summary_id, language, voice)
    audio_fields = dict(summary_language=ctx.language_name, audio_language=language, voice_selected=voice)

    col_translate, col_audio = st.columns(2)
    translate_clicked = needs_translation and col_translate.button(f"Translate summary to {language}")
    audio_clicked = col_audio.button("Generate audio")

    if translate_clicked:
        get_translation(ctx, result, language, translation_key, audio_fields)

    if audio_clicked:
        st.session_state.pop("audio", None)
        if needs_translation:
            texts = get_translation(ctx, result, language, translation_key, audio_fields)
            if texts is None:
                return
        else:
            texts = [s.summary_text for s in result.sections if s.summary_text.strip()]
        with st.spinner("Generating audio... this can take a minute for long summaries."):
            try:
                audio_bytes, ext, warnings = synthesize(" ".join(texts), language, voice)
            except (YarnGPTError, requests.RequestException, ValueError) as e:
                st.error(f"Audio generation failed: {e}")
                log.warning("Audio generation failed: %s", e)
                ctx.log(False, "tts", **audio_fields)
                return
        for warning in warnings:
            st.info(f"YarnGPT note: {warning}")
        ctx.log(True, audio_generated=True, **audio_fields)
        st.session_state.audio = {
            "key": audio_key,
            "bytes": audio_bytes,
            "format": "audio/mpeg" if ext == "mp3" else "audio/wav",
            "ext": ext,
        }

    translation = st.session_state.get("translation")
    if needs_translation and translation and translation["key"] == translation_key:
        st.markdown(f"**Summary in {language}:**")
        st.write(" ".join(translation["sections"]))
        st.caption("Machine translation. Check important details against the original document.")
        st.download_button(
            f"Download {language} summary (Word)",
            data=build_translation_docx(ctx.file_name, language, result, translation["sections"]),
            file_name=f"summary_{language.lower()}_{ctx.stem}.docx",
            mime=DOCX_MIME,
        )

    audio = st.session_state.get("audio")
    if audio and audio["key"] == audio_key:
        st.success("Audio ready!")
        st.audio(audio["bytes"], format=audio["format"])
        st.download_button(
            "Download audio",
            data=audio["bytes"],
            file_name=f"summary_{language.lower()}.{audio['ext']}",
            mime=audio["format"],
        )


def main() -> None:
    st.title("Kàwé.ai")
    st.caption("Read smart. Listen better. Every summary section is traceable to its source.")
    user_id = require_signup()
    account_sidebar(user_id)

    uploaded_file = st.file_uploader("Upload a PDF or Word document", type=["pdf", "docx"])
    if uploaded_file is None:
        return
    try:
        validate_upload_size(uploaded_file.size)
    except ValueError as error:
        st.error(str(error))
        return

    extracted = extract_uploaded(uploaded_file)
    if extracted is None:
        return
    segments, language_code = extracted
    language_name = LANGUAGE_NAMES.get(language_code, "English")
    st.success(f"Extracted {len(segments)} section(s). Detected document language: {language_name}")

    st.subheader("1. How much of the document should be summarized?")
    percentage = st.slider("Percentage of the document's passages to summarize", min_value=10, max_value=80, value=30, step=10)
    chunk_count = st.session_state.chunk_count
    top_n = summary_top_n(percentage, chunk_count)
    note = f" (the most a summary can use is {TOP_N_MAX})" if top_n == TOP_N_MAX and percentage / 100 * chunk_count > TOP_N_MAX else ""
    st.caption(
        f"The document has {chunk_count} passage(s); {min(top_n, chunk_count)} will be rewritten into the summary{note}. "
        "Each becomes 1–3 sentences, so the summary is much shorter than this percentage of the text."
    )
    ctx = DocumentContext(
        user_id=user_id,
        file_name=uploaded_file.name,
        document_type=os.path.splitext(uploaded_file.name)[1].lstrip("."),
        length_words=sum(len(seg.text.split()) for seg in segments),
        percentage=percentage,
        language_name=language_name,
    )

    if st.button("Generate grounded summary"):
        summarize_step(ctx, segments, language_code, top_n)
    if "summary_result" not in st.session_state:
        return
    result = st.session_state.summary_result
    review_step(ctx, result)

    st.subheader("3. Happy with this summary?")
    confirmed = st.checkbox(
        "Yes, I'm satisfied with this summary — continue to translation and audio",
        value=st.session_state.get("confirmed", False),
    )
    st.session_state.confirmed = confirmed
    if not confirmed:
        st.info("Review the summary above, or adjust the % and regenerate, before moving on.")
        return
    listen_step(ctx, result)


if __name__ == "__main__":
    main()

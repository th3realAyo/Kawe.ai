import base64
import os
import tempfile

import requests
import streamlit as st
from dotenv import load_dotenv
from langdetect import detect

from doc_loader import (
    DocumentExtractionError,
    UnsupportedFileTypeError,
    extract_document,
)
from docx_export import build_summary_docx
from elevenlabs_tts import ElevenLabsTTSError, synthesize_with_elevenlabs
from grounded_summarizer import (
    LANGUAGE_NAMES,
    GroundedSummarizerError,
    generate_grounded_summary,
)
from main_integration_snippet import log_document_outcome, require_signup

load_dotenv()

MODAL_TRANSLATE_URL = os.environ.get("MODAL_TRANSLATE_URL")
MODAL_YARNGPT_URL = os.environ.get("MODAL_YARNGPT_URL")
RAG_AUTH_TOKEN = os.environ.get("RAG_AUTH_TOKEN")

SUPPORTED_AUDIO_LANGUAGES = ["English", "Yoruba", "Igbo", "Hausa"]

YARNGPT_VOICES = {
    "yoruba": ["yoruba_male2", "yoruba_female2", "yoruba_feamle1"],
    "igbo": ["igbo_female2", "igbo_male2", "igbo_female1"],
    "hausa": ["hausa_feamle1", "hausa_female2", "hausa_male2", "hausa_male1"],
}

ELEVENLABS_VOICES = ["Daniel", "Chris", "Bella", "Lilly", "Liam"]

TOP_N_MIN, TOP_N_MAX = 3, 15


def save_uploaded_file(uploaded_file) -> str:
    """Streamlit's UploadedFile is file-like but doc_loader.py works off a
    path, so write it to a temp file first."""
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


def detect_document_language(sample_text: str) -> str:
    try:
        return detect(sample_text[:2000])
    except Exception:
        return "en"


def call_modal_translate(text: str, target_language: str) -> str:
    response = requests.post(
        MODAL_TRANSLATE_URL,
        json={"text": text, "target_language": target_language},
        headers={"Authorization": f"Bearer {RAG_AUTH_TOKEN}"},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["translated_text"]


def call_modal_yarngpt(text: str, language: str, speaker_name: str) -> bytes:
    response = requests.post(
        MODAL_YARNGPT_URL,
        json={"text": text, "language": language, "speaker_name": speaker_name},
        headers={"Authorization": f"Bearer {RAG_AUTH_TOKEN}"},
        timeout=600,
    )
    response.raise_for_status()
    audio_b64 = response.json()["audio_base64"]
    return base64.b64decode(audio_b64)


def render_citations(sections):
    for i, section in enumerate(sections, start=1):
        with st.expander(f"Section {i} — source: {section.location_label}"):
            st.markdown(f"**Summary:** {section.summary_text}")
            st.markdown("**Original excerpt:**")
            st.caption(section.source_excerpt)


def reset_downstream_state():
    for key in ["segments", "language_code", "summary_result", "confirmed"]:
        st.session_state.pop(key, None)


def main():  # sourcery skip: low-code-quality
    st.title("Grounded PDF/Word Summarizer")
    st.caption("Every summary section is traceable to the exact passage it came from.")

    user_id = require_signup()

    uploaded_file = st.file_uploader("Upload a PDF or Word document", type=["pdf", "docx"])

    if uploaded_file is None:
        return

    # New file uploaded -> wipe any previous document's state
    if st.session_state.get("processed_filename") != uploaded_file.name:
        reset_downstream_state()
        st.session_state.processed_filename = uploaded_file.name

    if "segments" not in st.session_state:
        with st.spinner("Extracting document text..."):
            temp_path = save_uploaded_file(uploaded_file)
            try:
                segments = extract_document(temp_path)
            except (UnsupportedFileTypeError, DocumentExtractionError) as e:
                st.error(str(e))
                return
            finally:
                os.unlink(temp_path)

            sample_text = " ".join(seg.text for seg in segments[:3])
            language_code = detect_document_language(sample_text)

        st.session_state.segments = segments
        st.session_state.language_code = language_code

    segments = st.session_state.segments
    language_code = st.session_state.language_code
    language_name = LANGUAGE_NAMES.get(language_code, "English")

    st.success(f"Extracted {len(segments)} section(s). Detected document language: {language_name}")

    st.subheader("1. How much of the document should be summarized?")
    percentage = st.slider(
        "Percentage of docum5ent to summarize", min_value=10, max_value=80, value=30, step=10
    )

    if st.button("Generate grounded summary"):
        approx_chunks = round((percentage / 100) * len(segments) * 2)
        top_n = max(TOP_N_MIN, min(TOP_N_MAX, approx_chunks))
        document_type = os.path.splitext(uploaded_file.name)[1].lstrip(".")
        document_length_words = sum(len(seg.text.split()) for seg in segments)

        with st.spinner(
            "Ranking passages and generating the grounded summary... "
            "first run can take a minute (model cold start on Modal)."
        ):
            try:
                result = generate_grounded_summary(
                    segments, language_code=language_code, top_n=top_n
                )
            except GroundedSummarizerError as e:
                st.error(f"Summarization failed: {e}")
                log_document_outcome(
                    user_id,
                    document_type=document_type,
                    document_length_words=document_length_words,
                    summarization_percentage=percentage,
                    processing_success=False,
                    error_type=str(e),
                )
                return
            except requests.RequestException as e:
                st.error(f"Could not reach the summarization service: {e}")
                log_document_outcome(
                    user_id,
                    document_type=document_type,
                    document_length_words=document_length_words,
                    summarization_percentage=percentage,
                    processing_success=False,
                    error_type=str(e),
                )
                return

        log_document_outcome(
            user_id,
            document_type=document_type,
            document_length_words=document_length_words,
            summarization_percentage=percentage,
            processing_success=True,
            summary_language=language_name,
        )

        st.session_state.summary_result = result
        st.session_state.confirmed = False

    if "summary_result" not in st.session_state:
        return

    result = st.session_state.summary_result

    st.subheader("2. Review your grounded summary")
    if result.skipped_chunks:
        st.caption(f"Note: {result.skipped_chunks} section(s) couldn't be summarized and were skipped.")

    st.write(result.full_text)

    st.markdown("**Citations — see exactly where each part came from:**")
    render_citations(result.sections)

    st.download_button(
        "Download summary (Word)",
        data=build_summary_docx(uploaded_file.name, language_name, result),
        file_name=f"summary_{os.path.splitext(uploaded_file.name)[0]}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
        
    st.subheader("3. Happy with this summary?")
    confirmed = st.checkbox(
        "Yes, I'm satisfied with this summary — let's make the audio",
        value=st.session_state.get("confirmed", False),
    )
    st.session_state.confirmed = confirmed

    if not confirmed:
        st.info("Review the summary above, or adjust the % and regenerate, before moving to audio.")
        return

    st.subheader("4. Audio options")
    audio_language = st.selectbox("Audio language", SUPPORTED_AUDIO_LANGUAGES)

    if audio_language == "English":
        voice_options = ELEVENLABS_VOICES
    else:
        voice_options = YARNGPT_VOICES[audio_language.lower()]

    voice = st.selectbox("Voice", voice_options)

    if st.button("Generate audio"):
        text_for_audio = result.full_text

        if audio_language.lower() != language_name.lower():
            with st.spinner(f"Translating summary into {audio_language}..."):
                try:
                    text_for_audio = call_modal_translate(result.full_text, audio_language)
                except requests.RequestException as e:
                    st.error(f"Translation failed: {e}")
                    return

        with st.spinner("Synthesizing audio... first run can take a while (model cold start)."):
            try:
                if audio_language == "English":
                    audio_bytes = synthesize_with_elevenlabs(text_for_audio, voice_name=voice)
                    audio_format = "audio/mpeg"
                    file_ext = "mp3"
                else:
                    audio_bytes = call_modal_yarngpt(
                        text_for_audio, audio_language.lower(), voice
                    )
                    audio_format = "audio/wav"
                    file_ext = "wav"
            except ElevenLabsTTSError as e:
                st.error(
                    f"ElevenLabs failed: {e}. There is currently no backup engine for "
                    "English audio if ElevenLabs is unavailable — this is a known gap."
                )
                return
            except requests.RequestException as e:
                st.error(f"Audio generation failed: {e}")
                return

        st.success("Audio ready!")

        log_document_outcome(
            user_id,
            document_type=document_type,
            document_length_words=document_length_words,
            summarization_percentage=percentage,
            processing_success=True,
            summary_language=language_name,
            audio_generated=True,
            audio_language=audio_language,
            voice_selected=voice,
        )

        st.audio(audio_bytes, format=audio_format)
        st.download_button(
            "Download audio",
            data=audio_bytes,
            file_name=f"summary_{audio_language.lower()}.{file_ext}",
            mime=audio_format,
        )


if __name__ == "__main__":
    main()

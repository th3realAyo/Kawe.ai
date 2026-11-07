import hashlib
import os
import tempfile
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv

# Load local development secrets before importing modules that initialize
# service clients from environment variables.
load_dotenv()

from langdetect import detect

from doc_loader import (
    DocumentExtractionError,
    UnsupportedFileTypeError,
    extract_document,
)
from docx_export import build_summary_docx, build_translation_docx
from elevenlabs_tts import ElevenLabsTTSError, synthesize_with_elevenlabs
from grounded_summarizer import (
    LANGUAGE_NAMES,
    GroundedSummarizerError,
    generate_grounded_summary,
)
from main_integration_snippet import log_document_outcome, require_signup
from product_metrics import build_summary_metrics, summary_top_n, validate_upload_size
from yarngpt_tts import YarnGPTError, voices_for_language
from yarngpt_tts import synthesize as synthesize_with_yarngpt

MODAL_TRANSLATE_URL = os.environ.get("MODAL_TRANSLATE_URL")
RAG_AUTH_TOKEN = os.environ.get("RAG_AUTH_TOKEN")

SUPPORTED_AUDIO_LANGUAGES = ["English", "Yoruba", "Igbo", "Hausa"]


ELEVENLABS_VOICES = ["Daniel", "Chris", "Bella", "Lilly", "Liam"]

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


def call_modal_translate(text: str, target_language: str, max_new_tokens: int = 512) -> str:
    """Translates one summary section. Sections are translated one at a time
    because the model stops after max_new_tokens, so sending the whole
    summary at once cut long translations short."""
    response = requests.post(
        MODAL_TRANSLATE_URL,
        json={
            "text": text,
            "target_language": target_language,
            "max_new_tokens": max_new_tokens,
        },
        headers={"Authorization": f"Bearer {RAG_AUTH_TOKEN}"},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["translated_text"]


@st.cache_data(ttl=3600, show_spinner=False)
def yarngpt_voice_options(language: str) -> list:
    """YarnGPT voices for a language, refreshed hourly (the catalogue changes)."""
    return voices_for_language(language)


def english_backup_voice() -> str:
    voices = yarngpt_voice_options("English")
    return "idera" if "idera" in voices else voices[0]


def render_citations(sections):
    for i, section in enumerate(sections, start=1):
        with st.expander(f"Section {i} — source: {section.location_label}"):
            st.markdown(f"**Summary:** {section.summary_text}")
            st.markdown("**Original excerpt:**")
            st.caption(section.source_excerpt)


def reset_downstream_state():
    for key in ["segments", "language_code", "summary_result", "summary_id", "confirmed", "audio", "translation"]:
        st.session_state.pop(key, None)


def main():  # sourcery skip: low-code-quality
    st.title("Kàwé.ai")
    st.caption("Read smart. Listen better. Every summary section is traceable to its source.")

    user_id = require_signup()

    uploaded_file = st.file_uploader("Upload a PDF or Word document", type=["pdf", "docx"])

    if uploaded_file is None:
        return

    try:
        validate_upload_size(uploaded_file.size)
    except ValueError as error:
        st.error(str(error))
        return

    # New file uploaded -> wipe any previous document's state. Compare file
    # contents, not names, so a different file called "report.pdf" isn't
    # mistaken for the previous one.
    file_id = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    if st.session_state.get("processed_file_id") != file_id:
        reset_downstream_state()
        st.session_state.processed_file_id = file_id

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
    
    document_type = os.path.splitext(uploaded_file.name)[1].lstrip(".")
    document_length_words = sum(len(seg.text.split()) for seg in segments)

    st.success(f"Extracted {len(segments)} section(s). Detected document language: {language_name}")

    st.subheader("1. How much of the document should be summarized?")
    percentage = st.slider(
        "Percentage of document to summarize", min_value=10, max_value=80, value=30, step=10
    )

    if st.button("Generate grounded summary"):
        top_n = summary_top_n(percentage, len(segments))

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
        st.session_state.summary_id = uuid.uuid4().hex
        st.session_state.confirmed = False

    if "summary_result" not in st.session_state:
        return

    result = st.session_state.summary_result

    st.subheader("2. Review your grounded summary")
    if result.skipped_chunks:
        st.caption(f"Note: {result.skipped_chunks} section(s) couldn't be summarized and were skipped.")

    st.write(result.full_text)
    metrics = build_summary_metrics(document_length_words, result.full_text)
    metric_columns = st.columns(3)
    metric_columns[0].metric("Source length", f"{metrics.source_words:,} words")
    metric_columns[1].metric("Summary length", f"{metrics.summary_words:,} words")
    metric_columns[2].metric(
        "Estimated listening time",
        f"{metrics.estimated_listening_minutes} min" if metrics.estimated_listening_minutes else "Under 1 min",
        help="Estimated at roughly 150 words per minute.",
    )
    st.caption(
        f"This summary is about {metrics.compression_percentage}% of the source document."
    )

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
        "Yes, I'm satisfied with this summary — continue to translation and audio",
        value=st.session_state.get("confirmed", False),
    )
    st.session_state.confirmed = confirmed

    if not confirmed:
        st.info("Review the summary above, or adjust the % and regenerate, before moving on.")
        return

    st.subheader("4. Translate and listen")
    audio_language = st.selectbox("Language", SUPPORTED_AUDIO_LANGUAGES)
    needs_translation = audio_language.lower() != language_name.lower()

    if audio_language == "English":
        voice_options = ELEVENLABS_VOICES
    else:
        voice_options = yarngpt_voice_options(audio_language)

    voice = st.selectbox("Voice", voice_options)

    def log_audio_outcome(success: bool, error_type: str | None = None) -> None:
        log_document_outcome(
            user_id,
            document_type=document_type,
            document_length_words=document_length_words,
            summarization_percentage=percentage,
            processing_success=success,
            error_type=error_type,
            summary_language=language_name,
            audio_generated=success,
            audio_language=audio_language,
            voice_selected=voice,
        )

    # Translation and audio are kept in the session so they survive Streamlit
    # reruns (e.g. the rerun triggered by a download click). The keys tie them
    # to this exact summary, language and voice.
    summary_id = st.session_state.get("summary_id")
    translation_key = (summary_id, audio_language)
    audio_key = (summary_id, audio_language, voice)

    def get_translation() -> list | None:
        """Translated summary sections, translating once per summary+language."""
        cached = st.session_state.get("translation")
        if cached and cached["key"] == translation_key:
            return cached["sections"]
        section_texts = [s.summary_text for s in result.sections if s.summary_text.strip()]
        with st.spinner(f"Translating summary into {audio_language}..."):
            try:
                translated = [call_modal_translate(text, audio_language) for text in section_texts]
            except requests.RequestException as e:
                st.error(f"Translation failed: {e}")
                log_audio_outcome(False, f"translation: {e}")
                return None
        st.session_state.translation = {"key": translation_key, "sections": translated}
        return translated

    col_translate, col_audio = st.columns(2)
    translate_clicked = needs_translation and col_translate.button(f"Translate summary to {audio_language}")
    audio_clicked = col_audio.button("Generate audio")

    if translate_clicked:
        get_translation()

    if audio_clicked:
        st.session_state.pop("audio", None)
        if needs_translation:
            section_texts = get_translation()
            if section_texts is None:
                return
        else:
            section_texts = [s.summary_text for s in result.sections if s.summary_text.strip()]
        text_for_audio = " ".join(section_texts)

        with st.spinner("Generating audio... this can take a minute for long summaries."):
            warnings = []
            try:
                if audio_language == "English":
                    try:
                        audio_bytes, file_ext = synthesize_with_elevenlabs(text_for_audio, voice_name=voice), "mp3"
                    except ElevenLabsTTSError as e:
                        backup = english_backup_voice()
                        st.warning(f"ElevenLabs failed ({e}). Using the YarnGPT voice '{backup}' instead.")
                        audio_bytes, file_ext, warnings = synthesize_with_yarngpt(text_for_audio, backup)
                else:
                    audio_bytes, file_ext, warnings = synthesize_with_yarngpt(text_for_audio, voice)
            except (YarnGPTError, requests.RequestException, ValueError) as e:
                st.error(f"Audio generation failed: {e}")
                log_audio_outcome(False, f"tts: {e}")
                return

        for warning in warnings:
            st.info(f"YarnGPT note: {warning}")
        log_audio_outcome(True)
        st.session_state.audio = {
            "key": audio_key,
            "bytes": audio_bytes,
            "format": "audio/mpeg" if file_ext == "mp3" else "audio/wav",
            "ext": file_ext,
        }

    translation = st.session_state.get("translation")
    if needs_translation and translation and translation["key"] == translation_key:
        st.markdown(f"**Summary in {audio_language}:**")
        st.write(" ".join(translation["sections"]))
        st.caption("Machine translation. Check important details against the original document.")
        st.download_button(
            f"Download {audio_language} summary (Word)",
            data=build_translation_docx(uploaded_file.name, audio_language, result, translation["sections"]),
            file_name=f"summary_{audio_language.lower()}_{os.path.splitext(uploaded_file.name)[0]}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    audio = st.session_state.get("audio")
    if audio and audio["key"] == audio_key:
        st.success("Audio ready!")
        st.audio(audio["bytes"], format=audio["format"])
        st.download_button(
            "Download audio",
            data=audio["bytes"],
            file_name=f"summary_{audio_language.lower()}.{audio['ext']}",
            mime=audio["format"],
        )

if __name__ == "__main__":
    main()

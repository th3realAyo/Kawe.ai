import datetime

import streamlit as st

from db import create_user, log_usage_event, user_exists


def _safe_user_exists(user_id: str) -> bool:
    """Treats an unreachable database as 'not recognized' instead of crashing."""
    try:
        return user_exists(user_id)
    except Exception as e:
        print(f"[signup] user lookup failed: {e!r}")
        return False


def require_signup() -> str:
    if "user_id" in st.session_state:
        return st.session_state["user_id"]

    # Returning visitor: the user id is kept in the page address (?u=...) so
    # refreshing or bookmarking the page doesn't create a new account.
    remembered_id = st.query_params.get("u")
    if remembered_id and _safe_user_exists(remembered_id):
        st.session_state["user_id"] = remembered_id
        return remembered_id

    st.subheader("Create your account")

    with st.form("signup_form"):
        username = st.text_input("Username")
        submitted = st.form_submit_button("Sign up")

    if submitted:
        username = username.strip()
        if not username:
            st.error("Username is required.")
        else:
            try:
                user_id = create_user(username=username)
            except Exception as e:  # network/DNS failure, paused project, bad key
                print(f"[signup] create_user failed: {e!r}")
                st.error(
                    "We couldn't reach the account server, so your account wasn't created. "
                    "Check your internet connection and try again in a moment."
                )
                st.stop()
            st.session_state["user_id"] = user_id
            st.query_params["u"] = user_id
            st.rerun()

    st.stop()  # nothing below require_signup() runs until signed up


def log_document_outcome(
    user_id: str,
    document_type: str,
    document_length_words: int,
    summarization_percentage: int,
    processing_success: bool,
    error_type: str | None = None,
    summary_language: str | None = None,
    audio_generated: bool = False,
    audio_language: str | None = None,
    voice_selected: str | None = None,
) -> None:

    now = datetime.datetime.now()
    # Analytics must never break the user's summary or audio, so a database
    # failure here is printed to the server log and otherwise ignored.
    try:
        log_usage_event(
            user_id,
            document_type=document_type,
            document_length_words=document_length_words,
            summarization_percentage=summarization_percentage,
            processing_success=processing_success,
            error_type=error_type,
            summary_language=summary_language,
            audio_generated=audio_generated,
            audio_language=audio_language,
            voice_selected=voice_selected,
            time_of_day=now.strftime("%H:%M"),
            day_of_week=now.strftime("%A"),
        )
    except Exception as e:
        print(f"[analytics] could not log usage event: {e!r}")

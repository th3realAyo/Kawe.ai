"""Email sign-in, account deletion and usage logging for the Streamlit app.

The email identifies an account; it does not authenticate it (there is no password or link
sent by email). Accounts hold only usage statistics, never documents or summaries.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Optional

import streamlit as st

from kawe import db

log = logging.getLogger(__name__)

PRIVACY_POLICY = Path(__file__).resolve().parent.parent / "policies" / "privacy-policy.md"
PRIVACY_NOTICE = (
    "We store your email only as a one-way cryptographic fingerprint: nobody, including us, can "
    "read it back. Your documents are processed for this session only and are not stored. We keep "
    "basic usage statistics (file type, length, languages, whether each step worked) linked to your "
    "account, and you can delete the account and those statistics at any time."
)


def _show_privacy_policy() -> None:
    with st.expander("Privacy Policy"):
        st.write(PRIVACY_NOTICE)
        if PRIVACY_POLICY.exists():
            st.markdown(PRIVACY_POLICY.read_text(encoding="utf-8"))


def require_signup() -> str:
    """Returns the signed-in user id, or shows the sign-in form and stops the script run."""
    if "user_id" in st.session_state:
        return st.session_state["user_id"]

    if st.session_state.pop("account_deleted", False):
        st.success("Your account and its usage statistics have been deleted.")
    st.subheader("Sign in or create your account")
    st.caption(PRIVACY_NOTICE)
    _show_privacy_policy()
    with st.form("signup_form"):
        email = st.text_input("Email address")
        agreed = st.checkbox("I am 18 or older, and I have read and agree to the Privacy Policy.")
        submitted = st.form_submit_button("Continue")

    if submitted:
        if not db.is_valid_email(email):
            st.error("Enter a valid email address.")
        elif not agreed:
            st.error("Please confirm you are 18 or older and agree to the Privacy Policy to continue.")
        else:
            try:
                user_id, _ = db.get_or_create_user(email)
            except Exception as e:
                log.error("Sign-in failed: %r", e)
                st.error(
                    "We couldn't reach the account server, so you weren't signed in. "
                    "Check your internet connection and try again in a moment."
                )
                st.stop()
            st.session_state["user_id"] = user_id
            st.rerun()
    st.stop()


def account_sidebar(user_id: str) -> None:
    """Sign-out and self-service account deletion."""
    with st.sidebar:
        st.markdown("**Your account**")
        if st.button("Sign out"):
            st.session_state.clear()
            st.rerun()
        with st.expander("Delete my account"):
            st.caption("This permanently deletes your account and its usage statistics.")
            with st.form("delete_account_form"):
                email = st.text_input("Type your email address to confirm")
                confirmed = st.form_submit_button("Delete my account permanently")
            if confirmed:
                try:
                    matches = db.is_valid_email(email) and db.find_user(email) == user_id
                    if not matches:
                        st.error("That email doesn't match this account.")
                        return
                    db.delete_user(user_id)
                except Exception as e:
                    log.error("Account deletion failed: %r", e)
                    st.error("We couldn't reach the account server. Nothing was deleted; please try again.")
                    return
                st.session_state.clear()
                st.session_state["account_deleted"] = True
                st.rerun()


def log_document_outcome(
    user_id: str,
    document_type: str,
    document_length_words: int,
    summarization_percentage: int,
    processing_success: bool,
    error_type: Optional[str] = None,
    summary_language: Optional[str] = None,
    audio_generated: bool = False,
    audio_language: Optional[str] = None,
    voice_selected: Optional[str] = None,
    event_type: str = "summary",
    question_answered: Optional[bool] = None,
) -> None:
    """Records a usage event. Failures are logged and never interrupt the user."""
    now = datetime.datetime.now()
    try:
        db.log_usage_event(
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
            event_type=event_type,
            question_answered=question_answered,
            time_of_day=now.strftime("%H:%M"),
            day_of_week=now.strftime("%A"),
        )
    except Exception as e:
        log.warning("Could not log usage event: %r", e)

"""Supabase storage for accounts and usage events.

Email addresses are never stored. Each is reduced to a keyed one-way hash (HMAC-SHA256 with the
server secret EMAIL_HASH_KEY), which can be matched against an address someone types in but
cannot be turned back into the address, by the operator or by anyone with database access.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from functools import lru_cache
from typing import Optional, Tuple

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MAX_EMAIL_CHARS = 254


@lru_cache(maxsize=1)
def _client():
    from supabase import create_client

    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])


def normalize_email(email: str) -> str:
    return email.strip().lower()


def is_valid_email(email: str) -> bool:
    email = normalize_email(email)
    return len(email) <= MAX_EMAIL_CHARS and bool(_EMAIL.match(email))


def hash_email(email: str) -> str:
    key = os.environ.get("EMAIL_HASH_KEY", "")
    if len(key) < 32:
        raise RuntimeError("EMAIL_HASH_KEY must be set to a random secret of at least 32 characters.")
    return hmac.new(key.encode(), normalize_email(email).encode(), hashlib.sha256).hexdigest()


def find_user(email: str) -> Optional[str]:
    rows = _client().table("users").select("id").eq("email_hash", hash_email(email)).limit(1).execute().data
    return rows[0]["id"] if rows else None


def get_or_create_user(email: str) -> Tuple[str, bool]:
    """Returns (user id, whether the account was just created)."""
    existing = find_user(email)
    if existing:
        return existing, False
    try:
        result = _client().table("users").insert({"email_hash": hash_email(email)}).execute()
        return result.data[0]["id"], True
    except Exception:
        existing = find_user(email)  # created by a simultaneous sign-up with the same email
        if existing:
            return existing, False
        raise


def log_usage_event(user_id: str, **fields) -> None:
    _client().table("usage_events").insert({**fields, "user_id": user_id}).execute()


def delete_user(user_id: str) -> None:
    """Deletes the account; its usage events go with it (on delete cascade)."""
    _client().table("users").delete().eq("id", user_id).execute()

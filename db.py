import os
import uuid

from cryptography.fernet import Fernet
from supabase import Client, create_client

_fernet = Fernet(os.environ["DB_ENCRYPTION_KEY"].encode())

_supabase: Client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"],
)


def encrypt(value: str) -> str:
    return _fernet.encrypt(value.encode()).decode()


def decrypt(token: str) -> str:
    return _fernet.decrypt(token.encode()).decode()


def create_user(username: str) -> str:
    """Creates the user row. Returns the new user_id."""
    user_row = {"username_encrypted": encrypt(username)}
    result = _supabase.table("users").insert(user_row).execute()
    return result.data[0]["id"]


def user_exists(user_id: str) -> bool:
    """True if user_id is a well-formed id that exists in the users table."""
    try:
        uuid.UUID(user_id)
    except ValueError:
        return False
    rows = _supabase.table("users").select("id").eq("id", user_id).limit(1).execute().data
    return bool(rows)


def log_usage_event(user_id: str, **event_fields) -> None:
    event_fields["user_id"] = user_id
    _supabase.table("usage_events").insert(event_fields).execute()


def delete_user(user_id: str) -> None:
    _supabase.table("users").delete().eq("id", user_id).execute()

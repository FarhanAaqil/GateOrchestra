"""
assistant/userkeys.py
=====================
Bring-your-own API keys for hosted mode: a person may store one Groq and one
Gemini key, which then answer their own chats instead of the operator's keys.

Keys are encrypted with Fernet (AES-128-CBC with an HMAC) under
GATE_ENCRYPTION_KEY before they touch the database, are decrypted only to
make a call, and are never returned by the API: listing shows only which
providers have a key and when it was added.

Make a key with:
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
Losing it makes the stored keys unreadable; people would then add them again.
"""

from __future__ import annotations

import os
import time
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from assistant.store import Store

SCHEMA = """
CREATE TABLE IF NOT EXISTS user_keys (
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    ciphertext TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (user_id, provider)
);
"""
PROVIDERS = ("groq", "gemini", "openai")


class KeyStoreError(RuntimeError):
    pass


def encryption_key_problem(value: str | None) -> str | None:
    """Why ``value`` is not a usable GATE_ENCRYPTION_KEY (None when it is)."""
    if not value:
        return "GATE_ENCRYPTION_KEY must be set (a Fernet key) to store people's own API keys."
    try:
        Fernet(value.encode())
    except (ValueError, TypeError):
        return "GATE_ENCRYPTION_KEY is not a valid Fernet key (32 url-safe base64 bytes)."
    return None


def _cipher() -> Fernet:
    value = os.getenv("GATE_ENCRYPTION_KEY", "").strip()
    problem = encryption_key_problem(value)
    if problem:
        raise KeyStoreError(problem)
    return Fernet(value.encode())


class UserKeys:
    def __init__(self, store: Store) -> None:
        self.store = store
        with store.transaction() as db:
            db.executescript(SCHEMA)

    def put(self, user_id: str, provider: str, key: str) -> None:
        if provider not in PROVIDERS:
            raise KeyStoreError("Provider must be groq, gemini or openai.")
        token = _cipher().encrypt(key.encode("utf-8")).decode("ascii")
        with self.store.transaction() as db:
            db.execute(
                "INSERT INTO user_keys (user_id, provider, ciphertext, created_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(user_id, provider) DO UPDATE SET "
                "ciphertext = excluded.ciphertext, created_at = excluded.created_at",
                (user_id, provider, token, time.time()),
            )

    def remove(self, user_id: str, provider: str) -> bool:
        with self.store.transaction() as db:
            gone = db.execute(
                "DELETE FROM user_keys WHERE user_id = ? AND provider = ?", (user_id, provider)
            ).rowcount
        return gone > 0

    def listing(self, user_id: str) -> list[dict[str, Any]]:
        """Which providers have a key, and since when. Never the key itself."""
        with self.store.read() as db:
            rows = db.execute(
                "SELECT provider, created_at FROM user_keys WHERE user_id = ? ORDER BY provider",
                (user_id,),
            ).fetchall()
        return [{"provider": r["provider"], "added_at": r["created_at"]} for r in rows]

    def keys(self, user_id: str) -> dict[str, str]:
        """Decrypted keys by provider, for making this person's calls (never sent out)."""
        with self.store.read() as db:
            rows = db.execute(
                "SELECT provider, ciphertext FROM user_keys WHERE user_id = ?", (user_id,)
            ).fetchall()
        if not rows:
            return {}
        cipher = _cipher()
        out: dict[str, str] = {}
        for row in rows:
            try:
                out[row["provider"]] = cipher.decrypt(row["ciphertext"].encode()).decode("utf-8")
            except InvalidToken:
                continue  # encrypted under an older GATE_ENCRYPTION_KEY: unusable, skip it
        return out

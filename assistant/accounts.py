"""
assistant/accounts.py
=====================
User accounts for hosted mode: sign-up, email verification, password reset
and browser sessions, stored in the assistant's SQLite database.

  * Emails are unique and stored lower-case.
  * Passwords are hashed with argon2id (argon2-cffi defaults); the hash is
    upgraded on login when the parameters change.
  * Session tokens and email tokens are random (32 bytes), handed out once and
    stored only as SHA-256 hashes, so a copy of the database cannot be used to
    sign in. Email tokens are single-use and expire.
  * Admins are the addresses in GATE_ADMIN_EMAILS, checked on every request,
    so removing an address takes effect at once. With GATE_REQUIRE_ADMIN_2FA
    they must also have two-factor sign-in switched on.
  * Two-factor sign-in (TOTP): the secret is encrypted with
    GATE_ENCRYPTION_KEY, the ten recovery codes are stored as hashes and work
    once each, and a code's time step is remembered so it cannot be replayed.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import time
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from assistant import totp
from assistant.store import Store, new_id

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL,
    verified_at REAL, created_at REAL NOT NULL,
    plan TEXT NOT NULL DEFAULT 'free', paid_until REAL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    csrf TEXT NOT NULL, created_at REAL NOT NULL, expires_at REAL NOT NULL, last_seen REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE TABLE IF NOT EXISTS email_tokens (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    purpose TEXT NOT NULL, created_at REAL NOT NULL, expires_at REAL NOT NULL, used_at REAL
);
CREATE TABLE IF NOT EXISTS recovery_codes (
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    code_hash TEXT NOT NULL, used_at REAL,
    PRIMARY KEY (user_id, code_hash)
);
"""
# Columns added after the first release; created on start-up when missing.
MIGRATIONS = {
    "users": {"totp_secret": "TEXT", "totp_enabled_at": "REAL", "totp_last_step": "INTEGER"},
    "email_tokens": {"data": "TEXT"},
}
TOKEN_SECONDS = {"verify": 24 * 3600, "reset": 3600, "email_change": 3600, "2fa": 300}

SESSION_SECONDS = 14 * 24 * 3600
VERIFY_SECONDS = 24 * 3600
RESET_SECONDS = 3600
MIN_PASSWORD = 10
MAX_PASSWORD = 128
_EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]+\.[^@\s]{2,}$")
_HASHER = PasswordHasher()
# Checked when the email is unknown, so a wrong address takes as long as a wrong password.
_DUMMY_HASH = _HASHER.hash("not-a-real-password-" + secrets.token_hex(8))


class AccountError(ValueError):
    pass


def normalize_email(email: str) -> str:
    value = (email or "").strip().lower()
    if len(value) > 254 or not _EMAIL.match(value):
        raise AccountError("Enter a valid email address.")
    return value


def check_password(password: str) -> None:
    if len(password or "") < MIN_PASSWORD:
        raise AccountError(f"Use at least {MIN_PASSWORD} characters for the password.")
    if len(password) > MAX_PASSWORD:
        raise AccountError(f"Passwords are limited to {MAX_PASSWORD} characters.")


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def admin_emails() -> set[str]:
    return {e.strip().lower() for e in os.getenv("GATE_ADMIN_EMAILS", "").split(",") if e.strip()}


def admin_2fa_required() -> bool:
    return os.getenv("GATE_REQUIRE_ADMIN_2FA", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _cipher():
    from assistant.userkeys import (
        _cipher as key_cipher,
    )  # noqa: PLC0415  (same key as own API keys)

    return key_cipher()


class Accounts:
    def __init__(self, store: Store) -> None:
        self.store = store
        with store.transaction() as db:
            db.executescript(SCHEMA)
            for table, columns in MIGRATIONS.items():
                existing = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
                for name, kind in columns.items():
                    if name not in existing:
                        db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")

    # Users

    def public(self, user: dict[str, Any]) -> dict[str, Any]:
        """What the browser may see about a user (never the password hash)."""
        return {
            "id": user["id"],
            "email": user["email"],
            "verified": user["verified_at"] is not None,
            # Only once the address is confirmed: otherwise anyone could sign up with an
            # admin's address before the admin does.
            "is_admin": user["verified_at"] is not None
            and user["email"] in admin_emails()
            and (user.get("totp_enabled_at") is not None or not admin_2fa_required()),
            "two_factor": user.get("totp_enabled_at") is not None,
            "plan": user["plan"],
            "paid_until": user["paid_until"],
            "created_at": user["created_at"],
        }

    def get(self, user_id: str) -> dict[str, Any] | None:
        with self.store.read() as db:
            row = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None

    def by_email(self, email: str) -> dict[str, Any] | None:
        with self.store.read() as db:
            row = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        return dict(row) if row else None

    def create(self, email: str, password: str) -> tuple[dict[str, Any], bool]:
        """(user, created). An existing address returns that user with created=False."""
        email = normalize_email(email)
        check_password(password)
        existing = self.by_email(email)
        if existing:
            return existing, False
        uid, now = new_id(), time.time()
        with self.store.transaction() as db:
            db.execute(
                "INSERT INTO users (id, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (uid, email, _HASHER.hash(password), now),
            )
        user = self.get(uid)
        assert user is not None
        return user, True

    def authenticate(self, email: str, password: str) -> dict[str, Any] | None:
        try:
            email = normalize_email(email)
        except AccountError:
            email = ""
        user = self.by_email(email) if email else None
        try:
            _HASHER.verify(user["password_hash"] if user else _DUMMY_HASH, password or "")
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            return None
        if user is None:
            return None
        if _HASHER.check_needs_rehash(user["password_hash"]):
            with self.store.transaction() as db:
                db.execute(
                    "UPDATE users SET password_hash = ? WHERE id = ?",
                    (_HASHER.hash(password), user["id"]),
                )
        return user

    def set_password(self, user_id: str, password: str) -> None:
        """New password; every session of the user is signed out."""
        check_password(password)
        with self.store.transaction() as db:
            db.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (_HASHER.hash(password), user_id),
            )
            db.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))

    def mark_verified(self, user_id: str) -> None:
        with self.store.transaction() as db:
            db.execute(
                "UPDATE users SET verified_at = COALESCE(verified_at, ?) WHERE id = ?",
                (time.time(), user_id),
            )

    def set_plan(self, user_id: str, plan: str, paid_until: float | None) -> bool:
        with self.store.transaction() as db:
            changed = db.execute(
                "UPDATE users SET plan = ?, paid_until = ? WHERE id = ?",
                (plan, paid_until, user_id),
            ).rowcount
        return changed > 0

    def search(self, query: str = "", limit: int = 50) -> list[dict[str, Any]]:
        """Users whose email contains ``query``, newest first (admin view)."""
        like = f"%{query.strip().lower()}%"
        with self.store.read() as db:
            rows = db.execute(
                "SELECT * FROM users WHERE email LIKE ? ORDER BY created_at DESC LIMIT ?",
                (like, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def change_email(self, user_id: str, new_email: str) -> None:
        """Switch to a confirmed new address; every session is signed out."""
        new_email = normalize_email(new_email)
        with self.store.transaction() as db:
            taken = db.execute(
                "SELECT 1 FROM users WHERE email = ? AND id != ?", (new_email, user_id)
            ).fetchone()
            if taken:
                raise AccountError("That address already has an account.")
            db.execute(
                "UPDATE users SET email = ?, verified_at = ? WHERE id = ?",
                (new_email, time.time(), user_id),
            )
            db.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))

    # Two-factor sign-in

    def begin_two_factor(self, user_id: str) -> str:
        """A new secret, stored (encrypted) but not active until a code from it is confirmed."""
        secret = totp.new_secret()
        sealed = _cipher().encrypt(secret.encode()).decode("ascii")
        with self.store.transaction() as db:
            db.execute(
                "UPDATE users SET totp_secret = ?, totp_enabled_at = NULL, totp_last_step = NULL "
                "WHERE id = ?",
                (sealed, user_id),
            )
        return secret

    def _secret(self, user: dict[str, Any]) -> str | None:
        sealed = user.get("totp_secret")
        if not sealed:
            return None
        return _cipher().decrypt(sealed.encode()).decode("ascii")

    def _use_step(self, user: dict[str, Any], code: str) -> bool:
        secret = self._secret(user)
        step = totp.matching_step(secret, code) if secret else None
        if step is None or step <= (user.get("totp_last_step") or -1):
            return False  # wrong, or already used (no replays)
        with self.store.transaction() as db:
            db.execute("UPDATE users SET totp_last_step = ? WHERE id = ?", (step, user["id"]))
        return True

    def enable_two_factor(self, user_id: str, code: str) -> list[str] | None:
        """Confirm the pending secret with a code; returns the recovery codes (shown once)."""
        user = self.get(user_id)
        if user is None or not self._use_step(user, code):
            return None
        codes = totp.new_recovery_codes()
        with self.store.transaction() as db:
            db.execute("DELETE FROM recovery_codes WHERE user_id = ?", (user_id,))
            db.executemany(
                "INSERT INTO recovery_codes (user_id, code_hash) VALUES (?, ?)",
                [(user_id, hash_token(c)) for c in codes],
            )
            db.execute("UPDATE users SET totp_enabled_at = ? WHERE id = ?", (time.time(), user_id))
        return codes

    def disable_two_factor(self, user_id: str) -> None:
        with self.store.transaction() as db:
            db.execute(
                "UPDATE users SET totp_secret = NULL, totp_enabled_at = NULL, totp_last_step = NULL "
                "WHERE id = ?",
                (user_id,),
            )
            db.execute("DELETE FROM recovery_codes WHERE user_id = ?", (user_id,))

    def check_second_factor(self, user_id: str, code: str) -> bool:
        """A current authenticator code, or an unused recovery code (spent on use)."""
        user = self.get(user_id)
        if user is None or user.get("totp_enabled_at") is None:
            return False
        if self._use_step(user, code):
            return True
        cleaned = (code or "").strip().lower()
        with self.store.transaction() as db:
            spent = db.execute(
                "UPDATE recovery_codes SET used_at = ? "
                "WHERE user_id = ? AND code_hash = ? AND used_at IS NULL",
                (time.time(), user_id, hash_token(cleaned)),
            ).rowcount
        return spent > 0

    def recovery_codes_left(self, user_id: str) -> int:
        with self.store.read() as db:
            return int(
                db.execute(
                    "SELECT COUNT(*) FROM recovery_codes WHERE user_id = ? AND used_at IS NULL",
                    (user_id,),
                ).fetchone()[0]
            )

    def delete(self, user_id: str) -> None:
        with self.store.transaction() as db:
            db.execute("DELETE FROM users WHERE id = ?", (user_id,))

    # Sessions

    def create_session(self, user_id: str) -> tuple[str, str]:
        """(session token for the cookie, CSRF token for the X-CSRF-Token header)."""
        token, csrf, now = secrets.token_urlsafe(32), secrets.token_urlsafe(24), time.time()
        with self.store.transaction() as db:
            db.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
            db.execute(
                "INSERT INTO sessions (token_hash, user_id, csrf, created_at, expires_at, last_seen) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (hash_token(token), user_id, csrf, now, now + SESSION_SECONDS, now),
            )
        return token, csrf

    def session(self, token: str | None) -> tuple[dict[str, Any], str] | None:
        """(user, csrf) for a live session token, or None."""
        if not token:
            return None
        with self.store.read() as db:
            row = db.execute(
                "SELECT s.csrf, s.expires_at, s.last_seen, u.* FROM sessions s "
                "JOIN users u ON u.id = s.user_id WHERE s.token_hash = ?",
                (hash_token(token),),
            ).fetchone()
        if row is None:
            return None
        now = time.time()
        if row["expires_at"] < now:
            self.end_session(token)
            return None
        if now - row["last_seen"] > 300:  # at most one write per session every 5 minutes
            with self.store.transaction() as db:
                db.execute(
                    "UPDATE sessions SET last_seen = ? WHERE token_hash = ?",
                    (now, hash_token(token)),
                )
        data = dict(row)
        csrf = data.pop("csrf")
        data.pop("expires_at")
        data.pop("last_seen")
        return data, csrf

    def end_session(self, token: str) -> None:
        with self.store.transaction() as db:
            db.execute("DELETE FROM sessions WHERE token_hash = ?", (hash_token(token),))

    # Email tokens (verification and password reset)

    def issue_token(self, user_id: str, purpose: str, data: str | None = None) -> str:
        """A fresh single-use token; earlier unused ones for the same purpose stop working."""
        ttl = TOKEN_SECONDS.get(purpose, RESET_SECONDS)
        token, now = secrets.token_urlsafe(32), time.time()
        with self.store.transaction() as db:
            db.execute(
                "DELETE FROM email_tokens WHERE user_id = ? AND purpose = ? AND used_at IS NULL",
                (user_id, purpose),
            )
            db.execute(
                "INSERT INTO email_tokens "
                "(token_hash, user_id, purpose, created_at, expires_at, data) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (hash_token(token), user_id, purpose, now, now + ttl, data),
            )
        return token

    def peek_token(self, token: str, purpose: str) -> tuple[str, str | None] | None:
        """(user id, data) of a live token without spending it."""
        if not token:
            return None
        with self.store.read() as db:
            row = db.execute(
                "SELECT user_id, data, expires_at, used_at FROM email_tokens "
                "WHERE token_hash = ? AND purpose = ?",
                (hash_token(token), purpose),
            ).fetchone()
        if row is None or row["used_at"] is not None or row["expires_at"] < time.time():
            return None
        return row["user_id"], row["data"]

    def consume_token_data(self, token: str, purpose: str) -> tuple[str, str | None] | None:
        """(user id, data) of a valid token; the token is spent either way."""
        found = self.peek_token(token, purpose)
        user_id = self.consume_token(token, purpose)
        return (user_id, found[1]) if user_id and found else None

    def consume_token(self, token: str, purpose: str) -> str | None:
        """The user id a valid token belongs to; the token is spent either way."""
        if not token:
            return None
        now = time.time()
        with self.store.transaction() as db:
            row = db.execute(
                "SELECT user_id, expires_at, used_at FROM email_tokens "
                "WHERE token_hash = ? AND purpose = ?",
                (hash_token(token), purpose),
            ).fetchone()
            if row is None or row["used_at"] is not None:
                return None
            db.execute(
                "UPDATE email_tokens SET used_at = ? WHERE token_hash = ?",
                (now, hash_token(token)),
            )
        return row["user_id"] if row["expires_at"] >= now else None

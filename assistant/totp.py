"""
assistant/totp.py
=================
Time-based one-time passwords (RFC 6238, the six-digit codes of Google
Authenticator, 1Password, Authy and the like), standard library only.

  * SHA-1, 30-second steps, 6 digits: what every authenticator app expects.
  * A code is accepted for the current step and one step either side, to
    allow for clock drift. The caller stores the last step used so a code
    cannot be replayed.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import struct
import time
from urllib.parse import quote, urlencode

STEP = 30
DIGITS = 6


def new_secret() -> str:
    """A random 160-bit secret, base32 without padding (how apps take it)."""
    return base64.b32encode(secrets.token_bytes(20)).decode("ascii").rstrip("=")


def _key(secret: str) -> bytes:
    padded = secret.upper() + "=" * (-len(secret) % 8)
    return base64.b32decode(padded)


def code_at(secret: str, step: int) -> str:
    digest = hmac.new(_key(secret), struct.pack(">Q", step), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    value = struct.unpack(">I", digest[offset : offset + 4])[0] & 0x7FFFFFFF
    return str(value % 10**DIGITS).zfill(DIGITS)


def current_step(now: float | None = None) -> int:
    return int((time.time() if now is None else now) // STEP)


def matching_step(secret: str, code: str, now: float | None = None, drift: int = 1) -> int | None:
    """The time step ``code`` belongs to (within +/- ``drift`` steps), or None."""
    code = "".join(ch for ch in (code or "") if ch.isdigit())
    if len(code) != DIGITS:
        return None
    step = current_step(now)
    for candidate in range(step - drift, step + drift + 1):
        if hmac.compare_digest(code_at(secret, candidate), code):
            return candidate
    return None


def otpauth_uri(secret: str, account: str, issuer: str = "GateOrchestra") -> str:
    """The link an authenticator app reads from the QR code."""
    label = quote(f"{issuer}:{account}")
    query = urlencode(
        {"secret": secret, "issuer": issuer, "algorithm": "SHA1", "digits": DIGITS, "period": STEP}
    )
    return f"otpauth://totp/{label}?{query}"


def new_recovery_codes(count: int = 10) -> list[str]:
    """One-time backup codes such as 'k7f2-9xqa', for when the phone is lost."""
    alphabet = "abcdefghjkmnpqrstuvwxyz23456789"
    return [
        "".join(secrets.choice(alphabet) for _ in range(4))
        + "-"
        + "".join(secrets.choice(alphabet) for _ in range(4))
        for _ in range(count)
    ]

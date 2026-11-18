"""Helpers for hosted-mode tests: a locked-down environment and signed-in clients."""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

ADMIN_TOKEN = "admin-" + "t" * 30
GROQ_KEY = "gsk_" + "h" * 40
PASSWORD = "correct horse battery staple"

GOOD_ENV = {
    "GATE_HOSTED": "true",
    "GATE_CORS_ORIGINS": "https://gate.example.com",
    "GATE_ADMIN_TOKEN": ADMIN_TOKEN,
    "ASSISTANT_PROVIDERS": "groq,gemini",
    "GATE_LLM_PROVIDER": "groq",
    "GROQ_API_KEY": GROQ_KEY,
    "GATE_PUBLIC_URL": "https://gate.example.com",
    "GATE_ADMIN_EMAILS": "boss@example.com",
    "GATE_MAIL_BACKEND": "console",
    "GATE_COOKIE_SECURE": "false",
    # A fixed test key (Fernet format); never used outside tests.
    "GATE_ENCRYPTION_KEY": "dGVzdC1vbmx5LWtleS1kby1ub3QtdXNlLTAwMDAwMDA=",
}


def last_link_token(to: str) -> str:
    """The token in the newest console email sent to ``to``."""
    from assistant.mailer import OUTBOX

    mail = next(m for m in reversed(OUTBOX) if m["to"] == to)
    match = re.search(r"token=([\w\-]+)", mail["body"])
    assert match, mail["subject"]
    return match.group(1)


def signed_in(email: str, *, verify: bool = True, password: str = PASSWORD) -> TestClient:
    """A client with its own cookie jar, signed up, optionally verified, and logged in.

    The CSRF token is set as a default header, as the console does.
    """
    from api.main import app

    client = TestClient(app)
    assert (
        client.post("/auth/signup", json={"email": email, "password": password}).status_code == 202
    )
    if verify:
        token = last_link_token(email.lower())
        assert client.post("/auth/verify", json={"token": token}).status_code == 200
    login = client.post("/auth/login", json={"email": email, "password": password})
    assert login.status_code == 200, login.text
    client.headers["X-CSRF-Token"] = login.json()["csrf_token"]
    return client

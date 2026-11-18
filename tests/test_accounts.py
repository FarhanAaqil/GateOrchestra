"""Hosted-mode accounts: sign-up, verification, sessions, CSRF, password reset, admins."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from assistant.accounts import AccountError, Accounts, hash_token, normalize_email
from assistant.mailer import OUTBOX
from assistant.security import RateLimiter
from assistant.service import AssistantService
from tests.hosted_utils import PASSWORD, last_link_token, signed_in


def client() -> TestClient:
    from api.main import app

    return TestClient(app)


def test_accounts_are_off_outside_hosted_mode(service: AssistantService) -> None:
    body = {"email": "a@example.com", "password": PASSWORD}
    assert client().post("/auth/signup", json=body).status_code == 404
    assert client().post("/auth/login", json=body).status_code == 404


def test_emails_are_normalised_and_passwords_checked(service: AssistantService) -> None:
    assert normalize_email("  Ada@Example.COM ") == "ada@example.com"
    for bad in ("", "no-at-sign", "a@b", "x" * 300 + "@example.com"):
        with pytest.raises(AccountError):
            normalize_email(bad)
    with pytest.raises(AccountError, match="at least"):
        service.accounts.create("a@example.com", "short")


def test_passwords_and_tokens_are_never_stored_in_the_clear(service: AssistantService) -> None:
    user, created = service.accounts.create("Ada@Example.com", PASSWORD)
    assert created and user["email"] == "ada@example.com"
    assert user["password_hash"].startswith("$argon2id$") and PASSWORD not in user["password_hash"]
    again, created_again = service.accounts.create("ada@example.com", "another long password")
    assert again["id"] == user["id"] and not created_again
    token, csrf = service.accounts.create_session(user["id"])
    with service.store.read() as db:
        stored = [r[0] for r in db.execute("SELECT token_hash FROM sessions")]
    assert stored == [hash_token(token)] and token not in stored
    assert service.accounts.session(token)[1] == csrf
    assert service.accounts.public(user).keys() >= {"id", "email", "verified", "is_admin"}
    assert "password_hash" not in service.accounts.public(user)


def test_signup_verify_login_me_logout(service: AssistantService, hosted: None) -> None:
    c = client()
    assert c.post(
        "/auth/signup", json={"email": "Ada@Example.com", "password": PASSWORD}
    ).json() == {"status": "check_email"}
    assert OUTBOX[-1]["to"] == "ada@example.com" and "/verify?token=" in OUTBOX[-1]["body"]
    login = c.post("/auth/login", json={"email": "ada@example.com", "password": PASSWORD})
    assert login.status_code == 200
    cookie = login.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    me = c.get("/auth/me").json()
    assert me["user"]["email"] == "ada@example.com" and me["user"]["verified"] is False
    assert me["csrf_token"] == login.json()["csrf_token"]

    # Unverified users may sign in but not chat.
    assert c.get("/conversations").status_code == 403
    token = last_link_token("ada@example.com")
    assert c.post("/auth/verify", json={"token": token}).status_code == 200
    assert c.post("/auth/verify", json={"token": token}).status_code == 400  # single use
    assert c.get("/conversations").status_code == 200

    # Logging out needs the CSRF token, then the session is gone.
    assert c.post("/auth/logout").status_code == 403
    assert c.post("/auth/logout", headers={"X-CSRF-Token": me["csrf_token"]}).status_code == 200
    assert c.get("/auth/me").status_code == 401


def test_wrong_passwords_and_unknown_emails_get_the_same_answer(
    service: AssistantService, hosted: None
) -> None:
    signed_in("ada@example.com")
    c = client()
    wrong = c.post("/auth/login", json={"email": "ada@example.com", "password": "nope" * 3})
    unknown = c.post("/auth/login", json={"email": "who@example.com", "password": PASSWORD})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()
    # Signing up again with a taken address gives the same reply as a new account.
    again = c.post("/auth/signup", json={"email": "ada@example.com", "password": PASSWORD})
    assert again.status_code == 202 and again.json() == {"status": "check_email"}
    assert "already have" in OUTBOX[-1]["subject"]


def test_chat_endpoints_need_a_session_and_csrf_on_writes(
    service: AssistantService, hosted: None
) -> None:
    anonymous = client()
    assert anonymous.get("/conversations").status_code == 401
    assert anonymous.post("/chat", json={"message": "hi"}).status_code == 401
    # The shared API token is not how users sign in when hosted.
    assert anonymous.get("/conversations", headers={"Authorization": "Bearer x"}).status_code == 401

    ada = signed_in("ada@example.com")
    csrf = ada.headers.pop("X-CSRF-Token")
    assert ada.post("/chat", json={"message": "hi"}).status_code == 403
    assert (
        ada.post("/chat", json={"message": "hi"}, headers={"X-CSRF-Token": "x"}).status_code == 403
    )
    turn = ada.post("/chat", json={"message": "What is 2+2?"}, headers={"X-CSRF-Token": csrf})
    assert turn.status_code == 200 and '"type": "done"' in turn.text


def test_password_reset_is_single_use_and_signs_everyone_out(
    service: AssistantService, hosted: None
) -> None:
    ada = signed_in("ada@example.com")
    c = client()
    assert c.post("/auth/forgot", json={"email": "ada@example.com"}).status_code == 202
    # Unknown addresses get the same reply and no email.
    sent = len(OUTBOX)
    assert c.post("/auth/forgot", json={"email": "nobody@example.com"}).json() == {
        "status": "check_email"
    }
    assert len(OUTBOX) == sent
    token = last_link_token("ada@example.com")
    assert c.post("/auth/reset", json={"token": token, "password": "short"}).status_code == 400
    new = "a brand new long password"
    assert c.post("/auth/reset", json={"token": token, "password": new}).status_code == 200
    assert c.post("/auth/reset", json={"token": token, "password": new}).status_code == 400
    assert ada.get("/auth/me").status_code == 401  # the old session was ended
    assert (
        c.post("/auth/login", json={"email": "ada@example.com", "password": PASSWORD}).status_code
        == 401
    )
    assert (
        c.post("/auth/login", json={"email": "ada@example.com", "password": new}).status_code == 200
    )


def test_email_tokens_expire(service: AssistantService) -> None:
    accounts: Accounts = service.accounts
    user, _ = accounts.create("ada@example.com", PASSWORD)
    token = accounts.issue_token(user["id"], "reset")
    with service.store.transaction() as db:
        db.execute("UPDATE email_tokens SET expires_at = ?", (time.time() - 1,))
    assert accounts.consume_token(token, "reset") is None
    fresh = accounts.issue_token(user["id"], "verify")
    assert accounts.consume_token(fresh, "reset") is None  # wrong purpose
    newer = accounts.issue_token(user["id"], "verify")
    assert accounts.consume_token(fresh, "verify") is None  # replaced by the newer link
    assert accounts.consume_token(newer, "verify") == user["id"]


def test_expired_sessions_are_refused(service: AssistantService) -> None:
    user, _ = service.accounts.create("ada@example.com", PASSWORD)
    token, _ = service.accounts.create_session(user["id"])
    with service.store.transaction() as db:
        db.execute("UPDATE sessions SET expires_at = ?", (time.time() - 1,))
    assert service.accounts.session(token) is None
    assert service.accounts.session("") is None and service.accounts.session("made-up") is None


def test_auth_endpoints_are_rate_limited(
    service: AssistantService, hosted: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from api import auth_routes

    monkeypatch.setattr(auth_routes, "_by_email", RateLimiter(2, window=900))
    c = client()
    body = {"email": "ada@example.com", "password": "wrong password!"}
    assert c.post("/auth/login", json=body).status_code == 401
    assert c.post("/auth/login", json=body).status_code == 401
    limited = c.post("/auth/login", json=body)
    assert limited.status_code == 429 and int(limited.headers["retry-after"]) >= 1
    # Another address is not held back by the first one's attempts.
    other = {"email": "bob@example.com", "password": "wrong password!"}
    assert c.post("/auth/login", json=other).status_code == 401


def test_admins_are_named_in_the_environment_and_must_be_verified(
    service: AssistantService, hosted: None
) -> None:
    boss = signed_in("boss@example.com", verify=False)
    # Named in GATE_ADMIN_EMAILS but not confirmed yet: whoever signed up may not own the inbox.
    assert boss.get("/auth/me").json()["user"]["is_admin"] is False
    assert boss.get("/assistant/stats").status_code == 403

    verify = (
        next(m for m in OUTBOX if "Confirm" in m["subject"])["body"].split("token=")[1].split()[0]
    )
    assert boss.post("/auth/verify", json={"token": verify}).status_code == 200
    assert boss.get("/auth/me").json()["user"]["is_admin"] is True
    assert boss.get("/assistant/stats").status_code == 200
    assert boss.get("/history").status_code == 200
    # Admin writes still need the CSRF token.
    csrf = boss.headers.pop("X-CSRF-Token")
    assert boss.delete("/assistant/cache").status_code == 403
    assert boss.delete("/assistant/cache", headers={"X-CSRF-Token": csrf}).status_code == 200

    user = signed_in("ada@example.com")
    for path in ("/assistant/stats", "/assistant/status", "/metrics", "/history"):
        assert user.get(path).status_code == 403, path
    assert user.delete("/assistant/cache").status_code == 403


def test_mail_failures_are_reported_not_hidden(
    service: AssistantService, hosted: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GATE_MAIL_BACKEND", "smtp")
    monkeypatch.setenv("GATE_SMTP_HOST", "127.0.0.1")
    monkeypatch.setenv("GATE_SMTP_PORT", "1")  # nothing listens there
    monkeypatch.setenv("GATE_MAIL_FROM", "noreply@example.com")
    reply = client().post("/auth/signup", json={"email": "ada@example.com", "password": PASSWORD})
    assert reply.status_code == 503 and "could not be sent" in reply.json()["detail"]

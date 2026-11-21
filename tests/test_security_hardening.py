"""Hosted mode hardening: lockout, two-factor sign-in, email change, headers, providers."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from assistant import totp
from assistant.config import AssistantSettings
from assistant.hosted import hosted_config_problems
from assistant.keys import KeyPool
from assistant.llm import Gateway, OpenAICompatibleClient
from assistant.mailer import OUTBOX
from assistant.service import AssistantService
from tests.hosted_utils import GOOD_ENV, PASSWORD, last_link_token, signed_in


def client() -> TestClient:
    from api.main import app

    return TestClient(app)


def turn_on_two_factor(c: TestClient) -> tuple[str, list[str]]:
    secret = c.post("/account/2fa/setup", json={"password": PASSWORD}).json()["secret"]
    codes = c.post(
        "/account/2fa/enable", json={"code": totp.code_at(secret, totp.current_step())}
    ).json()
    return secret, codes["recovery_codes"]


def test_rfc6238_vectors() -> None:
    import base64

    secret = base64.b32encode(b"12345678901234567890").decode().rstrip("=")
    assert totp.code_at(secret, 59 // 30) == "287082"
    assert totp.code_at(secret, 1111111109 // 30) == "081804"
    assert totp.matching_step(secret, "287082", now=59) == 1
    assert totp.matching_step(secret, "000000", now=59) is None
    assert totp.otpauth_uri(secret, "ada@example.com").startswith(
        "otpauth://totp/GateOrchestra%3Aada%40example.com?"
    )
    codes = totp.new_recovery_codes()
    assert len(codes) == len(set(codes)) == 10 and all(len(c) == 9 and c[4] == "-" for c in codes)


def test_repeated_wrong_passwords_pause_the_account(
    service: AssistantService, hosted: None
) -> None:
    signed_in("ada@example.com")
    c = client()
    for _ in range(5):
        assert (
            c.post(
                "/auth/login", json={"email": "ada@example.com", "password": "wrong password"}
            ).status_code
            == 401
        )
    paused = c.post("/auth/login", json={"email": "ada@example.com", "password": PASSWORD})
    assert paused.status_code == 429 and "paused" in paused.json()["detail"]
    assert any("paused" in m["subject"] for m in OUTBOX if m["to"] == "ada@example.com")
    # Someone else's account is not affected.
    signed_in("bob@example.com")


def test_two_factor_sign_in(service: AssistantService, hosted: None) -> None:
    ada = signed_in("ada@example.com")
    assert ada.post("/account/2fa/setup", json={"password": "wrong password"}).status_code == 403
    secret = ada.post("/account/2fa/setup", json={"password": PASSWORD}).json()["secret"]
    assert ada.post("/account/2fa/enable", json={"code": "000000"}).status_code == 400
    step = totp.current_step()
    codes = ada.post("/account/2fa/enable", json={"code": totp.code_at(secret, step)}).json()[
        "recovery_codes"
    ]
    assert len(codes) == 10
    assert ada.get("/auth/me").json()["user"]["two_factor"] is True
    assert ada.get("/account/security").json() == {"two_factor": True, "recovery_codes_left": 10}
    with service.store.read() as db:
        sealed = db.execute(
            "SELECT totp_secret FROM users WHERE email = 'ada@example.com'"
        ).fetchone()[0]
    assert secret not in sealed  # encrypted at rest

    c = client()
    first = c.post("/auth/login", json={"email": "ada@example.com", "password": PASSWORD})
    assert first.status_code == 200 and first.json()["two_factor_required"] is True
    assert "set-cookie" not in first.headers  # no session until the second step
    challenge = first.json()["challenge"]
    assert (
        c.post("/auth/login/2fa", json={"challenge": challenge, "code": "123456"}).status_code
        == 401
    )
    # The code used to switch it on cannot be replayed.
    assert (
        c.post(
            "/auth/login/2fa", json={"challenge": challenge, "code": totp.code_at(secret, step)}
        ).status_code
        == 401
    )
    done = c.post(
        "/auth/login/2fa", json={"challenge": challenge, "code": totp.code_at(secret, step + 1)}
    )
    assert done.status_code == 200 and "csrf_token" in done.json()
    assert (
        c.post(
            "/auth/login/2fa", json={"challenge": challenge, "code": totp.code_at(secret, step + 1)}
        ).status_code
        == 400
    )

    # A recovery code works once.
    again = client()
    challenge = again.post(
        "/auth/login", json={"email": "ada@example.com", "password": PASSWORD}
    ).json()["challenge"]
    assert (
        again.post("/auth/login/2fa", json={"challenge": challenge, "code": codes[0]}).status_code
        == 200
    )
    third = client()
    challenge = third.post(
        "/auth/login", json={"email": "ada@example.com", "password": PASSWORD}
    ).json()["challenge"]
    assert (
        third.post("/auth/login/2fa", json={"challenge": challenge, "code": codes[0]}).status_code
        == 401
    )

    # Switching it off needs the password and a code.
    assert (
        ada.post("/account/2fa/disable", json={"password": PASSWORD, "code": "000000"}).status_code
        == 403
    )
    assert (
        ada.post("/account/2fa/disable", json={"password": PASSWORD, "code": codes[1]}).status_code
        == 200
    )
    assert (
        "two_factor_required"
        not in client()
        .post("/auth/login", json={"email": "ada@example.com", "password": PASSWORD})
        .json()
    )


def test_admins_can_be_required_to_use_two_factor(
    service: AssistantService, hosted: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GATE_REQUIRE_ADMIN_2FA", "true")
    boss = signed_in("boss@example.com")
    assert boss.get("/auth/me").json()["user"]["is_admin"] is False
    assert boss.get("/assistant/stats").status_code == 403
    turn_on_two_factor(boss)
    assert boss.get("/auth/me").json()["user"]["is_admin"] is True
    assert boss.get("/assistant/stats").status_code == 200


def test_changing_the_email_needs_the_new_address_to_confirm(
    service: AssistantService, hosted: None
) -> None:
    ada = signed_in("ada@example.com")
    assert (
        ada.post(
            "/account/email",
            json={"new_email": "ada.new@example.com", "password": "wrong password"},
        ).status_code
        == 403
    )
    assert (
        ada.post(
            "/account/email", json={"new_email": "ada@example.com", "password": PASSWORD}
        ).status_code
        == 400
    )
    assert (
        ada.post(
            "/account/email", json={"new_email": "Ada.New@Example.com", "password": PASSWORD}
        ).status_code
        == 202
    )
    assert any("being changed" in m["subject"] for m in OUTBOX if m["to"] == "ada@example.com")
    token = last_link_token("ada.new@example.com")
    assert ada.get("/auth/me").json()["user"]["email"] == "ada@example.com"  # nothing changes yet
    c = client()
    assert (
        c.post("/auth/confirm-email", json={"token": token}).json()["email"]
        == "ada.new@example.com"
    )
    assert c.post("/auth/confirm-email", json={"token": token}).status_code == 400  # single use
    assert ada.get("/auth/me").status_code == 401  # every session was signed out
    assert (
        c.post(
            "/auth/login", json={"email": "ada.new@example.com", "password": PASSWORD}
        ).status_code
        == 200
    )


def test_an_email_change_to_a_taken_address_changes_nothing(
    service: AssistantService, hosted: None
) -> None:
    signed_in("bob@example.com")
    ada = signed_in("ada@example.com")
    reply = ada.post("/account/email", json={"new_email": "bob@example.com", "password": PASSWORD})
    assert reply.status_code == 202 and reply.json() == {"status": "check_email"}
    assert (
        "already has" in next(m for m in reversed(OUTBOX) if m["to"] == "bob@example.com")["body"]
    )
    assert ada.get("/auth/me").json()["user"]["email"] == "ada@example.com"


def test_hosted_responses_carry_security_headers(service: AssistantService, hosted: None) -> None:
    ada = signed_in("ada@example.com")
    me = ada.get("/auth/me")
    assert me.headers["x-content-type-options"] == "nosniff"
    assert me.headers["x-frame-options"] == "DENY"
    assert me.headers["cache-control"] == "no-store"
    assert "cache-control" not in client().get("/health").headers


def test_single_tenant_responses_are_unchanged(service: AssistantService) -> None:
    assert "x-frame-options" not in client().get("/health").headers


def test_openai_compatible_provider() -> None:
    body = OpenAICompatibleClient("https://api.example.com/v1", 30, "low")._body(
        "openai/gpt-oss-120b", [{"role": "user", "content": "hi"}], 100, 0.2, None, True
    )
    assert "reasoning_effort" not in body and "reasoning_format" not in body
    assert body["stream_options"] == {"include_usage": True}
    settings = AssistantSettings(providers=["openai"], openai_model="gpt-4o-mini")
    gateway = Gateway(settings, KeyPool.from_env({"OPENAI_API_KEY": "sk-" + "o" * 40}))
    assert gateway.provider_order() == ["openai"]
    assert gateway.model_for("openai", "answer") == "gpt-4o-mini"
    assert gateway.pool.acquire("openai") == "sk-" + "o" * 40


def test_hosted_mode_accepts_openai_but_keeps_the_guard_honest() -> None:
    env = {
        **GOOD_ENV,
        "ASSISTANT_PROVIDERS": "openai",
        "GROQ_API_KEY": "",
        "OPENAI_API_KEY": "sk-" + "o" * 40,
    }
    problems = hosted_config_problems(env)
    assert any("prompt-guard" in p for p in problems)
    assert hosted_config_problems({**env, "ASSISTANT_GUARD": "false"}) == []

"""Hosted mode (GATE_HOSTED=true): start-up checks, admin-only endpoints, proxy handling."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from assistant.hosted import check_hosted_config, client_address, hosted_config_problems
from assistant.service import AssistantService
from tests.hosted_utils import ADMIN_TOKEN, GOOD_ENV


def test_a_locked_down_config_has_no_problems() -> None:
    assert hosted_config_problems(GOOD_ENV) == []
    check_hosted_config(GOOD_ENV)  # does not raise
    check_hosted_config({"GATE_HOSTED": "false"})  # off: nothing is checked


@pytest.mark.parametrize(
    ("change", "complaint"),
    [
        ({"GATE_CORS_ORIGINS": ""}, "GATE_CORS_ORIGINS must list"),
        ({"GATE_CORS_ORIGINS": "*"}, "must not contain '*'"),
        ({"GATE_CORS_ORIGINS": "https://a.example.com,*"}, "must not contain '*'"),
        ({"GATE_ADMIN_TOKEN": "short"}, "GATE_ADMIN_TOKEN"),
        ({"ASSISTANT_PROVIDERS": "groq,ollama"}, "only name cloud providers"),
        ({"ASSISTANT_LOCAL_FALLBACK": "true"}, "never uses Ollama"),
        ({"GATE_LLM_PROVIDER": "ollama"}, "GATE_LLM_PROVIDER"),
        ({"GROQ_API_KEY": ""}, "API key is required"),
        ({"GATE_PUBLIC_URL": "http://gate.example.com"}, "GATE_PUBLIC_URL"),
        ({"GATE_ADMIN_EMAILS": ""}, "GATE_ADMIN_EMAILS"),
        ({"GATE_MAIL_BACKEND": "smtp"}, "GATE_SMTP_HOST"),
        ({"GATE_MAIL_BACKEND": "pigeon"}, "smtp or console"),
    ],
)
def test_hosted_mode_refuses_an_unsafe_config(change: dict[str, str], complaint: str) -> None:
    env = {**GOOD_ENV, **change}
    assert any(complaint in p for p in hosted_config_problems(env))
    with pytest.raises(RuntimeError, match="not safe to start"):
        check_hosted_config(env)


def test_the_app_will_not_start_hosted_without_explicit_origins(
    service: AssistantService, hosted: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from api.main import app

    monkeypatch.setenv("GATE_CORS_ORIGINS", "*")
    with pytest.raises(RuntimeError, match="GATE_CORS_ORIGINS"), TestClient(app):
        pass


def test_root_reports_whether_the_server_is_hosted(
    service: AssistantService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from api.main import app

    client = TestClient(app)
    assert client.get("/").json()["hosted"] is False
    monkeypatch.setenv("GATE_HOSTED", "true")
    assert client.get("/").json()["hosted"] is True


OPERATOR_CALLS = [
    ("post", "/run", {"task_id": "t", "question": "2+2?", "force_simulation": True}),
    ("get", "/history", None),
    ("delete", "/history", None),
    ("delete", "/assistant/cache", None),
    ("post", "/assistant/keys", {"provider": "groq", "key": "gsk_" + "k" * 40}),
    ("get", "/assistant/stats", None),
    ("get", "/assistant/status", None),
    ("get", "/metrics", None),
]


@pytest.mark.parametrize(("method", "path", "body"), OPERATOR_CALLS)
def test_operator_endpoints_are_admin_only_when_hosted(
    service: AssistantService, hosted: None, method: str, path: str, body: dict | None
) -> None:
    from api.main import app

    client = TestClient(app)  # its host is "testclient", which counts as local
    kwargs = {"json": body} if body is not None else {}
    refused = getattr(client, method)(path, **kwargs)
    assert refused.status_code == 403, refused.text
    wrong = getattr(client, method)(path, headers={"x-admin-token": "nope"}, **kwargs)
    assert wrong.status_code == 403


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [c for c in OPERATOR_CALLS if c[1] != "/assistant/keys"],
)
def test_the_admin_token_opens_operator_endpoints_when_hosted(
    service: AssistantService, hosted: None, method: str, path: str, body: dict | None
) -> None:
    from api.main import app

    client = TestClient(app)
    kwargs = {"json": body} if body is not None else {}
    allowed = getattr(client, method)(path, headers={"x-admin-token": ADMIN_TOKEN}, **kwargs)
    assert allowed.status_code == 200, allowed.text


def test_single_tenant_mode_keeps_the_local_operator_rule(service: AssistantService) -> None:
    from api.main import app

    client = TestClient(app)
    assert client.get("/history").status_code == 200
    assert client.delete("/assistant/cache").status_code == 200
    assert client.get("/assistant/stats").status_code == 200


def test_people_can_see_which_models_answer_but_not_the_keys(
    service: AssistantService, hosted: None
) -> None:
    from tests.hosted_utils import signed_in

    reply = signed_in("ada@example.com").get("/assistant/models")
    assert reply.status_code == 200 and set(reply.json()) == {"fast", "answer"}
    assert "gsk_" not in reply.text


def test_research_pages_stay_public_when_hosted(service: AssistantService, hosted: None) -> None:
    from api.main import app

    client = TestClient(app)
    for path in ("/health", "/models", "/tasks?split=val&limit=1", "/results/runs"):
        assert client.get(path).status_code == 200, path


# Client address behind a proxy

PROXY_ENV = {"GATE_HOSTED": "true", "GATE_TRUSTED_PROXIES": "10.0.0.0/8, 127.0.0.1"}


def test_forwarded_for_is_ignored_from_an_untrusted_peer() -> None:
    assert client_address("198.51.100.7", "1.2.3.4", PROXY_ENV) == "198.51.100.7"
    # Hosted with no proxies configured trusts nobody.
    assert client_address("198.51.100.7", "1.2.3.4", {"GATE_HOSTED": "true"}) == "198.51.100.7"


def test_forwarded_for_is_read_from_the_right_behind_a_trusted_proxy() -> None:
    # The client prepended a fake address; the proxy appended the real one.
    assert client_address("10.0.0.2", "6.6.6.6, 203.0.113.9", PROXY_ENV) == "203.0.113.9"
    # Two trusted hops (CDN -> proxy) are skipped.
    assert client_address("127.0.0.1", "203.0.113.9, 10.1.2.3", PROXY_ENV) == "203.0.113.9"
    assert client_address("10.0.0.2", "", PROXY_ENV) == "10.0.0.2"
    assert client_address("10.0.0.2", "not-an-ip", PROXY_ENV) == "not-an-ip"


def test_single_tenant_default_reads_forwarded_for_as_before() -> None:
    assert client_address("127.0.0.1", "203.0.113.9, 10.0.0.1", {}) == "203.0.113.9"
    assert client_address("127.0.0.1", None, {}) == "127.0.0.1"


def test_a_forged_header_cannot_dodge_the_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import FastAPI

    from api.security import SecurityMiddleware

    monkeypatch.setenv("GATE_HOSTED", "true")
    monkeypatch.setenv("GATE_RATE_LIMIT_PER_MINUTE", "1")
    tiny = FastAPI()

    @tiny.post("/chat")
    def chat_stub() -> dict:
        return {"ok": True}

    tiny.add_middleware(SecurityMiddleware)
    client = TestClient(tiny)
    assert client.post("/chat", headers={"x-forwarded-for": "1.1.1.1"}).status_code == 200
    assert client.post("/chat", headers={"x-forwarded-for": "2.2.2.2"}).status_code == 429

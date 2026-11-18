"""Shared pytest fixtures."""

from __future__ import annotations

import urllib.request

import pytest


@pytest.fixture(autouse=True)
def _no_real_llm_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any unmocked HTTP call so the suite never spends real tokens.

    Tests that exercise the providers patch ``urllib.request.urlopen``
    themselves, which takes precedence over this fixture.
    """

    def _blocked(*args: object, **kwargs: object) -> None:
        raise OSError("Network access is disabled in tests; inject an llm_caller instead.")

    monkeypatch.setattr(urllib.request, "urlopen", _blocked)


@pytest.fixture(autouse=True)
def _no_background_warm_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """The server loads spaCy in the background at start-up; tests do not need that."""
    monkeypatch.setenv("ASSISTANT_WARMUP", "0")


@pytest.fixture(autouse=True)
def _isolated_api_history(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Keep each test's API run history in its own file, never in logs/."""
    import sys

    api_main = sys.modules.get("api.main")
    if api_main is not None:
        from api.history import RunHistory

        monkeypatch.setattr(api_main, "_HISTORY", RunHistory(tmp_path / "history.jsonl"))


@pytest.fixture()
def service(tmp_path):
    """A chat service on the offline mock provider, installed as the API's service."""
    from assistant.config import AssistantSettings
    from assistant.gates import GateSet
    from assistant.service import AssistantService, set_service

    svc = AssistantService(
        AssistantSettings(
            db_path=tmp_path / "a.db", env_path=tmp_path / ".env", providers=["mock"]
        ),
        gates=GateSet(),
    )
    set_service(svc)
    yield svc
    set_service(None)


@pytest.fixture()
def hosted(monkeypatch: pytest.MonkeyPatch) -> None:
    """GATE_HOSTED=true with a locked-down configuration and generous auth limits."""
    from api import auth_routes
    from assistant.mailer import OUTBOX
    from assistant.security import RateLimiter
    from tests.hosted_utils import GOOD_ENV

    for name, value in GOOD_ENV.items():
        monkeypatch.setenv(name, value)
    for name in ("GATE_API_TOKEN", "GATE_TRUSTED_PROXIES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(auth_routes, "_by_address", RateLimiter(10_000))
    monkeypatch.setattr(auth_routes, "_by_email", RateLimiter(10_000))
    OUTBOX.clear()
    _relax_address_limit(monkeypatch)


def _relax_address_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every TestClient comes from one address, so give the per-address limit room."""
    from api.main import app
    from api.security import SecurityMiddleware
    from assistant.security import RateLimiter

    if app.middleware_stack is None:
        app.middleware_stack = app.build_middleware_stack()
    node = app.middleware_stack
    while node is not None:
        if isinstance(node, SecurityMiddleware):
            monkeypatch.setattr(node, "limiter", RateLimiter(10_000))
        node = getattr(node, "app", None)

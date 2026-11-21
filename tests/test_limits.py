"""Hosted mode: people's own API keys, plan token caps, the global cap, rate and sandbox limits."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi import HTTPException

from api import limits
from assistant.hosted import hosted_config_problems
from assistant.plans import effective_plan, plan_book
from assistant.sandbox import SandboxLimits, run_python_for
from assistant.security import RateLimiter
from assistant.service import AssistantService
from assistant.usage import route_of
from tests.hosted_utils import GOOD_ENV, signed_in

OWN_KEY = "gsk_" + "o" * 40


@pytest.fixture(autouse=True)
def _fresh_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(limits, "_per_user", RateLimiter(10_000))


def user_id(client) -> str:
    return client.get("/auth/me").json()["user"]["id"]


def chat(client, message: str = "Who wrote Middlemarch?"):
    return client.post("/chat", json={"message": message})


def small_plans(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **free) -> None:
    spec = {"messages_per_day": 30, "documents": 3, "daily_tokens": 40000, **free}
    lines = ["default_plan: free", "plans:", "  free:"]
    lines += [f"    {k}: {v}" for k, v in spec.items()]
    lines += [
        "  pro:",
        "    messages_per_day: 400",
        "    documents: 100",
        "    daily_tokens: 1000000",
    ]
    path = tmp_path / "plans.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setenv("GATE_PLANS_PATH", str(path))


def test_the_plans_file_loads_and_paid_plans_lapse() -> None:
    book = plan_book()
    assert set(book.plans) == {"free", "pro"} and book.default == "free"
    assert book.plans["pro"].daily_tokens > book.plans["free"].daily_tokens
    now = time.time()
    assert effective_plan({"plan": "pro", "paid_until": now + 60}).id == "pro"
    assert effective_plan({"plan": "pro", "paid_until": now - 60}).id == "free"
    assert effective_plan({"plan": "pro", "paid_until": None}).id == "free"
    assert effective_plan({"plan": "free", "paid_until": None}).id == "free"


def test_hosted_mode_needs_a_valid_encryption_key() -> None:
    assert any(
        "GATE_ENCRYPTION_KEY" in p
        for p in hosted_config_problems({**GOOD_ENV, "GATE_ENCRYPTION_KEY": ""})
    )
    bad = {**GOOD_ENV, "GATE_ENCRYPTION_KEY": "not-a-key"}
    assert any("not a valid Fernet key" in p for p in hosted_config_problems(bad))


def test_own_keys_are_checked_encrypted_and_never_returned(
    service: AssistantService, hosted: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from assistant.llm import ProviderError

    ada = signed_in("ada@example.com")
    monkeypatch.setattr(service.gateway, "validate_key", lambda provider, key: ["model"])
    saved = ada.put("/account/keys", json={"provider": "groq", "key": OWN_KEY})
    assert saved.status_code == 200 and OWN_KEY not in saved.text
    listing = ada.get("/account/keys")
    assert [k["provider"] for k in listing.json()["keys"]] == [
        "groq"
    ] and OWN_KEY not in listing.text

    with service.store.read() as db:
        stored = db.execute("SELECT ciphertext FROM user_keys").fetchone()[0]
    assert OWN_KEY not in stored and service.user_keys.keys(user_id(ada)) == {"groq": OWN_KEY}

    gateway, own = service.gateway_for(user_id(ada))
    assert own and gateway is not service.gateway
    assert gateway.pool.acquire("groq") == OWN_KEY and gateway.provider_order() == ["groq"]
    assert service.gateway_for(None) == (service.gateway, False)

    def reject(provider: str, key: str) -> list[str]:
        raise ProviderError("invalid_key", "401")

    monkeypatch.setattr(service.gateway, "validate_key", reject)
    refused = ada.put("/account/keys", json={"provider": "gemini", "key": "AIza" + "x" * 35})
    assert refused.status_code == 400 and "rejected" in refused.json()["detail"]

    # Nobody else sees or uses Ada's key.
    bob = signed_in("bob@example.com")
    assert bob.get("/account/keys").json()["keys"] == []
    assert service.gateway_for(user_id(bob))[1] is False

    assert ada.delete("/account/keys/groq").json()["keys"] == []
    assert ada.delete("/account/keys/groq").status_code == 404


def test_account_keys_are_hosted_only(service: AssistantService) -> None:
    from fastapi.testclient import TestClient

    from api.main import app

    assert TestClient(app).get("/account/keys").status_code == 404


def test_the_plan_s_daily_tokens_stop_further_turns(
    service: AssistantService, hosted: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    small_plans(tmp_path, monkeypatch, daily_tokens=1)
    ada = signed_in("ada@example.com")
    first = chat(ada)
    assert first.status_code == 200 and '"type": "done"' in first.text
    today = service.usage.today(user_id(ada))
    assert today["messages"] == 1 and today["tokens"] > 0 and today["own_key_tokens"] == 0
    stopped = chat(ada, "And who wrote Persuasion?")
    assert stopped.status_code == 429 and "tokens" in stopped.json()["detail"]
    # Another person still has their own allowance.
    assert chat(signed_in("bob@example.com")).status_code == 200


def test_own_key_turns_do_not_use_the_allowance(
    service: AssistantService, hosted: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    small_plans(tmp_path, monkeypatch, daily_tokens=1)
    ada = signed_in("ada@example.com")
    # The mock provider answers; pretend it is Ada's own key.
    monkeypatch.setattr(service, "gateway_for", lambda uid: (service.gateway, True))
    for _ in range(2):
        assert chat(ada).status_code == 200
    today = service.usage.today(user_id(ada))
    assert today["tokens"] == 0 and today["own_key_tokens"] > 0 and today["messages"] == 2


def test_the_global_cap_is_the_operator_s_safety_switch(
    service: AssistantService, hosted: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GATE_GLOBAL_DAILY_TOKENS", "1")
    assert chat(signed_in("ada@example.com")).status_code == 200
    blocked = chat(signed_in("bob@example.com"))
    assert blocked.status_code == 503 and "daily capacity" in blocked.json()["detail"]


def test_each_person_has_a_rate_limit(
    service: AssistantService, hosted: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(limits, "_per_user", RateLimiter(1))
    ada = signed_in("ada@example.com")
    assert chat(ada).status_code == 200
    assert chat(ada).status_code == 429
    assert chat(signed_in("bob@example.com")).status_code == 200


def test_answers_in_progress_are_capped_per_person(service: AssistantService, hosted: None) -> None:
    ada = signed_in("ada@example.com")
    held = limits.admit_turn(service, user_id(ada))  # one answer still streaming
    with pytest.raises(HTTPException) as busy:
        limits.admit_turn(service, user_id(ada))
    assert busy.value.status_code == 429
    held.release()
    limits.admit_turn(service, user_id(ada)).release()


def test_sandbox_runs_are_rationed_per_person_and_overall(monkeypatch: pytest.MonkeyPatch) -> None:
    slots = SandboxLimits(per_user=1, total=1)
    assert slots.acquire("ada") is None
    assert "already have code running" in (slots.acquire("ada") or "")
    assert "busy" in (slots.acquire("bob", wait=0.01) or "")
    slots.release("ada")
    assert slots.acquire("bob") is None
    slots.release("bob")

    import assistant.sandbox as sandbox

    monkeypatch.setattr(sandbox, "run_python", lambda code: "exit code 0\n4")
    assert run_python_for("ada", "print(2+2)", slots) == "exit code 0\n4"
    assert slots.acquire("ada") is None
    assert "unavailable right now" in run_python_for("ada", "print(1)", slots)


def test_turn_routes_are_classified() -> None:
    assert route_of({"cached": True, "decision": "ESCALATE"}) == "cached"
    assert route_of({"gate": "triage", "decision": "STOP"}) == "small_talk"
    assert route_of({"gate": "learned", "decision": "ESCALATE"}) == "escalated"
    assert route_of({"gate": "learned", "decision": "STOP"}) == "direct"


def test_deleting_an_account_removes_its_keys_and_usage(
    service: AssistantService, monkeypatch
) -> None:
    monkeypatch.setenv("GATE_ENCRYPTION_KEY", GOOD_ENV["GATE_ENCRYPTION_KEY"])
    user, _ = service.accounts.create("ada@example.com", "a long enough password")
    service.user_keys.put(user["id"], "groq", OWN_KEY)
    service.usage.record(user["id"], {"total_tokens": 5, "decision": "STOP"}, own_key=False)
    service.accounts.delete(user["id"])
    assert service.user_keys.listing(user["id"]) == []
    assert service.usage.today(user["id"])["messages"] == 0

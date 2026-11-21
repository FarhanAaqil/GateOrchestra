"""Hosted mode: the usage page's numbers and the plans listing."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from assistant.service import AssistantService
from tests.hosted_utils import signed_in


def test_usage_counts_turns_against_the_plan(service: AssistantService, hosted: None) -> None:
    ada = signed_in("ada@example.com")
    assert ada.post("/chat", json={"message": "Who wrote Middlemarch?"}).status_code == 200
    assert ada.post("/chat", json={"message": "hi"}).status_code == 200
    usage = ada.get("/account/usage").json()
    assert usage["plan"]["id"] == "free"
    assert usage["today"]["messages"] == 2 and usage["month"]["messages"] == 2
    assert usage["today"]["small_talk"] == 1
    assert usage["today"]["direct"] + usage["today"]["escalated"] == 1
    assert usage["today"]["tokens"] > 0 and usage["own_key"] is False
    assert usage["documents"] == 0
    # Someone else's page starts at zero.
    assert signed_in("bob@example.com").get("/account/usage").json()["month"]["messages"] == 0


def test_the_saved_estimate_waits_for_real_escalations(
    service: AssistantService, hosted: None, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from api import account_routes

    monkeypatch.setattr(account_routes, "CHAT_EVAL_SUMMARY", tmp_path / "missing.json")
    ada = signed_in("ada@example.com")
    estimate = ada.get("/account/usage").json()["estimate"]
    assert estimate["available"] is False and estimate["saved_tokens"] is None
    assert "Estimate only" in estimate["method"]

    for agent_tokens in (800, 1000, 1200, 900, 1100):
        service.store.log_event(
            "chat", {"decision": "ESCALATE", "cached": False, "tokens": {"agent": agent_tokens}}
        )
    uid = ada.get("/auth/me").json()["user"]["id"]
    for _ in range(3):
        service.usage.record(
            uid, {"gate": "learned", "decision": "STOP", "total_tokens": 50}, own_key=False
        )
    estimate = ada.get("/account/usage").json()["estimate"]
    assert estimate["available"] is True and estimate["sample_size"] == 5
    assert estimate["per_escalation"] == 1000 and estimate["saved_tokens"] == 3000


def test_plans_are_listed_publicly(service: AssistantService, hosted: None) -> None:
    from api.main import app

    plans = TestClient(app).get("/plans").json()
    assert plans["default"] == "free"
    assert [p["id"] for p in plans["plans"]] == ["free", "pro"]
    assert plans["plans"][0]["price_usd_month"] == 0


def test_usage_is_hosted_only(service: AssistantService) -> None:
    from api.main import app

    assert TestClient(app).get("/account/usage").status_code == 404


def test_the_estimate_uses_the_measured_evaluation_cost_until_the_server_has_its_own(
    service: AssistantService, hosted: None
) -> None:
    from api.account_routes import measured_research_cost

    measured = measured_research_cost()
    assert measured and measured["per_escalation"] > 0 and measured["sample_size"] == 80
    ada = signed_in("ada@example.com")
    estimate = ada.get("/account/usage").json()["estimate"]
    assert estimate["available"] is True and estimate["source"] == "evaluation"
    assert estimate["per_escalation"] == measured["per_escalation"]
    assert "80-question evaluation" in estimate["method"]

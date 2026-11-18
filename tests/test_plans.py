"""Hosted mode: plan limits on answers and documents, the payment link and admin plan changes."""

from __future__ import annotations

import base64
import time
from pathlib import Path

import pytest

from api import limits
from assistant.hosted import hosted_config_problems
from assistant.security import RateLimiter
from assistant.service import AssistantService
from tests.hosted_utils import GOOD_ENV, signed_in
from tests.test_limits import small_plans

DAY = 86400


@pytest.fixture(autouse=True)
def _fresh_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(limits, "_per_user", RateLimiter(10_000))
    monkeypatch.setattr(limits, "_running", type(limits._running)(int))


def uid(client) -> str:
    return client.get("/auth/me").json()["user"]["id"]


def upload(client, name: str):
    body = {"name": name, "content_base64": base64.b64encode(b"some notes").decode()}
    return client.post("/documents", json=body)


def test_answers_per_day_are_limited_by_plan(
    service: AssistantService, hosted: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    small_plans(tmp_path, monkeypatch, messages_per_day=1)
    ada = signed_in("ada@example.com")
    assert ada.post("/chat", json={"message": "hi"}).status_code == 200
    stopped = ada.post("/chat", json={"message": "hello again"})
    assert stopped.status_code == 429 and "answers" in stopped.json()["detail"]
    # Turns on the person's own key still count as answers.
    monkeypatch.setattr(service, "gateway_for", lambda user_id: (service.gateway, True))
    assert ada.post("/chat", json={"message": "hello again"}).status_code == 429


def test_stored_documents_are_limited_by_plan(
    service: AssistantService, hosted: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    small_plans(tmp_path, monkeypatch, documents=1)
    ada = signed_in("ada@example.com")
    first = upload(ada, "a.txt")
    assert first.status_code == 200
    refused = upload(ada, "b.txt")
    assert refused.status_code == 403 and "stores up to 1" in refused.json()["detail"]
    assert ada.delete(f"/documents/{first.json()['id']}").status_code == 200
    assert upload(ada, "b.txt").status_code == 200  # room again


def test_upgrade_link_is_filled_in_for_the_person(
    service: AssistantService, hosted: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    ada = signed_in("ada+test@example.com")
    assert ada.get("/account/upgrade").status_code == 404  # not configured
    assert ada.get("/plans").json()["payments"] is False
    monkeypatch.setenv("GATE_PAYMENT_LINK", "https://pay.example.com/l/pro?prefilled_email={email}")
    assert ada.get("/plans").json()["payments"] is True
    assert ada.get("/account/upgrade").json()["url"] == (
        "https://pay.example.com/l/pro?prefilled_email=ada%2Btest%40example.com"
    )
    bad = {**GOOD_ENV, "GATE_PAYMENT_LINK": "http://pay.example.com"}
    assert any("GATE_PAYMENT_LINK" in p for p in hosted_config_problems(bad))


def test_pay_upgrade_extend_and_lapse(service: AssistantService, hosted: None) -> None:
    boss, ada = signed_in("boss@example.com"), signed_in("ada@example.com")
    ada_id = uid(ada)

    found = boss.get("/admin/users", params={"q": "ada@"}).json()["users"]
    assert [u["email"] for u in found] == ["ada@example.com"]
    assert found[0]["effective_plan"] == "free" and "password_hash" not in found[0]

    upgraded = boss.post(f"/admin/users/{ada_id}/plan", json={"plan": "pro", "days": 31}).json()
    assert upgraded["user"]["plan"] == "pro" and upgraded["user"]["effective_plan"] == "pro"
    first_end = upgraded["user"]["paid_until"]
    assert first_end == pytest.approx(time.time() + 31 * DAY, abs=60)
    assert ada.get("/account/usage").json()["plan"]["id"] == "pro"

    # Paying again extends from the end of the current period.
    again = boss.post(f"/admin/users/{ada_id}/plan", json={"plan": "pro", "days": 31}).json()
    assert again["user"]["paid_until"] == pytest.approx(first_end + 31 * DAY, abs=1)

    # When the period ends Ada is on the free plan again, with nothing to run.
    service.accounts.set_plan(ada_id, "pro", time.time() - 1)
    usage = ada.get("/account/usage").json()
    assert usage["plan"]["id"] == "free" and usage["paid_until"] is None
    assert (
        boss.get("/admin/users", params={"q": "ada@"}).json()["users"][0]["effective_plan"]
        == "free"
    )

    # Back to free on purpose clears the paid period.
    down = boss.post(f"/admin/users/{ada_id}/plan", json={"plan": "free"}).json()
    assert down["user"]["plan"] == "free" and down["user"]["paid_until"] is None

    assert boss.post(f"/admin/users/{ada_id}/plan", json={"plan": "gold"}).status_code == 400
    assert boss.post("/admin/users/nobody/plan", json={"plan": "pro"}).status_code == 404


def test_only_admins_manage_plans(service: AssistantService, hosted: None) -> None:
    ada = signed_in("ada@example.com")
    assert ada.get("/admin/users").status_code == 403
    assert (
        ada.post(f"/admin/users/{uid(ada)}/plan", json={"plan": "pro", "days": 365}).status_code
        == 403
    )
    assert ada.get("/account/usage").json()["plan"]["id"] == "free"


def test_plan_admin_is_hosted_only(service: AssistantService) -> None:
    from fastapi.testclient import TestClient

    from api.main import app

    assert TestClient(app).get("/admin/users").status_code == 404

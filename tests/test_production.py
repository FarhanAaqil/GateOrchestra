"""Hosted mode in production: backups, request logs, error scrubbing, export and deletion."""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from pathlib import Path

import pytest

from api.observability import init_error_tracking, scrub_event
from assistant.service import AssistantService
from assistant.store import Store
from scripts import backup_db
from tests.hosted_utils import GOOD_ENV, PASSWORD, signed_in


def test_backups_are_consistent_restorable_and_pruned(tmp_path: Path) -> None:
    store = Store(tmp_path / "assistant.db")
    view = store.scoped("ada")
    view.add_message(view.create_conversation("hello"), "user", "hi")
    out = tmp_path / "backups"

    made = backup_db.backup(store.path, out, now=time.time())
    counts = backup_db.verify(made)
    assert counts["conversations"] == 1 and counts["messages"] == 1

    old = out / "assistant-20000101-000000.db.gz"
    old.write_bytes(made.read_bytes())
    os.utime(old, (0, 0))
    assert backup_db.prune(out, keep_days=14) == [old]
    assert made.exists()  # the newest backup is always kept

    assert backup_db.main(["--db", str(store.path), "--out", str(out), "--keep", "14"]) == 0
    assert backup_db.main(["--verify", str(made)]) == 0


def test_a_broken_backup_fails_loudly(tmp_path: Path) -> None:
    broken = tmp_path / "assistant-broken.db.gz"
    broken.write_bytes(b"not a gzip file")
    assert backup_db.main(["--verify", str(broken)]) == 1
    assert backup_db.main(["--db", str(tmp_path / "missing.db"), "--out", str(tmp_path)]) == 1


def test_requests_are_logged_with_ids_and_never_content(
    service: AssistantService, hosted: None, caplog: pytest.LogCaptureFixture
) -> None:
    ada = signed_in("ada@example.com")
    ada_id = ada.get("/auth/me").json()["user"]["id"]
    caplog.set_level(logging.INFO, logger="gateorchestra.access")
    caplog.clear()
    reply = ada.post(
        "/chat?secret=query-value",
        json={"message": "my private question about ZEBRA"},
        headers={"X-Request-ID": "req-abc-12345"},
    )
    assert reply.headers["x-request-id"] == "req-abc-12345"
    line = json.loads(caplog.records[-1].getMessage())
    assert line["request_id"] == "req-abc-12345" and line["user_id"] == ada_id
    assert line["path"] == "/chat" and line["status"] == 200 and line["method"] == "POST"
    text = caplog.text
    assert "ZEBRA" not in text and "query-value" not in text and "csrf" not in text.lower()

    # A malformed id is replaced, and anonymous requests are logged without a user.
    caplog.clear()
    anon = ada.__class__(ada.app)
    reply = anon.get("/health", headers={"X-Request-ID": "bad id with spaces"})
    assert len(reply.headers["x-request-id"]) == 32
    assert json.loads(caplog.records[-1].getMessage())["user_id"] is None


def test_request_logs_are_off_in_single_tenant_mode(
    service: AssistantService, caplog: pytest.LogCaptureFixture
) -> None:
    from fastapi.testclient import TestClient

    from api.main import app

    caplog.set_level(logging.INFO, logger="gateorchestra.access")
    reply = TestClient(app).get("/health")
    assert "x-request-id" not in reply.headers and not caplog.records


def test_error_reports_carry_no_request_data(monkeypatch: pytest.MonkeyPatch) -> None:
    event = {
        "request": {"method": "POST", "url": "https://x/chat", "data": "secret", "cookies": "c"},
        "user": {"email": "ada@example.com"},
        "breadcrumbs": {"values": [{"message": "secret"}]},
        "exception": {"values": [{"stacktrace": {"frames": [{"vars": {"key": "gsk_..."}}]}}]},
    }
    clean = scrub_event(event)
    assert clean["request"] == {"method": "POST"} and "user" not in clean
    assert "breadcrumbs" not in clean
    assert "vars" not in clean["exception"]["values"][0]["stacktrace"]["frames"][0]
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    assert init_error_tracking() is False


def test_export_holds_only_the_person_s_own_data(service: AssistantService, hosted: None) -> None:
    ada, bob = signed_in("ada@example.com"), signed_in("bob@example.com")
    assert ada.post("/chat", json={"message": "Who wrote Middlemarch?"}).status_code == 200
    assert bob.post("/chat", json={"message": "Bob's own question"}).status_code == 200
    exported = ada.get("/account/export")
    assert "attachment" in exported.headers["content-disposition"]
    data = exported.json()
    assert data["account"]["email"] == "ada@example.com" and "password_hash" not in exported.text
    assert [m["content"] for m in data["conversations"][0]["messages"]][
        0
    ] == "Who wrote Middlemarch?"
    assert "Bob's own question" not in exported.text


def test_deleting_an_account_removes_everything_of_it(
    service: AssistantService, hosted: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    ada, bob = signed_in("ada@example.com"), signed_in("bob@example.com")
    ada_id = ada.get("/auth/me").json()["user"]["id"]
    monkeypatch.setattr(service.gateway, "validate_key", lambda provider, key: ["model"])
    assert (
        ada.put("/account/keys", json={"provider": "groq", "key": "gsk_" + "d" * 40}).status_code
        == 200
    )
    body = {"name": "notes.txt", "content_base64": base64.b64encode(b"private notes").decode()}
    assert ada.post("/documents", json=body).status_code == 200
    question = "Who wrote Middlemarch?"
    assert ada.post("/chat", json={"message": question}).status_code == 200
    assert bob.post("/chat", json={"message": question}).status_code == 200

    assert ada.request("DELETE", "/account", json={"password": "wrong password"}).status_code == 403
    gone = ada.request("DELETE", "/account", json={"password": PASSWORD})
    assert gone.status_code == 200

    with service.store.read() as db:
        for table in ("conversations", "messages", "documents", "chunks", "cache", "feedback"):
            left = db.execute(
                f"SELECT COUNT(*) FROM {table} WHERE user_id = ?", (ada_id,)
            ).fetchone()[0]
            assert left == 0, table
        for table in ("users", "sessions", "user_keys", "usage_daily", "email_tokens"):
            column = "id" if table == "users" else "user_id"
            left = db.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {column} = ?", (ada_id,)
            ).fetchone()[0]
            assert left == 0, table
    assert ada.get("/auth/me").status_code == 401
    login = ada.post("/auth/login", json={"email": "ada@example.com", "password": PASSWORD})
    assert login.status_code == 401
    # Bob keeps his conversation and his cached answer.
    assert len(bob.get("/conversations").json()["conversations"]) == 1
    assert service.store.scoped(bob.get("/auth/me").json()["user"]["id"]).cache_entries(3600)


def test_changing_the_password_keeps_this_browser_and_ends_the_others(
    service: AssistantService, hosted: None
) -> None:
    here = signed_in("ada@example.com")
    elsewhere = signed_in("ada@example.com", verify=False)  # same account, second browser
    new = "an even longer new password"
    assert (
        here.post("/account/password", json={"current": "nope" * 3, "new": new}).status_code == 403
    )
    assert (
        here.post("/account/password", json={"current": PASSWORD, "new": "short"}).status_code
        == 400
    )
    changed = here.post("/account/password", json={"current": PASSWORD, "new": new})
    assert changed.status_code == 200
    here.headers["X-CSRF-Token"] = changed.json()["csrf_token"]
    assert here.get("/auth/me").status_code == 200
    assert elsewhere.get("/auth/me").status_code == 401


def test_hosted_settings_are_documented() -> None:
    root = Path(__file__).resolve().parents[1]
    example = (root / ".env.example").read_text(encoding="utf-8")
    for name in GOOD_ENV:
        assert f"{name}=" in example, name
    assert (root / "deploy" / "HOSTED.md").exists()

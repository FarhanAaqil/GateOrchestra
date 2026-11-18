"""Hosted mode keeps every user's rows to themselves: user B never sees user A's data."""

from __future__ import annotations

import base64
import json
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from assistant.service import AssistantService
from assistant.store import Store
from tests.hosted_utils import signed_in

SECRET = "The vault code for project Nightingale is 4471."


def turn(client: TestClient, message: str, **extra) -> list[dict]:
    reply = client.post("/chat", json={"message": message, **extra})
    assert reply.status_code == 200, reply.text
    return [json.loads(line[6:]) for line in reply.text.splitlines() if line.startswith("data: ")]


def upload(client: TestClient, name: str, text: str) -> dict:
    body = {"name": name, "content_base64": base64.b64encode(text.encode()).decode()}
    reply = client.post("/documents", json=body)
    assert reply.status_code == 200, reply.text
    return reply.json()


def test_user_b_cannot_reach_anything_of_user_a(service: AssistantService, hosted: None) -> None:
    ada, bob = signed_in("ada@example.com"), signed_in("bob@example.com")

    doc = upload(ada, "nightingale.txt", SECRET)
    events = turn(ada, "What is the vault code?", document_ids=[doc["id"]])
    done = events[-1]
    assert done["type"] == "done"
    cid, mid = done["conversation_id"], done["message_id"]

    # Listing: Bob has nothing.
    assert bob.get("/conversations").json()["conversations"] == []
    assert bob.get("/documents").json()["documents"] == []

    # Reading, renaming, deleting and rating Ada's rows: all "not found".
    assert bob.get(f"/conversations/{cid}").status_code == 404
    assert bob.patch(f"/conversations/{cid}", json={"title": "mine now"}).status_code == 404
    assert bob.delete(f"/conversations/{cid}").status_code == 404
    assert bob.post(f"/messages/{mid}/feedback", json={"rating": -1}).status_code == 404
    assert bob.delete(f"/documents/{doc['id']}").status_code == 404

    # Ada's rows are untouched.
    mine = ada.get(f"/conversations/{cid}").json()
    assert mine["title"] != "mine now" and len(mine["messages"]) == 2
    assert [d["id"] for d in ada.get("/documents").json()["documents"]] == [doc["id"]]
    assert ada.post(f"/messages/{mid}/feedback", json={"rating": 1}).status_code == 200


def test_document_search_only_reads_the_askers_own_chunks(
    service: AssistantService, hosted: None
) -> None:
    ada, bob = signed_in("ada@example.com"), signed_in("bob@example.com")
    doc = upload(ada, "nightingale.txt", SECRET)
    own = turn(ada, "What is the vault code for Nightingale?", document_ids=[doc["id"]])
    assert next(e for e in own if e["type"] == "gate")["features"]["has_context"] is True

    # Bob names Ada's document id and also asks over "all my documents".
    for ids in ([doc["id"]], []):
        events = turn(bob, "What is the vault code for Nightingale?", document_ids=ids)
        gate = next(e for e in events if e["type"] == "gate")
        assert gate["features"]["has_context"] is False, ids
        assert all("4471" not in json.dumps(e) for e in events)

    bob_store = service.store.scoped(
        bob.get("/auth/me").json()["user"]["id"],
    )
    assert bob_store.all_chunks() == [] and bob_store.all_chunks([doc["id"]]) == []
    assert service.retrieve_context("vault code", [doc["id"]], store=bob_store) is None
    tool = service.document_tool([doc["id"]], bob_store)
    assert "Nothing in the documents" in tool.run({"query": "vault code"})


def test_a_conversation_id_from_someone_else_starts_a_new_chat(
    service: AssistantService, hosted: None
) -> None:
    ada, bob = signed_in("ada@example.com"), signed_in("bob@example.com")
    ada_cid = turn(ada, "Remember: my favourite colour is teal.")[-1]["conversation_id"]

    done = turn(bob, "What is my favourite colour?", conversation_id=ada_cid)[-1]
    assert done["conversation_id"] != ada_cid
    assert len(ada.get(f"/conversations/{ada_cid}").json()["messages"]) == 2


def test_user_b_is_never_served_user_a_s_cached_answer(
    service: AssistantService, hosted: None
) -> None:
    ada, bob = signed_in("ada@example.com"), signed_in("bob@example.com")
    question = "Who wrote the novel Middlemarch?"

    assert turn(ada, question)[-1]["meta"]["cached"] is False
    assert turn(ada, question)[-1]["meta"]["cached"] is True  # Ada's own cache works
    bob_first = turn(bob, question)[-1]
    assert bob_first["meta"]["cached"] is False
    assert bob_first["meta"]["total_tokens"] > 0

    rows = service.store.cache_entries(3600)
    assert len(rows) == 2  # one entry each, not one shared


def test_single_tenant_mode_still_shares_one_history(service: AssistantService) -> None:
    from api.main import app

    client = TestClient(app)
    events = turn(client, "Who painted the Mona Lisa?")
    cid = events[-1]["conversation_id"]
    assert client.get(f"/conversations/{cid}").status_code == 200
    assert [c["id"] for c in client.get("/conversations").json()["conversations"]] == [cid]
    assert turn(client, "Who painted the Mona Lisa?")[-1]["meta"]["cached"] is True


def test_old_databases_gain_user_columns_and_keep_their_rows(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as db:
        db.executescript(
            "CREATE TABLE conversations (id TEXT PRIMARY KEY, title TEXT NOT NULL, "
            "created_at REAL NOT NULL, updated_at REAL NOT NULL);"
            "INSERT INTO conversations VALUES ('c1', 'from before', 1, 1);"
        )
    store = Store(path)
    assert [c["id"] for c in store.scoped(None).list_conversations()] == ["c1"]
    assert store.scoped("someone").list_conversations() == []
    Store(path)  # migrating twice is harmless


def test_export_and_wipe_cover_only_one_user(tmp_path: Path) -> None:
    store = Store(tmp_path / "a.db")
    ada, bob = store.scoped("ada"), store.scoped("bob")
    for view in (ada, bob):
        cid = view.create_conversation("hello")
        mid = view.add_message(cid, "assistant", "hi")
        view.set_feedback(mid, 1, "learned")
        view.add_document("notes.txt", ["some text"])
        view.cache_put("k", "q", "a", {})
        view.log_event("chat", {"total_tokens": 3})

    exported = ada.export()
    assert len(exported["conversations"]) == 1 and exported["conversations"][0]["messages"]
    assert [d["name"] for d in exported["documents"]] == ["notes.txt"]

    ada.delete_everything()
    assert ada.export() == {"conversations": [], "documents": []}
    assert ada.cache_get("k", 60) is None and ada.events("chat") == []
    assert len(bob.list_conversations()) == 1 and bob.cache_get("k", 60) is not None
    assert len(store.events("chat")) == 2  # the operator's totals keep the anonymised event
    for view in (store, store.scoped(None)):
        try:
            view.delete_everything()
        except ValueError:
            continue
        raise AssertionError("an unscoped wipe must be refused")

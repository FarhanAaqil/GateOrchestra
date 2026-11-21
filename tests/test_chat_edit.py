"""Editing an earlier question replaces it and everything after it, only once the new answer exists."""

from __future__ import annotations

import pytest

from assistant.llm import ProvidersExhausted
from assistant.pipeline import ChatPipeline
from assistant.service import AssistantService


def ask(service: AssistantService, text: str, cid: str | None = None, **kw) -> list[dict]:
    return list(ChatPipeline(service).run(text, cid, **kw))


def conversation(service: AssistantService) -> tuple[str, list[dict]]:
    cid = ask(service, "Who wrote Middlemarch?")[-1]["conversation_id"]
    ask(service, "When was it published?", cid)
    ask(service, "Who wrote Persuasion?", cid)
    return cid, service.store.messages(cid)


def test_editing_an_earlier_question_replaces_it_and_what_followed(
    service: AssistantService,
) -> None:
    cid, rows = conversation(service)
    second_question = rows[2]["id"]
    events = ask(
        service, "In which year was Middlemarch first published?", cid, edit_from=second_question
    )
    assert events[-1]["type"] == "done"
    after = service.store.messages(cid)
    assert [r["content"] for r in after if r["role"] == "user"] == [
        "Who wrote Middlemarch?",
        "In which year was Middlemarch first published?",
    ]
    assert len(after) == 4  # first exchange kept, the edited one answered, the third gone


def test_a_failed_edit_keeps_the_old_conversation(
    service: AssistantService, monkeypatch: pytest.MonkeyPatch
) -> None:
    cid, rows = conversation(service)

    def out_of_keys(*args, **kwargs):
        raise ProvidersExhausted([{"provider": "groq", "reason": "rate_limited"}])

    monkeypatch.setattr("assistant.pipeline.run_probe", out_of_keys)
    events = ask(service, "An edit that fails", cid, edit_from=rows[2]["id"])
    assert events[-1]["code"] == "needs_key"
    assert [r["id"] for r in service.store.messages(cid)] == [r["id"] for r in rows]


def test_only_a_question_of_this_conversation_can_be_edited(service: AssistantService) -> None:
    cid, rows = conversation(service)
    other = ask(service, "A different chat")[-1]["conversation_id"]
    foreign = service.store.messages(other)[0]["id"]
    for bad in (
        foreign,
        rows[1]["id"],
        "made-up-id",
    ):  # another chat's message, an answer, nonsense
        events = ask(service, "edited", cid, edit_from=bad)
        assert events[-1]["code"] == "not_found", bad
    assert len(service.store.messages(cid)) == 6


def test_the_api_accepts_an_edit(service: AssistantService) -> None:
    from fastapi.testclient import TestClient

    from api.main import app

    cid, rows = conversation(service)
    reply = TestClient(app).post(
        "/chat", json={"message": "Rephrased", "conversation_id": cid, "edit_from": rows[4]["id"]}
    )
    assert reply.status_code == 200 and '"type": "done"' in reply.text
    assert [r["content"] for r in service.store.messages(cid) if r["role"] == "user"][
        -1
    ] == "Rephrased"

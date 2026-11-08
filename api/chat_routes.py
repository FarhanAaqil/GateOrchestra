"""
api/chat_routes.py
==================
HTTP endpoints for the chat assistant.

  POST   /chat                      stream one chat turn (Server-Sent Events)
  GET    /conversations             recent conversations
  GET    /conversations/{id}        one conversation with its messages
  PATCH  /conversations/{id}        rename
  DELETE /conversations/{id}        delete
  GET    /assistant/status          providers, masked key status, models, gates
  POST   /assistant/keys            add an API key (checked, then saved to .env)

Adding a key is accepted only from this machine, or with the admin token
(GATE_ADMIN_TOKEN) when the API runs behind a proxy. Keys are never echoed back.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from assistant.pipeline import ChatPipeline
from assistant.service import KeyRejected, get_service

router = APIRouter()
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=20000)
    conversation_id: str | None = None
    gate_mode: str | None = Field(default=None, description="auto, or a gate name to force")
    document_ids: list[str] | None = None


class RenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)


class KeyRequest(BaseModel):
    provider: str = Field(..., pattern="^(groq|gemini)$")
    key: str = Field(..., min_length=20, max_length=200)


def _sse(events: Iterator[dict[str, Any]]) -> Iterator[str]:
    for event in events:
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _require_admin(request: Request) -> None:
    token = os.getenv("GATE_ADMIN_TOKEN", "")
    supplied = request.headers.get("x-admin-token", "")
    host = request.client.host if request.client else ""
    if token and supplied == token:
        return
    if host in LOCAL_HOSTS and not request.headers.get("x-forwarded-for"):
        return
    raise HTTPException(
        status_code=403, detail="Keys can only be added from this machine or with the admin token."
    )


@router.post("/chat")
def chat(body: ChatRequest) -> StreamingResponse:
    service = get_service()
    if not service.slots.acquire(timeout=30):
        raise HTTPException(status_code=503, detail="The assistant is busy; try again in a moment.")

    def events() -> Iterator[str]:
        try:
            yield from _sse(
                ChatPipeline(service).run(
                    body.message,
                    body.conversation_id,
                    gate_mode=body.gate_mode,
                    document_ids=body.document_ids,
                )
            )
        finally:
            service.slots.release()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/conversations")
def list_conversations(limit: int = 50) -> dict[str, Any]:
    return {"conversations": get_service().store.list_conversations(min(max(limit, 1), 200))}


@router.get("/conversations/{cid}")
def get_conversation(cid: str) -> dict[str, Any]:
    store = get_service().store
    if not store.conversation_exists(cid):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    convo = next((c for c in store.list_conversations(500) if c["id"] == cid), {"id": cid})
    return {**convo, "messages": store.messages(cid)}


@router.patch("/conversations/{cid}")
def rename_conversation(cid: str, body: RenameRequest) -> dict[str, str]:
    store = get_service().store
    if not store.conversation_exists(cid):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    store.rename_conversation(cid, body.title)
    return {"status": "renamed"}


@router.delete("/conversations/{cid}")
def delete_conversation(cid: str) -> dict[str, str]:
    get_service().store.delete_conversation(cid)
    return {"status": "deleted"}


@router.get("/assistant/status")
def assistant_status() -> dict[str, Any]:
    return get_service().status()


@router.post("/assistant/keys")
def add_key(body: KeyRequest, request: Request) -> dict[str, Any]:
    _require_admin(request)
    try:
        return get_service().add_key(body.provider, body.key)
    except KeyRejected as err:
        raise HTTPException(status_code=400, detail=str(err)) from err

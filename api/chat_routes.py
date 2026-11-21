"""
api/chat_routes.py
==================
HTTP endpoints for the chat assistant.

  POST   /chat                      stream one chat turn (Server-Sent Events)
  GET    /conversations             recent conversations
  GET    /conversations/{id}        one conversation with its messages
  PATCH  /conversations/{id}        rename
  DELETE /conversations/{id}        delete
  POST   /messages/{id}/feedback    thumbs up (1), down (-1) or clear (0) on an answer
  GET    /assistant/gates           each gate's expected cost and rated accuracy
  POST   /documents                 upload a text, code or PDF file (base64) to ask about
  GET    /documents                 uploaded documents
  DELETE /documents/{id}            remove one
  GET    /assistant/stats           usage, tokens, latency, routing and errors (Admin page)
  GET    /metrics                   the same in Prometheus text format
  DELETE /assistant/cache           clear the answer cache (local or admin token)
  GET    /assistant/models          the chat's fast and answer models (Chat page badge)
  GET    /assistant/status          providers, masked key status, models, gates
  POST   /assistant/keys            add an API key (checked, then saved to .env)

Adding a key is accepted only from this machine, or with the admin token
(GATE_ADMIN_TOKEN) when the API runs behind a proxy. Keys are never echoed back.
With GATE_HOSTED=true the operator endpoints (stats, metrics, status, cache,
keys) are admin-only and a request from localhost no longer counts as admin
(see api/access.py).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from api.access import admin_when_hosted, require_admin
from api.auth_routes import chat_owner
from api.limits import Admission, admit_turn, check_document_room, limit_rate
from assistant.pipeline import ChatPipeline
from assistant.service import KeyRejected, get_service

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=20000)
    conversation_id: str | None = None
    gate_mode: str | None = Field(default=None, description="auto, or a gate name to force")
    document_ids: list[str] | None = None
    edit_from: str | None = Field(
        default=None,
        max_length=40,
        description="id of an earlier question of yours: it and everything after it are replaced",
    )
    replace_last: bool = Field(
        default=False, description="drop the last question and answer first (regenerate or edit)"
    )


class RenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)


class RatingRequest(BaseModel):
    rating: int = Field(..., ge=-1, le=1, description="1 helpful, -1 not helpful, 0 clear")


class DocumentRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    content_base64: str = Field(..., min_length=1, max_length=14_000_000)


class KeyRequest(BaseModel):
    provider: str = Field(..., pattern="^(groq|gemini|openai)$")
    key: str = Field(..., min_length=20, max_length=200)


def _sse(events: Iterator[dict[str, Any]]) -> Iterator[str]:
    for event in events:
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _store(owner: str | None):
    """The caller's own rows: someone else's conversation or document is simply not found."""
    return get_service().store.scoped(owner)


def _counted(
    events: Iterator[dict[str, Any]], service: Any, admission: Admission | None
) -> Iterator[dict[str, Any]]:
    """Pass the turn's events through, recording a finished turn against the person's plan."""
    for event in events:
        if admission is not None and event.get("type") == "done":
            service.usage.record(admission.user_id, event["meta"], own_key=admission.own_key)
        yield event


@router.post("/chat")
def chat(body: ChatRequest, owner: str | None = Depends(chat_owner)) -> StreamingResponse:
    service = get_service()
    admission = admit_turn(service, owner) if owner else None
    if not service.slots.acquire(timeout=30):
        if admission:
            admission.release()
        raise HTTPException(status_code=503, detail="The assistant is busy; try again in a moment.")

    def events() -> Iterator[str]:
        try:
            pipeline = ChatPipeline(service, owner, admission.gateway if admission else None)
            turn = pipeline.run(
                body.message,
                body.conversation_id,
                gate_mode=body.gate_mode,
                document_ids=body.document_ids,
                replace_last=body.replace_last,
                edit_from=body.edit_from,
            )
            yield from _sse(_counted(turn, service, admission))
        finally:
            service.slots.release()
            if admission:
                admission.release()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/conversations")
def list_conversations(limit: int = 50, owner: str | None = Depends(chat_owner)) -> dict[str, Any]:
    return {"conversations": _store(owner).list_conversations(min(max(limit, 1), 200))}


@router.get("/conversations/{cid}")
def get_conversation(cid: str, owner: str | None = Depends(chat_owner)) -> dict[str, Any]:
    store = _store(owner)
    convo = store.conversation(cid)
    if convo is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {**convo, "messages": store.messages(cid)}


@router.patch("/conversations/{cid}")
def rename_conversation(
    cid: str, body: RenameRequest, owner: str | None = Depends(chat_owner)
) -> dict[str, str]:
    if not _store(owner).rename_conversation(cid, body.title):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"status": "renamed"}


@router.delete("/conversations/{cid}")
def delete_conversation(cid: str, owner: str | None = Depends(chat_owner)) -> dict[str, str]:
    if not _store(owner).delete_conversation(cid) and owner is not None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"status": "deleted"}


@router.post("/messages/{mid}/feedback")
def rate_message(
    mid: str, body: RatingRequest, owner: str | None = Depends(chat_owner)
) -> dict[str, Any]:
    try:
        return get_service().rate(mid, body.rating, store=_store(owner))
    except LookupError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err


@router.get("/assistant/gates")
def gate_report(owner: str | None = Depends(chat_owner)) -> dict[str, Any]:
    return get_service().gate_report()


@router.post("/documents")
def upload_document(
    body: DocumentRequest, owner: str | None = Depends(chat_owner)
) -> dict[str, Any]:
    from assistant.documents import DocumentError

    if owner:
        limit_rate(owner)
        check_document_room(get_service(), owner)
    try:
        return get_service().add_document(body.name, body.content_base64, store=_store(owner))
    except DocumentError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err


@router.get("/documents")
def list_documents(owner: str | None = Depends(chat_owner)) -> dict[str, Any]:
    return {"documents": _store(owner).documents()}


@router.delete("/documents/{did}")
def delete_document(did: str, owner: str | None = Depends(chat_owner)) -> dict[str, str]:
    if not _store(owner).delete_document(did) and owner is not None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"status": "deleted"}


@router.get("/assistant/stats", dependencies=[Depends(admin_when_hosted)])
def assistant_stats(hours: float = 24.0) -> dict[str, Any]:
    from assistant.metrics import summarize

    return summarize(get_service().store, min(max(hours, 0.1), 24 * 30))


@router.get("/metrics", response_class=PlainTextResponse, dependencies=[Depends(admin_when_hosted)])
def metrics() -> str:
    from assistant.metrics import prometheus

    return prometheus(get_service().store)


@router.delete("/assistant/cache")
def clear_cache(request: Request) -> dict[str, int]:
    require_admin(request)
    return {"cleared": get_service().store.cache_clear()}


@router.get("/assistant/models")
def chat_models(owner: str | None = Depends(chat_owner)) -> dict[str, str]:
    """Which models answer chats (shown on the Chat page). No keys, no provider state."""
    s = get_service().settings
    return {"fast": s.fast_model, "answer": s.answer_model}


@router.get("/assistant/status", dependencies=[Depends(admin_when_hosted)])
def assistant_status() -> dict[str, Any]:
    return get_service().status()


@router.post("/assistant/keys")
def add_key(body: KeyRequest, request: Request) -> dict[str, Any]:
    require_admin(request)
    try:
        return get_service().add_key(body.provider, body.key)
    except KeyRejected as err:
        raise HTTPException(status_code=400, detail=str(err)) from err

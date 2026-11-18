"""
api/account_routes.py
=====================
A signed-in person's own settings in hosted mode (404 outside it).

  GET    /account/keys              which providers have a key of yours (never the key)
  PUT    /account/keys              store your own Groq or Gemini key (checked first)
  DELETE /account/keys/{provider}   forget it
  GET    /account/usage             today's and this month's usage against your plan
  GET    /plans                     the plans and their limits (public)
  GET    /account/upgrade           where to pay for an upgrade (GATE_PAYMENT_LINK)
  GET    /account/export            everything you wrote here, as JSON
  POST   /account/password          change your password (signs out your other sessions)
  DELETE /account                   delete the account and all of its data (needs the password)

The usage page's "saved" figure is an estimate and says so: answers the gate
handled directly, times the average research-agent tokens that escalated
answers on this server actually used over the last 30 days. It leaves out
small talk and cached answers, which would never have been researched.
"""

from __future__ import annotations

import os
import time
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api.auth_routes import COOKIE, chat_owner, cookie_secure, require_user
from assistant.accounts import AccountError
from assistant.plans import effective_plan, plan_book
from assistant.service import KeyRejected, get_service
from assistant.userkeys import KeyStoreError

router = APIRouter()


class OwnKey(BaseModel):
    provider: str = Field(..., pattern="^(groq|gemini)$")
    key: str = Field(..., min_length=20, max_length=200)


def owner_required(owner: str | None = Depends(chat_owner)) -> str:
    if owner is None:
        raise HTTPException(status_code=404, detail="Accounts are only used in hosted mode.")
    return owner


@router.get("/account/keys")
def list_keys(owner: str = Depends(owner_required)) -> dict[str, Any]:
    return {"keys": get_service().user_keys.listing(owner)}


@router.put("/account/keys")
def put_key(body: OwnKey, owner: str = Depends(owner_required)) -> dict[str, Any]:
    service = get_service()
    try:
        service.check_user_key(body.provider, body.key)
        service.user_keys.put(owner, body.provider, body.key.strip())
    except KeyRejected as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except KeyStoreError as err:
        raise HTTPException(status_code=503, detail="Own keys cannot be stored right now.") from err
    return {"keys": service.user_keys.listing(owner)}


@router.delete("/account/keys/{provider}")
def delete_key(provider: str, owner: str = Depends(owner_required)) -> dict[str, Any]:
    service = get_service()
    if not service.user_keys.remove(owner, provider):
        raise HTTPException(status_code=404, detail="No key stored for that provider.")
    return {"keys": service.user_keys.listing(owner)}


ESTIMATE_WINDOW_DAYS = 30
MIN_ESCALATIONS_FOR_ESTIMATE = 5


def research_cost_estimate(service: Any) -> dict[str, Any]:
    """Average research-agent tokens per escalated answer on this server (last 30 days)."""
    since = time.time() - ESTIMATE_WINDOW_DAYS * 86400
    agent = [
        int((e.get("tokens") or {}).get("agent") or 0)
        for e in service.store.events("chat", since=since, limit=20000)
        if e.get("decision") == "ESCALATE" and not e.get("cached")
    ]
    agent = [a for a in agent if a > 0]
    if len(agent) < MIN_ESCALATIONS_FOR_ESTIMATE:
        return {"available": False, "sample_size": len(agent), "per_escalation": None}
    return {
        "available": True,
        "sample_size": len(agent),
        "per_escalation": round(sum(agent) / len(agent)),
    }


@router.get("/account/usage")
def usage(owner: str = Depends(owner_required)) -> dict[str, Any]:
    service = get_service()
    user = service.accounts.get(owner)
    if user is None:
        raise HTTPException(status_code=401, detail="Log in to continue.")
    plan = effective_plan(user)
    month = service.usage.month(owner)
    cost = research_cost_estimate(service)
    saved = month["direct"] * cost["per_escalation"] if cost["available"] else None
    return {
        "plan": plan.as_dict(),
        "paid_until": user["paid_until"] if plan.id == user["plan"] else None,
        "today": service.usage.today(owner),
        "month": month,
        "documents": len(service.store.scoped(owner).documents()),
        "own_key": bool(service.user_keys.listing(owner)),
        "estimate": {
            **cost,
            "saved_tokens": saved,
            "window_days": ESTIMATE_WINDOW_DAYS,
            "method": (
                "Estimate only: answers handled directly this month x the average "
                "research-agent tokens of escalated answers on this server over the last "
                f"{ESTIMATE_WINDOW_DAYS} days. Small talk and cached answers are left out. "
                "It is not a measurement of what those answers would really have cost."
            ),
        },
    }


@router.get("/plans")
def plans() -> dict[str, Any]:
    book = plan_book()
    return {
        "default": book.default,
        "plans": [p.as_dict() for p in book.plans.values()],
        "payments": bool(os.getenv("GATE_PAYMENT_LINK", "").strip()),
    }


@router.get("/account/upgrade")
def upgrade(owner: str = Depends(owner_required)) -> dict[str, str]:
    """The payment page for this person. ``{email}`` in the link is filled in, so the
    provider's checkout can show it and the admin can match the payment to the account."""
    link = os.getenv("GATE_PAYMENT_LINK", "").strip()
    if not link:
        raise HTTPException(status_code=404, detail="Payments are not set up on this server yet.")
    user = get_service().accounts.get(owner)
    email = user["email"] if user else ""
    return {"url": link.replace("{email}", quote(email, safe=""))}


class PasswordCheck(BaseModel):
    password: str = Field(..., min_length=1, max_length=256)


class PasswordChange(BaseModel):
    current: str = Field(..., min_length=1, max_length=256)
    new: str = Field(..., min_length=1, max_length=256)


def _signed_in(request: Request) -> dict[str, Any]:
    """Hosted only; verified or not (an unconfirmed person may still leave or export)."""
    from assistant.hosted import hosted_mode  # noqa: PLC0415

    if not hosted_mode():
        raise HTTPException(status_code=404, detail="Accounts are only used in hosted mode.")
    return require_user(request)


def _check_password(user: dict[str, Any], password: str) -> None:
    if get_service().accounts.authenticate(user["email"], password) is None:
        raise HTTPException(status_code=403, detail="That password is not right.")


@router.get("/account/export")
def export(request: Request) -> JSONResponse:
    user = _signed_in(request)
    service = get_service()
    data = {
        "account": {k: v for k, v in user.items() if k != "csrf"},
        "usage_this_month": service.usage.month(user["id"]),
        "own_keys": service.user_keys.listing(user["id"]),
        **service.store.scoped(user["id"]).export(),
    }
    return JSONResponse(
        data, headers={"Content-Disposition": 'attachment; filename="gateorchestra-export.json"'}
    )


@router.post("/account/password")
def change_password(body: PasswordChange, request: Request, response: Response) -> dict[str, Any]:
    from api.auth_routes import _start_session  # noqa: PLC0415

    user = _signed_in(request)
    _check_password(user, body.current)
    accounts = get_service().accounts
    try:
        accounts.set_password(user["id"], body.new)  # ends every session, this one too
    except AccountError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    fresh = accounts.get(user["id"])
    assert fresh is not None
    return _start_session(response, fresh)  # keep this browser signed in


@router.delete("/account")
def delete_account(body: PasswordCheck, request: Request, response: Response) -> dict[str, str]:
    """Remove the person: conversations, messages, documents, chunks, cached answers,
    feedback, own API keys, usage counts, sessions and the account itself."""
    user = _signed_in(request)
    _check_password(user, body.password)
    service = get_service()
    service.store.scoped(user["id"]).delete_everything()
    service.accounts.delete(user["id"])  # cascades to sessions, email tokens, keys and usage
    response.delete_cookie(COOKIE, path="/", secure=cookie_secure(), samesite="strict")
    return {"status": "deleted"}

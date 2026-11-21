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

import json
import os
import time
from pathlib import Path
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
    provider: str = Field(..., pattern="^(groq|gemini|openai)$")
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
CHAT_EVAL_SUMMARY = (
    Path(__file__).resolve().parent.parent / "results" / "chat_eval" / "summary.json"
)


def measured_research_cost(path: Path | None = None) -> dict[str, Any] | None:
    """Extra tokens research cost per question in the 80-question chat evaluation, if recorded."""
    path = path or CHAT_EVAL_SUMMARY
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        extra = float(summary["research_vs_direct"]["avg_extra_tokens_for_research"])
        return {"per_escalation": round(extra), "sample_size": int(summary["paired"])}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def research_cost_estimate(service: Any) -> dict[str, Any]:
    """Average research-agent tokens per escalated answer on this server (last 30 days)."""
    since = time.time() - ESTIMATE_WINDOW_DAYS * 86400
    agent = [
        int((e.get("tokens") or {}).get("agent") or 0)
        for e in service.store.events("chat", since=since, limit=20000)
        if e.get("decision") == "ESCALATE" and not e.get("cached")
    ]
    agent = [a for a in agent if a > 0]
    if len(agent) >= MIN_ESCALATIONS_FOR_ESTIMATE:
        return {
            "available": True,
            "source": "server",
            "sample_size": len(agent),
            "per_escalation": round(sum(agent) / len(agent)),
        }
    measured = measured_research_cost()
    if measured:  # until this server has enough of its own, the measured evaluation cost
        return {"available": True, "source": "evaluation", **measured}
    return {"available": False, "source": None, "sample_size": len(agent), "per_escalation": None}


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
                "Estimate only: answers handled directly this month x the extra tokens research "
                + (
                    f"used on escalated answers on this server over the last {ESTIMATE_WINDOW_DAYS} days"
                    if cost.get("source") == "server"
                    else "cost per question in our 80-question evaluation (reports/chat_eval.md), "
                    "until this server has enough researched answers of its own"
                )
                + ". Small talk and cached answers are left out. It is not a measurement of what "
                "those answers would really have cost."
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


# Two-factor sign-in and email change


class TwoFactorCode(BaseModel):
    code: str = Field(..., min_length=6, max_length=20)


class TwoFactorOff(BaseModel):
    password: str = Field(..., min_length=1, max_length=256)
    code: str = Field(..., min_length=6, max_length=20)


class EmailChange(BaseModel):
    new_email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=1, max_length=256)


@router.get("/account/security")
def security_status(request: Request) -> dict[str, Any]:
    user = _signed_in(request)
    accounts = get_service().accounts
    return {
        "two_factor": user["two_factor"],
        "recovery_codes_left": (
            accounts.recovery_codes_left(user["id"]) if user["two_factor"] else 0
        ),
    }


@router.post("/account/2fa/setup")
def two_factor_setup(body: PasswordCheck, request: Request) -> dict[str, str]:
    """A new secret for the authenticator app; it only counts once a code is confirmed."""
    from assistant.totp import otpauth_uri  # noqa: PLC0415
    from assistant.userkeys import KeyStoreError  # noqa: PLC0415

    user = _signed_in(request)
    _check_password(user, body.password)
    try:
        secret = get_service().accounts.begin_two_factor(user["id"])
    except KeyStoreError as err:
        raise HTTPException(
            status_code=503, detail="Two-factor sign-in is not available right now."
        ) from err
    return {"secret": secret, "otpauth_uri": otpauth_uri(secret, user["email"])}


@router.post("/account/2fa/enable")
def two_factor_enable(body: TwoFactorCode, request: Request) -> dict[str, Any]:
    user = _signed_in(request)
    codes = get_service().accounts.enable_two_factor(user["id"], body.code)
    if codes is None:
        raise HTTPException(
            status_code=400,
            detail="That code is not right. Check the time on your phone and try again.",
        )
    return {"recovery_codes": codes}


@router.post("/account/2fa/disable")
def two_factor_disable(body: TwoFactorOff, request: Request) -> dict[str, str]:
    user = _signed_in(request)
    _check_password(user, body.password)
    accounts = get_service().accounts
    if not accounts.check_second_factor(user["id"], body.code):
        raise HTTPException(status_code=403, detail="That code is not right.")
    accounts.disable_two_factor(user["id"])
    return {"status": "disabled"}


@router.post("/account/email", status_code=202)
def request_email_change(body: EmailChange, request: Request) -> dict[str, str]:
    """Email a confirmation link to the new address and a notice to the current one."""
    from api.auth_routes import _limit, public_url  # noqa: PLC0415
    from assistant.accounts import normalize_email  # noqa: PLC0415
    from assistant.mailer import MailError, send_mail  # noqa: PLC0415

    user = _signed_in(request)
    _limit(request, user["email"])
    _check_password(user, body.password)
    try:
        new_email = normalize_email(body.new_email)
    except AccountError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    if new_email == user["email"]:
        raise HTTPException(status_code=400, detail="That is already your address.")
    accounts = get_service().accounts
    try:
        if accounts.by_email(new_email):
            # Same reply as a free address, so this cannot be used to find who has an account.
            send_mail(
                new_email,
                "Someone tried to move a GateOrchestra account to this address",
                "This address already has a GateOrchestra account, so nothing was changed.",
            )
        else:
            token = accounts.issue_token(user["id"], "email_change", data=new_email)
            send_mail(
                new_email,
                "Confirm your new GateOrchestra email",
                f"Confirm this address for your GateOrchestra account:\n{public_url()}/confirm-email?token={token}\n\n"
                "The link works once and expires in 1 hour. If you did not ask for this, ignore it.",
            )
        send_mail(
            user["email"],
            "Your GateOrchestra email is being changed",
            f"Someone signed in to your account asked to move it to {new_email}. It changes only "
            "when that address confirms. If this was not you, change your password now.",
        )
    except MailError as err:
        raise HTTPException(status_code=503, detail=str(err)) from err
    return {"status": "check_email"}

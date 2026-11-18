"""
api/admin_routes.py
===================
People and plans, for admins of a hosted server (404 outside hosted mode).

  GET  /admin/users?q=            accounts whose email contains q, with plan and today's use
  POST /admin/users/{id}/plan     put someone on a plan for a number of days

This is payments version 1: a person pays through the payment link, the
admin sees the payment and sets the plan here. Paying again extends the paid
period from where it ends. When it ends the person is simply on the free plan
again (assistant/plans.py), so nothing has to run to downgrade them.
Version 2, an automatic upgrade from the payment provider's webhook, is not
built yet.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from api.access import require_admin
from assistant.hosted import hosted_mode
from assistant.plans import effective_plan, plan_book
from assistant.service import get_service

router = APIRouter()
DAY = 86400


def hosted_admin(request: Request) -> None:
    if not hosted_mode():
        raise HTTPException(status_code=404, detail="Accounts are only used in hosted mode.")
    require_admin(request)


class PlanChange(BaseModel):
    plan: str = Field(..., min_length=1, max_length=40)
    days: int = Field(default=31, ge=0, le=3660, description="ignored for the free plan")


def _row(user: dict[str, Any]) -> dict[str, Any]:
    service = get_service()
    return {
        **service.accounts.public(user),
        "effective_plan": effective_plan(user).id,
        "today": service.usage.today(user["id"]),
    }


@router.get("/admin/users", dependencies=[Depends(hosted_admin)])
def list_users(q: str = "", limit: int = 50) -> dict[str, Any]:
    users = get_service().accounts.search(q[:200], min(max(limit, 1), 500))
    return {"users": [_row(u) for u in users]}


@router.post("/admin/users/{user_id}/plan", dependencies=[Depends(hosted_admin)])
def set_plan(user_id: str, body: PlanChange) -> dict[str, Any]:
    service = get_service()
    book = plan_book()
    if body.plan not in book.plans:
        raise HTTPException(status_code=400, detail=f"Unknown plan {body.plan!r}.")
    user = service.accounts.get(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="No such user.")
    if body.plan == book.default:
        paid_until = None
    else:
        now = time.time()
        still_paid = user["plan"] == body.plan and (user["paid_until"] or 0) > now
        start = user["paid_until"] if still_paid else now
        paid_until = start + body.days * DAY
    service.accounts.set_plan(user_id, body.plan, paid_until)
    updated = service.accounts.get(user_id)
    assert updated is not None
    return {"user": _row(updated)}

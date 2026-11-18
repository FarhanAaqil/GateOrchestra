"""
api/limits.py
=============
What one person may use in hosted mode, checked before a chat turn starts:

  * a request rate per person (GATE_USER_RATE_LIMIT_PER_MINUTE, default 10),
    on top of the per-address limit in the middleware;
  * answers in progress at once (the plan's ``concurrent_chats``);
  * the plan's answers per day (every turn counts, whoever's key answers it);
  * the plan's daily token allowance on the operator's keys;
  * the operator's global daily token cap (GATE_GLOBAL_DAILY_TOKENS or
    configs/plans.yaml), the safety switch for the whole service.

Turns answered with the person's own API key do not use up either token
allowance, since the operator does not pay for them. A turn is checked
before it starts, so the last one of the day can go a little over.
"""

from __future__ import annotations

import os
import threading
from collections import defaultdict
from dataclasses import dataclass

from fastapi import HTTPException

from assistant.llm import Gateway
from assistant.plans import Plan, effective_plan, global_daily_tokens
from assistant.security import RateLimiter
from assistant.service import AssistantService

_per_user = RateLimiter(int(os.getenv("GATE_USER_RATE_LIMIT_PER_MINUTE", "10")))
_running: dict[str, int] = defaultdict(int)
_running_lock = threading.Lock()


def limit_rate(user_id: str) -> None:
    wait = _per_user.check(f"user:{user_id}")
    if wait:
        raise HTTPException(
            status_code=429,
            detail=f"You are sending messages very quickly. Try again in {wait:.0f}s.",
            headers={"Retry-After": str(max(1, round(wait)))},
        )


def check_document_room(service: AssistantService, user_id: str) -> None:
    """The plan's limit on stored documents (checked before an upload)."""
    plan = plan_for(service, user_id)
    if len(service.store.scoped(user_id).documents()) >= plan.documents:
        raise HTTPException(
            status_code=403,
            detail=(
                f"The {plan.label} plan stores up to {plan.documents} documents. "
                "Remove one, or upgrade for more."
            ),
        )


def plan_for(service: AssistantService, user_id: str) -> Plan:
    user = service.accounts.get(user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="Log in to continue.")
    return effective_plan(user)


@dataclass
class Admission:
    user_id: str
    plan: Plan
    gateway: Gateway
    own_key: bool

    def release(self) -> None:
        with _running_lock:
            _running[self.user_id] = max(0, _running[self.user_id] - 1)


def admit_turn(service: AssistantService, user_id: str) -> Admission:
    """Check every limit for one chat turn; the caller must ``release()`` the admission."""
    limit_rate(user_id)
    plan = plan_for(service, user_id)
    gateway, own_key = service.gateway_for(user_id)
    if not own_key:
        cap = global_daily_tokens()
        if cap and service.usage.everyone_today()["tokens"] >= cap:
            raise HTTPException(
                status_code=503,
                detail="The assistant has reached its daily capacity. It resets at midnight UTC.",
            )
        used = service.usage.today(user_id)["tokens"]
        if used >= plan.daily_tokens:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"You have used today's {plan.daily_tokens:,} tokens on the {plan.label} plan. "
                    "It resets at midnight UTC; upgrading or adding your own API key in Settings "
                    "lifts the limit."
                ),
            )
    if service.usage.today(user_id)["messages"] >= plan.messages_per_day:
        raise HTTPException(
            status_code=429,
            detail=(
                f"You have had today's {plan.messages_per_day:,} answers on the {plan.label} plan. "
                "It resets at midnight UTC, or upgrade for more."
            ),
        )
    with _running_lock:
        if _running[user_id] >= plan.concurrent_chats:
            raise HTTPException(
                status_code=429,
                detail="Wait for your other answer to finish before asking again.",
            )
        _running[user_id] += 1
    return Admission(user_id, plan, gateway, own_key)

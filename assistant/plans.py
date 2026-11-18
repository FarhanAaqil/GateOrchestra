"""
assistant/plans.py
==================
Plans and their limits for hosted mode, read from configs/plans.yaml
(GATE_PLANS_PATH overrides the location).

A paid plan holds while ``paid_until`` is in the future; after that the user
is treated as being on the default plan again, with no job needed to
downgrade them.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from shared.config import ROOT_DIR

DEFAULT_PATH = ROOT_DIR / "configs" / "plans.yaml"


@dataclass(frozen=True)
class Plan:
    id: str
    label: str
    price_usd_month: float
    messages_per_day: int
    documents: int
    daily_tokens: int
    concurrent_chats: int
    blurb: str = ""

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class PlanBook:
    plans: dict[str, Plan]
    default: str
    global_daily_tokens: int

    def get(self, plan_id: str | None) -> Plan:
        return self.plans.get(plan_id or "", self.plans[self.default])


def plans_path() -> Path:
    return Path(os.getenv("GATE_PLANS_PATH", str(DEFAULT_PATH)))


@lru_cache(maxsize=4)
def _load(path: str, mtime: float) -> PlanBook:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    plans = {
        pid: Plan(
            id=pid,
            label=str(spec.get("label", pid.title())),
            price_usd_month=float(spec.get("price_usd_month", 0)),
            messages_per_day=int(spec["messages_per_day"]),
            documents=int(spec["documents"]),
            daily_tokens=int(spec["daily_tokens"]),
            concurrent_chats=int(spec.get("concurrent_chats", 1)),
            blurb=str(spec.get("blurb", "")),
        )
        for pid, spec in (raw.get("plans") or {}).items()
    }
    default = str(raw.get("default_plan", "free"))
    if default not in plans:
        raise ValueError(f"{path}: default_plan {default!r} is not one of the plans")
    return PlanBook(plans, default, int(raw.get("global_daily_tokens", 0)))


def plan_book() -> PlanBook:
    path = plans_path()
    return _load(str(path), path.stat().st_mtime)


def global_daily_tokens() -> int:
    override = os.getenv("GATE_GLOBAL_DAILY_TOKENS", "").strip()
    return int(override) if override else plan_book().global_daily_tokens


def effective_plan(user: dict[str, Any], now: float | None = None) -> Plan:
    """The plan whose limits apply now: a lapsed paid plan counts as the default plan."""
    book = plan_book()
    plan_id = user.get("plan") or book.default
    paid_until = user.get("paid_until")
    if plan_id != book.default and (paid_until is None or paid_until <= (now or time.time())):
        plan_id = book.default
    return book.get(plan_id)

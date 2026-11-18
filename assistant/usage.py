"""
assistant/usage.py
==================
Per-person usage for hosted mode, counted per UTC day: answered turns, tokens
on the operator's keys, tokens on the person's own key, and how each turn was
routed (answered directly, escalated to the research agent, small talk or
served from the cache). Only counts are stored, never message content.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from assistant.store import Store

SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_daily (
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    day TEXT NOT NULL,
    messages INTEGER NOT NULL DEFAULT 0,
    tokens INTEGER NOT NULL DEFAULT 0,
    own_key_tokens INTEGER NOT NULL DEFAULT 0,
    direct INTEGER NOT NULL DEFAULT 0,
    escalated INTEGER NOT NULL DEFAULT 0,
    small_talk INTEGER NOT NULL DEFAULT 0,
    cached INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, day)
);
CREATE INDEX IF NOT EXISTS idx_usage_day ON usage_daily(day);
"""
COUNTERS = ("messages", "tokens", "own_key_tokens", "direct", "escalated", "small_talk", "cached")


def utc_day(ts: float | None = None) -> str:
    return datetime.fromtimestamp(ts or time.time(), tz=timezone.utc).strftime("%Y-%m-%d")


def route_of(meta: dict[str, Any]) -> str:
    """How a finished turn was answered: cached, small_talk, escalated or direct."""
    if meta.get("cached"):
        return "cached"
    if meta.get("gate") == "triage":
        return "small_talk"
    return "escalated" if meta.get("decision") == "ESCALATE" else "direct"


class Usage:
    def __init__(self, store: Store) -> None:
        self.store = store
        with store.transaction() as db:
            db.executescript(SCHEMA)

    def record(
        self, user_id: str, meta: dict[str, Any], *, own_key: bool, ts: float | None = None
    ) -> None:
        tokens = int(meta.get("total_tokens") or 0)
        route = route_of(meta)
        with self.store.transaction() as db:
            db.execute(
                f"INSERT INTO usage_daily (user_id, day, messages, tokens, own_key_tokens, {route}) "
                "VALUES (?, ?, 1, ?, ?, 1) ON CONFLICT(user_id, day) DO UPDATE SET "
                "messages = messages + 1, tokens = tokens + excluded.tokens, "
                f"own_key_tokens = own_key_tokens + excluded.own_key_tokens, {route} = {route} + 1",
                (user_id, utc_day(ts), 0 if own_key else tokens, tokens if own_key else 0),
            )

    def _sum(self, where: str, params: tuple) -> dict[str, int]:
        columns = ", ".join(f"COALESCE(SUM({c}), 0) AS {c}" for c in COUNTERS)
        with self.store.read() as db:
            row = db.execute(f"SELECT {columns} FROM usage_daily WHERE {where}", params).fetchone()
        return {c: int(row[c]) for c in COUNTERS}

    def today(self, user_id: str, ts: float | None = None) -> dict[str, int]:
        return self._sum("user_id = ? AND day = ?", (user_id, utc_day(ts)))

    def month(self, user_id: str, ts: float | None = None) -> dict[str, int]:
        """This UTC calendar month so far."""
        return self._sum("user_id = ? AND day LIKE ?", (user_id, utc_day(ts)[:7] + "-%"))

    def everyone_today(self, ts: float | None = None) -> dict[str, int]:
        return self._sum("day = ?", (utc_day(ts),))

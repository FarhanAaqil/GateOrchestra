"""
assistant/shared_limits.py
==========================
Rate limits and concurrency slots shared by every server process.

A hosted server can run several uvicorn workers. Limits kept in a Python
dict would then be per process, so a person could get N times their allowance
and a restart would forget everything. These live in the assistant's SQLite
database instead:

  * SharedRateLimiter: at most ``limit`` hits per key in a sliding ``window``.
    Same ``check(key) -> seconds to wait`` interface as the in-memory
    assistant.security.RateLimiter.
  * Leases: concurrency slots ("answers in progress", "sandbox runs") that
    expire on their own, so a crashed worker can never hold a slot forever.

Every decision runs inside ``BEGIN IMMEDIATE``, which takes SQLite's write
lock, so two processes cannot both take the last slot.

Single-tenant mode keeps the in-memory limiters it always had.
"""

from __future__ import annotations

import random
import time
import uuid
from collections.abc import Callable
from typing import Any

from assistant.store import Store

SCHEMA = """
CREATE TABLE IF NOT EXISTS rate_hits (bucket TEXT NOT NULL, ts REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_rate_hits_bucket ON rate_hits(bucket, ts);
CREATE TABLE IF NOT EXISTS leases (
    id TEXT PRIMARY KEY, bucket TEXT NOT NULL, expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_leases_bucket ON leases(bucket, expires_at);
"""
_ready: set[str] = set()


def _ensure_schema(store: Store) -> None:
    key = str(store.path)
    if key in _ready:
        return
    with store.transaction() as db:
        db.executescript(SCHEMA)
    _ready.add(key)


def default_store() -> Store:
    from assistant.service import get_service  # noqa: PLC0415  (avoids an import cycle)

    return get_service().store


class SharedRateLimiter:
    """At most ``limit`` hits per key in any rolling ``window`` seconds, across processes."""

    def __init__(
        self,
        name: str,
        limit: int,
        window: float = 60.0,
        store: Callable[[], Store] = default_store,
    ) -> None:
        self.name = name
        self.limit = limit
        self.window = window
        self._store = store

    def check(self, client: str, now: float | None = None) -> float:
        """0 if allowed (and counted), else seconds until the key may try again."""
        if self.limit <= 0:
            return 0.0
        now = time.time() if now is None else now
        store = self._store()
        _ensure_schema(store)
        bucket = f"{self.name}:{client}"
        with store.transaction() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "DELETE FROM rate_hits WHERE bucket = ? AND ts <= ?", (bucket, now - self.window)
            )
            row = db.execute(
                "SELECT COUNT(*) AS n, MIN(ts) AS oldest FROM rate_hits WHERE bucket = ?", (bucket,)
            ).fetchone()
            if row["n"] >= self.limit:
                return max(0.1, self.window - (now - row["oldest"]))
            db.execute("INSERT INTO rate_hits (bucket, ts) VALUES (?, ?)", (bucket, now))
            if random.random() < 0.01:  # now and then, drop every expired hit
                db.execute("DELETE FROM rate_hits WHERE ts < ?", (now - 86400,))
        return 0.0

    def blocked(self, client: str, now: float | None = None) -> float:
        """Seconds until ``client`` is under the limit again, without counting a hit (0 = free)."""
        if self.limit <= 0:
            return 0.0
        now = time.time() if now is None else now
        store = self._store()
        _ensure_schema(store)
        with store.read() as db:
            row = db.execute(
                "SELECT COUNT(*) AS n, MIN(ts) AS oldest FROM rate_hits WHERE bucket = ? AND ts > ?",
                (f"{self.name}:{client}", now - self.window),
            ).fetchone()
        if row["n"] < self.limit:
            return 0.0
        return max(0.1, self.window - (now - row["oldest"]))

    def clear(self, client: str) -> None:
        store = self._store()
        _ensure_schema(store)
        with store.transaction() as db:
            db.execute("DELETE FROM rate_hits WHERE bucket = ?", (f"{self.name}:{client}",))


class Leases:
    """Expiring concurrency slots, shared across processes."""

    def __init__(self, store: Callable[[], Store] = default_store) -> None:
        self._store = store

    def acquire(self, bucket: str, limit: int, ttl: float, now: float | None = None) -> str | None:
        """A lease id if fewer than ``limit`` live leases hold ``bucket``, else None."""
        now = time.time() if now is None else now
        store = self._store()
        _ensure_schema(store)
        with store.transaction() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM leases WHERE bucket = ? AND expires_at <= ?", (bucket, now))
            held = db.execute("SELECT COUNT(*) FROM leases WHERE bucket = ?", (bucket,)).fetchone()[
                0
            ]
            if held >= limit:
                return None
            lease = uuid.uuid4().hex
            db.execute(
                "INSERT INTO leases (id, bucket, expires_at) VALUES (?, ?, ?)",
                (lease, bucket, now + ttl),
            )
        return lease

    def acquire_all(
        self, wanted: list[tuple[str, int]], ttl: float, now: float | None = None
    ) -> list[str] | None:
        """Leases on several buckets at once, or none of them (all-or-nothing)."""
        taken: list[str] = []
        for bucket, limit in wanted:
            lease = self.acquire(bucket, limit, ttl, now)
            if lease is None:
                self.release(*taken)
                return None
            taken.append(lease)
        return taken

    def release(self, *lease_ids: str) -> None:
        ids = [i for i in lease_ids if i]
        if not ids:
            return
        store = self._store()
        _ensure_schema(store)
        with store.transaction() as db:
            db.execute(f"DELETE FROM leases WHERE id IN ({','.join('?' * len(ids))})", tuple(ids))

    def held(self, bucket: str, now: float | None = None) -> int:
        now = time.time() if now is None else now
        store = self._store()
        _ensure_schema(store)
        with store.read() as db:
            return int(
                db.execute(
                    "SELECT COUNT(*) FROM leases WHERE bucket = ? AND expires_at > ?", (bucket, now)
                ).fetchone()[0]
            )


class HostedAwareLimiter:
    """The in-memory limiter in single-tenant mode, the shared one when hosted."""

    def __init__(self, local: Any, shared: SharedRateLimiter) -> None:
        self.local = local
        self.shared = shared

    def check(self, client: str, now: float | None = None) -> float:
        from assistant.hosted import hosted_mode  # noqa: PLC0415

        return self.shared.check(client) if hosted_mode() else self.local.check(client, now)

"""
assistant/security.py
=====================
Request protection for a deployed assistant.

  * API token (GATE_API_TOKEN): when set, every endpoint except / and /health
    needs "Authorization: Bearer <token>". Unset, the API is open, which is
    only safe on localhost.
  * Rate limit (GATE_RATE_LIMIT_PER_MINUTE, default 30): a sliding one-minute
    window per client for the expensive endpoints (chat and uploads).
  * Prompt-injection screening: Groq's prompt-guard model scores each message;
    above ASSISTANT_GUARD_THRESHOLD (default 0.9) the message is refused. If
    the guard cannot be reached the message goes through and the miss is
    logged, so an outage of the guard does not take the chat down.
"""

from __future__ import annotations

import hmac
import logging
import os
import threading
import time
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

OPEN_PATHS = {"/", "/health", "/docs", "/openapi.json", "/redoc"}
LIMITED_PATHS = ("/chat", "/documents")


def api_token() -> str:
    return os.getenv("GATE_API_TOKEN", "").strip()


def token_ok(header: str | None) -> bool:
    """Constant-time check of an 'Authorization: Bearer <token>' header."""
    expected = api_token()
    if not expected:
        return True
    supplied = (header or "").removeprefix("Bearer ").strip()
    return bool(supplied) and hmac.compare_digest(supplied, expected)


class RateLimiter:
    """At most ``limit`` requests per client in any rolling ``window`` seconds."""

    def __init__(self, limit: int, window: float = 60.0) -> None:
        self.limit = limit
        self.window = window
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, client: str, now: float | None = None) -> float:
        """0 if allowed (and counted), else seconds until the client may retry."""
        if self.limit <= 0:
            return 0.0
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits[client]
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return max(0.1, self.window - (now - hits[0]))
            hits.append(now)
            return 0.0


def rate_limit_from_env() -> RateLimiter:
    return RateLimiter(int(os.getenv("GATE_RATE_LIMIT_PER_MINUTE", "30")))


def guard_threshold() -> float:
    return float(os.getenv("ASSISTANT_GUARD_THRESHOLD", "0.9"))


def parse_guard_score(text: str) -> float | None:
    try:
        return float(text.strip().split()[0])
    except (ValueError, IndexError):
        return None

"""
api/security.py
===============
ASGI middleware for the API token and the per-client rate limit.

Plain ASGI rather than BaseHTTPMiddleware, so streamed chat answers pass
through untouched. Preflight (OPTIONS) requests are never blocked.
"""

from __future__ import annotations

import json
from typing import Any

from assistant.security import LIMITED_PATHS, OPEN_PATHS, rate_limit_from_env, token_ok


class SecurityMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app
        self.limiter = rate_limit_from_env()

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope.get("method") == "OPTIONS":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        if path not in OPEN_PATHS and not token_ok(headers.get("authorization")):
            await _reply(
                send,
                401,
                {"detail": "A valid API token is required."},
                {"www-authenticate": "Bearer"},
            )
            return
        if scope.get("method") == "POST" and path.startswith(LIMITED_PATHS):
            forwarded = headers.get("x-forwarded-for", "").split(",")[0].strip()
            client = forwarded or (scope.get("client") or ("unknown", 0))[0]
            wait = self.limiter.check(client)
            if wait:
                await _reply(
                    send,
                    429,
                    {"detail": f"Too many requests. Try again in {wait:.0f}s."},
                    {"retry-after": str(max(1, round(wait)))},
                )
                return
        await self.app(scope, receive, send)


async def _reply(send: Any, status: int, body: dict, extra: dict[str, str]) -> None:
    payload = json.dumps(body).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(payload)).encode()),
    ]
    headers += [(k.encode(), v.encode()) for k, v in extra.items()]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": payload})

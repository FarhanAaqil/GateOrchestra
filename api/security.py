"""
api/security.py
===============
ASGI middleware for the API token and the per-client rate limit.

The client is the direct peer unless it is a trusted proxy (see
assistant/hosted.py), so a forged X-Forwarded-For cannot dodge the limit.

Plain ASGI rather than BaseHTTPMiddleware, so streamed chat answers pass
through untouched. Preflight (OPTIONS) requests are never blocked.
"""

from __future__ import annotations

import json
from typing import Any

from assistant.hosted import client_address, hosted_mode
from assistant.security import LIMITED_PATHS, OPEN_PATHS, rate_limit_from_env, token_ok
from assistant.shared_limits import HostedAwareLimiter, SharedRateLimiter

# Hosted mode: sent with every API response.
SECURITY_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"strict-origin-when-cross-origin"),
    (b"cross-origin-opener-policy", b"same-origin"),
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
]
# Responses with personal data are never stored by a browser or proxy cache.
PRIVATE_PREFIXES = ("/auth", "/account", "/admin", "/conversations", "/documents", "/messages")


def _with_security_headers(send: Any, path: str) -> Any:
    async def tagged(message: dict) -> None:
        if message["type"] == "http.response.start":
            extra = list(SECURITY_HEADERS)
            if path.startswith(PRIVATE_PREFIXES):
                extra.append((b"cache-control", b"no-store"))
            names = {k.lower() for k, _ in message.get("headers", [])}
            message["headers"] = [
                *message.get("headers", []),
                *[(k, v) for k, v in extra if k not in names],
            ]
        await send(message)

    return tagged


class SecurityMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app
        local = rate_limit_from_env()
        # Hosted servers may run several workers, so their limit lives in the database.
        self.limiter = HostedAwareLimiter(local, SharedRateLimiter("address", local.limit))

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope.get("method") == "OPTIONS":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if hosted_mode():
            send = _with_security_headers(send, path)
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        # Hosted mode signs people in with accounts; the shared token is for single-tenant use.
        if (
            not hosted_mode()
            and path not in OPEN_PATHS
            and not token_ok(headers.get("authorization"))
        ):
            await _reply(
                send,
                401,
                {"detail": "A valid API token is required."},
                {"www-authenticate": "Bearer"},
            )
            return
        if scope.get("method") == "POST" and path.startswith(LIMITED_PATHS):
            peer = (scope.get("client") or ("unknown", 0))[0]
            client = client_address(peer, headers.get("x-forwarded-for"))
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

"""
api/observability.py
====================
Production logging and error tracking for hosted mode.

  * Every request gets a request id (a well-formed incoming ``X-Request-ID``
    is kept, so it matches the proxy's logs) that is echoed in the response.
  * One JSON line per request on the ``gateorchestra.access`` logger: time,
    request id, user id, method, path, status and duration. Never the query
    string, the body, cookies, headers, message content, documents or keys.
  * Error tracking with Sentry when SENTRY_DSN is set and ``sentry-sdk`` is
    installed. Request data, cookies, headers and local variables are dropped
    before anything leaves the server.

All of it is on in hosted mode, or with GATE_REQUEST_LOG=1.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from typing import Any

from assistant.hosted import hosted_mode

access_log = logging.getLogger("gateorchestra.access")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._\-]{8,64}$")


def request_logging_enabled() -> bool:
    return hosted_mode() or os.getenv("GATE_REQUEST_LOG", "0") == "1"


class RequestLogMiddleware:
    """Plain ASGI, so streamed answers pass through untouched."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or not request_logging_enabled():
            await self.app(scope, receive, send)
            return
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        incoming = headers.get("x-request-id", "")
        request_id = incoming if _REQUEST_ID.match(incoming) else uuid.uuid4().hex
        started = time.perf_counter()
        status = {"code": 500}

        async def tagged_send(message: dict) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                message.setdefault("headers", [])
                message["headers"] = [
                    *message["headers"],
                    (b"x-request-id", request_id.encode()),
                ]
            await send(message)

        try:
            await self.app(scope, receive, tagged_send)
        finally:
            user = (scope.get("state") or {}).get("session_user") or {}
            access_log.info(
                json.dumps(
                    {
                        "ts": round(time.time(), 3),
                        "request_id": request_id,
                        "user_id": user.get("id"),
                        "method": scope.get("method"),
                        "path": scope.get("path"),
                        "status": status["code"],
                        "ms": round((time.perf_counter() - started) * 1000),
                    }
                )
            )


def scrub_event(event: dict[str, Any], _hint: Any = None) -> dict[str, Any]:
    """Keep only what locates an error: no request data, cookies, headers, user or locals."""
    request = event.get("request") or {}
    event["request"] = {k: request[k] for k in ("method",) if k in request}
    event.pop("user", None)
    for exc in (event.get("exception") or {}).get("values", []) or []:
        for frame in (exc.get("stacktrace") or {}).get("frames", []) or []:
            frame.pop("vars", None)
    event.pop("breadcrumbs", None)
    return event


def init_error_tracking() -> bool:
    """Start Sentry when SENTRY_DSN is set and sentry-sdk is installed."""
    dsn = os.getenv("SENTRY_DSN", "").strip()
    if not dsn:
        return False
    try:
        import sentry_sdk  # noqa: PLC0415
    except ImportError:
        logging.getLogger(__name__).warning("SENTRY_DSN is set but sentry-sdk is not installed")
        return False
    sentry_sdk.init(
        dsn=dsn,
        send_default_pii=False,
        include_local_variables=False,
        max_breadcrumbs=0,
        traces_sample_rate=0.0,
        before_send=scrub_event,
        environment=os.getenv("GATE_ENVIRONMENT", "production"),
    )
    return True

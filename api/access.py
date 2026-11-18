"""
api/access.py
=============
Who may call the operator endpoints.

Outside hosted mode the original single-tenant rules apply: a request from
this machine, the operator's API token, or the admin token. In hosted mode
only the admin token (``X-Admin-Token``) or a signed-in admin counts; a
request from localhost is not trusted, because behind a reverse proxy every
request arrives from localhost.
"""

from __future__ import annotations

import hmac
import os

from fastapi import HTTPException, Request

from api.auth_routes import csrf_ok, load_session_user
from assistant.hosted import hosted_mode
from assistant.security import api_token, token_ok

LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


def admin_token_ok(request: Request) -> bool:
    token = os.getenv("GATE_ADMIN_TOKEN", "")
    supplied = request.headers.get("x-admin-token", "")
    return bool(token and supplied) and hmac.compare_digest(supplied, token)


def session_is_admin(request: Request) -> bool:
    """A signed-in admin (hosted mode); writes must carry the session's CSRF token."""
    user = load_session_user(request)
    return bool(user and user.get("is_admin") and csrf_ok(request, user))


def is_admin(request: Request) -> bool:
    if admin_token_ok(request):
        return True
    if hosted_mode():
        return session_is_admin(request)
    # Single-tenant: whoever holds the operator's API token is the operator.
    if api_token() and token_ok(request.headers.get("authorization")):
        return True
    host = request.client.host if request.client else ""
    return host in LOCAL_HOSTS and not request.headers.get("x-forwarded-for")


def require_admin(request: Request) -> None:
    """The original operator check (cache, keys), hosted-aware."""
    if not is_admin(request):
        raise HTTPException(
            status_code=403,
            detail=(
                "Only an admin can do this."
                if hosted_mode()
                else "Keys can only be added from this machine, with the API token, or with the admin token."
            ),
        )


def admin_when_hosted(request: Request) -> None:
    """Open in single-tenant mode (as before); admin-only in hosted mode."""
    if hosted_mode() and not is_admin(request):
        raise HTTPException(status_code=403, detail="Only an admin can do this.")

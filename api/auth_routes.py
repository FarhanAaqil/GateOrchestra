"""
api/auth_routes.py
==================
Accounts for hosted mode (GATE_HOSTED=true). Outside hosted mode these
endpoints answer 404 and nothing here is used.

  POST /auth/signup                 create an account and email a verification link
  POST /auth/login                  sign in: sets the session cookie, returns the CSRF token
  POST /auth/logout                 end this session
  GET  /auth/me                     the signed-in user and the CSRF token
  POST /auth/verify                 confirm the email address with the emailed token
  POST /auth/resend-verification    email a new verification link
  POST /auth/forgot                 email a password-reset link (same reply whether or not
                                    the address has an account)
  POST /auth/reset                  set a new password with the emailed token
  POST /auth/login/2fa              finish a login with an authenticator or recovery code
  POST /auth/confirm-email          switch to a new address with the link sent to it

After GATE_LOGIN_FAILURES (5) wrong passwords or codes in 15 minutes an
account's sign-in is paused for the rest of that window, whichever address
the attempts come from, and the owner gets an email saying so.

The session lives in an httpOnly, SameSite=Strict cookie. Every request that
changes something and is authenticated by that cookie must also send the
CSRF token in an ``X-CSRF-Token`` header. All of these endpoints are rate
limited per client address and per email address.
"""

from __future__ import annotations

import hmac
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from assistant.accounts import SESSION_SECONDS, AccountError, Accounts, normalize_email
from assistant.hosted import client_address, hosted_mode
from assistant.mailer import MailError, send_mail
from assistant.shared_limits import SharedRateLimiter

router = APIRouter()

COOKIE = "gate_session"
CSRF_HEADER = "x-csrf-token"
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}

# Kept in the database, so the limits hold across worker processes and restarts.
_by_address = SharedRateLimiter(
    "auth-address", int(os.getenv("GATE_AUTH_RATE_LIMIT_PER_MINUTE", "10")), window=60
)
_by_email = SharedRateLimiter(
    "auth-email", int(os.getenv("GATE_AUTH_EMAIL_LIMIT", "8")), window=900
)
# Failed passwords or codes per account: the lockout.
_failures = SharedRateLimiter(
    "login-failures", int(os.getenv("GATE_LOGIN_FAILURES", "5")), window=900
)


class Credentials(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=1, max_length=256)


class EmailOnly(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)


class TokenOnly(BaseModel):
    token: str = Field(..., min_length=10, max_length=200)


class SecondFactor(BaseModel):
    challenge: str = Field(..., min_length=10, max_length=200)
    code: str = Field(..., min_length=6, max_length=20)


class ResetRequest(BaseModel):
    token: str = Field(..., min_length=10, max_length=200)
    password: str = Field(..., min_length=1, max_length=256)


def accounts() -> Accounts:
    from assistant.service import get_service  # noqa: PLC0415

    return get_service().accounts


def public_url() -> str:
    return os.getenv("GATE_PUBLIC_URL", "http://localhost:3700").rstrip("/")


def cookie_secure() -> bool:
    return os.getenv("GATE_COOKIE_SECURE", "true").strip().lower() not in {"0", "false", "no"}


# Sessions for other routes


def load_session_user(request: Request) -> dict[str, Any] | None:
    """The signed-in user (public fields plus ``csrf``), cached on the request."""
    if not hosted_mode():
        return None
    if hasattr(request.state, "session_user"):
        return request.state.session_user
    found = accounts().session(request.cookies.get(COOKIE))
    user = None
    if found:
        raw, csrf = found
        user = {**accounts().public(raw), "csrf": csrf}
    request.state.session_user = user
    return user


def csrf_ok(request: Request, user: dict[str, Any]) -> bool:
    if request.method in SAFE_METHODS:
        return True
    supplied = request.headers.get(CSRF_HEADER, "")
    return bool(supplied) and hmac.compare_digest(supplied, user["csrf"])


def require_user(request: Request) -> dict[str, Any]:
    """Hosted mode: the signed-in user, with the CSRF header checked on writes."""
    user = load_session_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Log in to continue.")
    if not csrf_ok(request, user):
        raise HTTPException(status_code=403, detail="Missing or wrong CSRF token. Reload the page.")
    return user


def chat_owner(request: Request) -> str | None:
    """Whose rows a chat request reads and writes: None outside hosted mode."""
    if not hosted_mode():
        return None
    user = require_user(request)
    if not user["verified"]:
        raise HTTPException(status_code=403, detail="Confirm your email address first.")
    return str(user["id"])


# Helpers


def _hosted_only() -> None:
    if not hosted_mode():
        raise HTTPException(status_code=404, detail="Accounts are only used in hosted mode.")


def _limit(request: Request, email: str | None = None) -> None:
    peer = request.client.host if request.client else "unknown"
    keys = [("address", client_address(peer, request.headers.get("x-forwarded-for")))]
    if email:
        keys.append(("email", email))
    for kind, key in keys:
        limiter = _by_address if kind == "address" else _by_email
        wait = limiter.check(f"{kind}:{key}")
        if wait:
            raise HTTPException(
                status_code=429,
                detail=f"Too many attempts. Try again in {wait:.0f}s.",
                headers={"Retry-After": str(max(1, round(wait)))},
            )


def _email_key(email: str) -> str | None:
    try:
        return normalize_email(email)
    except AccountError:
        return None


def _send_verification(user: dict[str, Any]) -> None:
    token = accounts().issue_token(user["id"], "verify")
    send_mail(
        user["email"],
        "Confirm your GateOrchestra email",
        "Welcome to GateOrchestra.\n\n"
        f"Confirm this address to start chatting:\n{public_url()}/verify?token={token}\n\n"
        "The link works once and expires in 24 hours. If you did not sign up, ignore this email.",
    )


def _start_session(response: Response, user: dict[str, Any]) -> dict[str, Any]:
    token, csrf = accounts().create_session(user["id"])
    response.set_cookie(
        COOKIE,
        token,
        max_age=SESSION_SECONDS,
        httponly=True,
        secure=cookie_secure(),
        samesite="strict",
        path="/",
    )
    return {"user": accounts().public(user), "csrf_token": csrf}


def _mail_failed(err: MailError) -> HTTPException:
    return HTTPException(status_code=503, detail=str(err))


# Endpoints


@router.post("/auth/signup", status_code=202)
def signup(body: Credentials, request: Request) -> dict[str, str]:
    _hosted_only()
    _limit(request, _email_key(body.email))
    try:
        user, created = accounts().create(body.email, body.password)
    except AccountError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    try:
        if created:
            _send_verification(user)
        else:
            # Same reply as a new sign-up, so the form does not reveal who has an account.
            send_mail(
                user["email"],
                "You already have a GateOrchestra account",
                "Someone tried to sign up with this address, which already has an account.\n\n"
                f"Log in at {public_url()}/login, or reset the password at {public_url()}/forgot.",
            )
    except MailError as err:
        raise _mail_failed(err) from err
    return {"status": "check_email"}


def _locked(email: str | None) -> None:
    wait = _failures.blocked(email) if email else 0.0
    if wait:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Sign-in for this account is paused for {max(1, round(wait / 60))} min after too "
                "many failed attempts. If that was not you, reset your password."
            ),
            headers={"Retry-After": str(max(1, round(wait)))},
        )


def _count_failure(email: str | None) -> None:
    """Count a failed attempt; tell the owner when it locks the account."""
    if not email:
        return
    _failures.check(email)
    if _failures.blocked(email):
        user = accounts().by_email(email)
        if user:
            try:
                send_mail(
                    email,
                    "Sign-in to your GateOrchestra account was paused",
                    "There were several failed attempts to sign in to your account, so sign-in is "
                    "paused for 15 minutes.\n\nIf that was you, wait and try again. If not, "
                    f"reset your password at {public_url()}/forgot.",
                )
            except MailError:
                pass  # the lockout matters more than the notice


@router.post("/auth/login")
def login(body: Credentials, request: Request, response: Response) -> dict[str, Any]:
    _hosted_only()
    email = _email_key(body.email)
    _limit(request, email)
    _locked(email)
    user = accounts().authenticate(body.email, body.password)
    if user is None:
        _count_failure(email)
        raise HTTPException(status_code=401, detail="Wrong email or password.")
    if user.get("totp_enabled_at") is not None:
        # The password was right; a code from the authenticator app finishes the login.
        return {"two_factor_required": True, "challenge": accounts().issue_token(user["id"], "2fa")}
    _failures.clear(user["email"])
    return _start_session(response, user)


@router.post("/auth/login/2fa")
def login_second_factor(body: SecondFactor, request: Request, response: Response) -> dict[str, Any]:
    _hosted_only()
    _limit(request)
    found = accounts().peek_token(body.challenge, "2fa")
    user = accounts().get(found[0]) if found else None
    if user is None:
        raise HTTPException(
            status_code=400, detail="This sign-in expired. Enter your password again."
        )
    _locked(user["email"])
    if not accounts().check_second_factor(user["id"], body.code):
        _count_failure(user["email"])
        raise HTTPException(status_code=401, detail="That code is not right.")
    accounts().consume_token(body.challenge, "2fa")
    _failures.clear(user["email"])
    return _start_session(response, user)


@router.post("/auth/logout")
def logout(request: Request, response: Response) -> dict[str, str]:
    _hosted_only()
    require_user(request)
    accounts().end_session(request.cookies.get(COOKIE, ""))
    response.delete_cookie(COOKIE, path="/", secure=cookie_secure(), samesite="strict")
    return {"status": "logged_out"}


@router.get("/auth/me")
def me(request: Request) -> dict[str, Any]:
    _hosted_only()
    user = require_user(request)
    public = {k: v for k, v in user.items() if k != "csrf"}
    return {"user": public, "csrf_token": user["csrf"]}


@router.post("/auth/verify")
def verify(body: TokenOnly, request: Request) -> dict[str, str]:
    _hosted_only()
    _limit(request)
    user_id = accounts().consume_token(body.token, "verify")
    if user_id is None:
        raise HTTPException(status_code=400, detail="This link is invalid or has expired.")
    accounts().mark_verified(user_id)
    return {"status": "verified"}


@router.post("/auth/resend-verification", status_code=202)
def resend_verification(request: Request) -> dict[str, str]:
    _hosted_only()
    user = require_user(request)
    _limit(request, user["email"])
    if not user["verified"]:
        raw = accounts().get(user["id"])
        if raw:
            try:
                _send_verification(raw)
            except MailError as err:
                raise _mail_failed(err) from err
    return {"status": "check_email"}


@router.post("/auth/forgot", status_code=202)
def forgot(body: EmailOnly, request: Request) -> dict[str, str]:
    _hosted_only()
    email = _email_key(body.email)
    _limit(request, email)
    user = accounts().by_email(email) if email else None
    if user:
        token = accounts().issue_token(user["id"], "reset")
        try:
            send_mail(
                user["email"],
                "Reset your GateOrchestra password",
                f"Choose a new password here:\n{public_url()}/reset?token={token}\n\n"
                "The link works once and expires in 1 hour. If you did not ask for this, ignore it.",
            )
        except MailError as err:
            raise _mail_failed(err) from err
    return {"status": "check_email"}


@router.post("/auth/reset")
def reset(body: ResetRequest, request: Request) -> dict[str, str]:
    _hosted_only()
    _limit(request)
    try:
        from assistant.accounts import check_password  # noqa: PLC0415

        check_password(body.password)
    except AccountError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    user_id = accounts().consume_token(body.token, "reset")
    if user_id is None:
        raise HTTPException(status_code=400, detail="This link is invalid or has expired.")
    accounts().set_password(user_id, body.password)
    accounts().mark_verified(user_id)  # the link proved they read this inbox
    return {"status": "reset"}


@router.post("/auth/confirm-email")
def confirm_email_change(body: TokenOnly, request: Request, response: Response) -> dict[str, str]:
    """Switch to the new address from the link sent to it; every session is signed out."""
    _hosted_only()
    _limit(request)
    found = accounts().consume_token_data(body.token, "email_change")
    if found is None or not found[1]:
        raise HTTPException(status_code=400, detail="This link is invalid or has expired.")
    user_id, new_email = found
    try:
        accounts().change_email(user_id, new_email)
    except AccountError as err:
        raise HTTPException(status_code=409, detail=str(err)) from err
    response.delete_cookie(COOKIE, path="/", secure=cookie_secure(), samesite="strict")
    return {"status": "changed", "email": new_email}

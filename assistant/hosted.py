"""
assistant/hosted.py
===================
Hosted (multi-user) mode: one switch, ``GATE_HOSTED=true``.

With the switch off nothing here changes how the project behaves. With it on:

  * the API refuses to start until the deployment is locked down
    (``hosted_config_problems`` lists what is missing);
  * research and operator endpoints are admin-only;
  * ``X-Forwarded-For`` is believed only when the request comes from a proxy
    listed in ``GATE_TRUSTED_PROXIES`` (IP addresses or CIDR ranges);
  * only cloud providers are used, never a local Ollama.
"""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Mapping

CLOUD_PROVIDERS = {"groq", "gemini"}
MIN_SECRET_LENGTH = 24


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def hosted_mode(env: Mapping[str, str] | None = None) -> bool:
    return _truthy((os.environ if env is None else env).get("GATE_HOSTED"))


def cors_origins(env: Mapping[str, str] | None = None) -> list[str]:
    source = os.environ if env is None else env
    return [o.strip() for o in source.get("GATE_CORS_ORIGINS", "*").split(",") if o.strip()]


def _has_cloud_key(env: Mapping[str, str]) -> bool:
    from assistant.keys import KeyPool

    pool = KeyPool.from_env(dict(env))
    return any(pool.keys.get(p) for p in CLOUD_PROVIDERS)


def hosted_config_problems(env: Mapping[str, str] | None = None) -> list[str]:
    """Why a hosted deployment is not safe to start yet (empty when it is)."""
    source = dict(os.environ if env is None else env)
    problems: list[str] = []

    raw_origins = source.get("GATE_CORS_ORIGINS", "").strip()
    origins = cors_origins(source) if raw_origins else []
    if not origins:
        problems.append("GATE_CORS_ORIGINS must list the site's origins explicitly.")
    elif any("*" in o for o in origins):
        problems.append("GATE_CORS_ORIGINS must not contain '*' in hosted mode.")

    if len(source.get("GATE_ADMIN_TOKEN", "").strip()) < MIN_SECRET_LENGTH:
        problems.append(f"GATE_ADMIN_TOKEN must be set (at least {MIN_SECRET_LENGTH} characters).")

    providers = [
        p.strip().lower()
        for p in source.get("ASSISTANT_PROVIDERS", "groq,gemini").split(",")
        if p.strip()
    ]
    local = [p for p in providers if p not in CLOUD_PROVIDERS]
    if local:
        problems.append(
            f"ASSISTANT_PROVIDERS may only name cloud providers (groq, gemini); found {', '.join(local)}."
        )
    if _truthy(source.get("ASSISTANT_LOCAL_FALLBACK")):
        problems.append("ASSISTANT_LOCAL_FALLBACK must be off: hosted mode never uses Ollama.")
    if source.get("GATE_LLM_PROVIDER", "ollama").strip().lower() not in {"groq", "mock"}:
        problems.append("GATE_LLM_PROVIDER must be groq (or mock) in hosted mode, not ollama.")
    if not _has_cloud_key(source):
        problems.append("A Groq or Gemini API key is required (GROQ_API_KEY or GEMINI_API_KEY).")

    url = source.get("GATE_PUBLIC_URL", "").strip()
    local = url.startswith(("http://localhost", "http://127.0.0.1"))
    if not (url.startswith("https://") or local):
        problems.append(
            "GATE_PUBLIC_URL must be the site's https:// address (used in emailed links)."
        )
    if not source.get("GATE_ADMIN_EMAILS", "").strip():
        problems.append("GATE_ADMIN_EMAILS must name at least one admin account.")
    problems += _mail_problems(source)

    from assistant.userkeys import encryption_key_problem  # noqa: PLC0415

    key_problem = encryption_key_problem(source.get("GATE_ENCRYPTION_KEY", "").strip())
    if key_problem:
        problems.append(key_problem)
    link = source.get("GATE_PAYMENT_LINK", "").strip()
    if link and not link.startswith("https://"):
        problems.append("GATE_PAYMENT_LINK must be an https:// address.")
    return problems


def _mail_problems(env: Mapping[str, str]) -> list[str]:
    backend = env.get("GATE_MAIL_BACKEND", "smtp").strip().lower() or "smtp"
    if backend == "console":
        return []
    if backend != "smtp":
        return ["GATE_MAIL_BACKEND must be smtp or console."]
    problems = []
    if not env.get("GATE_SMTP_HOST", "").strip():
        problems.append("GATE_SMTP_HOST must be set to send account emails.")
    if "@" not in env.get("GATE_MAIL_FROM", ""):
        problems.append("GATE_MAIL_FROM must be the address account emails come from.")
    return problems


def check_hosted_config(env: Mapping[str, str] | None = None) -> None:
    """Raise when hosted mode is on and the deployment is not locked down."""
    if not hosted_mode(env):
        return
    problems = hosted_config_problems(env)
    if problems:
        raise RuntimeError(
            "GATE_HOSTED=true but the deployment is not safe to start:\n  - "
            + "\n  - ".join(problems)
        )


# Client addresses behind a reverse proxy

Network = ipaddress.IPv4Network | ipaddress.IPv6Network


def trusted_proxies(env: Mapping[str, str] | None = None) -> list[Network]:
    source = os.environ if env is None else env
    networks: list[Network] = []
    for item in source.get("GATE_TRUSTED_PROXIES", "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            networks.append(ipaddress.ip_network(item, strict=False))
        except ValueError:
            continue
    return networks


def _is_trusted(address: str, networks: list[Network]) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    return any(ip in net for net in networks)


def client_address(
    peer: str, forwarded_for: str | None, env: Mapping[str, str] | None = None
) -> str:
    """The address to rate-limit on.

    Outside hosted mode, with no proxies configured, this keeps the original
    behaviour (first ``X-Forwarded-For`` entry). Otherwise the header is only
    read when the direct peer is a trusted proxy, and then from the right: the
    first address that is not itself a trusted proxy is the client. Anything a
    client writes further left is ignored, so the header cannot be forged.
    """
    networks = trusted_proxies(env)
    header = (forwarded_for or "").strip()
    if not networks and not hosted_mode(env):
        return header.split(",")[0].strip() or peer
    if not header or not _is_trusted(peer, networks):
        return peer
    hops = [h.strip() for h in header.split(",") if h.strip()]
    for hop in reversed(hops):
        if not _is_trusted(hop, networks):
            return hop
    return hops[0] if hops else peer

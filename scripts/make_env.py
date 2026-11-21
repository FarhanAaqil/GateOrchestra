"""
Write a hosted-mode .env with every secret generated.

    python scripts/make_env.py --domain gate.example.com --admin-email you@example.com

Starts from .env.example, switches hosted mode on, derives the origin and
public URL from the domain, generates GATE_ADMIN_TOKEN and
GATE_ENCRYPTION_KEY, and requires two-factor sign-in for admins. What only
you can supply (provider keys, SMTP account, payment link) is left blank and
listed at the end. An existing file is never overwritten without --force.
"""

from __future__ import annotations

import argparse
import re
import secrets
import sys
from pathlib import Path

from cryptography.fernet import Fernet

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / ".env.example"
NEEDS_YOU = [
    ("GROQ_API_KEY", "a Groq key (also needed for the prompt-injection screen)"),
    ("GATE_SMTP_HOST", "your SMTP server, with GATE_SMTP_USER / GATE_SMTP_PASSWORD"),
    ("GATE_MAIL_FROM", "the address account emails come from"),
    ("GATE_PAYMENT_LINK", "optional: your payment page for the Upgrade button"),
]
_DOMAIN = re.compile(r"^(?=.{1,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}$")


def hosted_values(domain: str, admin_emails: list[str]) -> dict[str, str]:
    origin = f"https://{domain}"
    return {
        "GATE_HOSTED": "true",
        "GATE_DOMAIN": domain,
        "GATE_CORS_ORIGINS": origin,
        "GATE_PUBLIC_URL": origin,
        "GATE_ADMIN_EMAILS": ",".join(admin_emails),
        "GATE_ADMIN_TOKEN": secrets.token_urlsafe(32),
        "GATE_ENCRYPTION_KEY": Fernet.generate_key().decode(),
        "GATE_REQUIRE_ADMIN_2FA": "true",
        "GATE_COOKIE_SECURE": "true",
        "GATE_API_TOKEN": "",
        "GATE_TRUSTED_PROXIES": "172.30.0.0/24",
        "GATE_LLM_PROVIDER": "groq",
        "ASSISTANT_PROVIDERS": "groq,gemini",
        "ASSISTANT_LOCAL_FALLBACK": "false",
        "GATE_MAIL_BACKEND": "smtp",
    }


def render(example: str, values: dict[str, str]) -> str:
    """The example file with ``values`` filled in; keys it lacks are appended."""
    seen: set[str] = set()
    lines = []
    for line in example.splitlines():
        name = (
            line.split("=", 1)[0].strip()
            if "=" in line and not line.lstrip().startswith("#")
            else ""
        )
        if name in values:
            lines.append(f"{name}={values[name]}")
            seen.add(name)
        else:
            lines.append(line)
    missing = [k for k in values if k not in seen]
    if missing:
        lines += ["", "# Added by scripts/make_env.py"] + [f"{k}={values[k]}" for k in missing]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--domain", required=True, help="e.g. gate.example.com (no https://)")
    parser.add_argument(
        "--admin-email", action="append", required=True, help="repeat for several admins"
    )
    parser.add_argument("--out", type=Path, default=ROOT / ".env")
    parser.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = parser.parse_args(argv)

    domain = (
        args.domain.strip().lower().removeprefix("https://").removeprefix("http://").rstrip("/")
    )
    if not _DOMAIN.match(domain):
        print(f"{domain!r} does not look like a domain name.", file=sys.stderr)
        return 1
    emails = [e.strip().lower() for e in args.admin_email if "@" in e]
    if not emails:
        print("Give at least one admin email address.", file=sys.stderr)
        return 1
    if args.out.exists() and not args.force:
        print(f"{args.out} already exists; pass --force to replace it.", file=sys.stderr)
        return 1
    args.out.write_text(
        render(EXAMPLE.read_text(encoding="utf-8"), hosted_values(domain, emails)), encoding="utf-8"
    )
    print(f"Wrote {args.out} with generated secrets. Keep it private (it is git-ignored).")
    print("Still to fill in:")
    for name, what in NEEDS_YOU:
        print(f"  {name:<20} {what}")
    print("Then run: python scripts/check_deploy.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

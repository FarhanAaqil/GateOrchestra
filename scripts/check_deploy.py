"""
Check a hosted deployment, before and after it goes live.

Before starting (on the server, next to .env):

    python scripts/check_deploy.py

  checks the .env the way the API will at start-up (plus what the hosted
  compose file derives from GATE_DOMAIN), that Docker Compose is new enough,
  that ./backups is writable and that configs/plans.yaml loads.

After starting:

    python scripts/check_deploy.py --url https://gate.example.com

  checks the live site from outside: HTTPS works and plain HTTP redirects to
  it, the console and API answer, the API is in hosted mode, the security
  headers are present, and an unauthenticated request is refused cleanly.
  Nothing is created on the server. --allow-http skips the HTTPS checks (for
  a local trial).

Exits with status 1 when anything fails.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PAGE_HEADERS = [
    "content-security-policy",
    "x-content-type-options",
    "x-frame-options",
    "referrer-policy",
]


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def ok(self, what: str) -> None:
        print(f"  ok    {what}")

    def fail(self, what: str) -> None:
        print(f"  FAIL  {what}")
        self.failures.append(what)

    def check(self, condition: bool, what: str) -> None:
        self.ok(what) if condition else self.fail(what)


def effective_env(env_file: Path) -> dict[str, str]:
    """The .env as the hosted compose file will hand it to the API."""
    from dotenv import dotenv_values

    values = {k: v or "" for k, v in (dotenv_values(env_file) or {}).items()}
    domain = values.get("GATE_DOMAIN", "").strip()
    if domain:  # deploy/docker-compose.hosted.yml sets these from the domain
        values.update(
            GATE_HOSTED="true",
            GATE_CORS_ORIGINS=f"https://{domain}",
            GATE_PUBLIC_URL=f"https://{domain}",
            GATE_TRUSTED_PROXIES="172.30.0.0/24",
        )
    return values


def compose_version() -> tuple[int, ...] | None:
    if shutil.which("docker") is None:
        return None
    try:
        out = subprocess.run(
            ["docker", "compose", "version", "--short"], capture_output=True, text=True, timeout=20
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", out.stdout)
    return tuple(int(x) for x in match.groups()) if match else None


def check_config(env_file: Path, report: Report) -> None:
    from assistant.hosted import hosted_config_problems
    from assistant.plans import plan_book

    print(f"Configuration ({env_file})")
    if not env_file.exists():
        report.fail(f"{env_file} is missing; create it with scripts/make_env.py")
        return
    env = effective_env(env_file)
    report.check(
        env.get("GATE_HOSTED", "").lower() in {"1", "true", "yes", "on"}, "hosted mode is on"
    )
    report.check(bool(env.get("GATE_DOMAIN")), "GATE_DOMAIN is set")
    for problem in hosted_config_problems(env):
        report.fail(problem)
    if not hosted_config_problems(env):
        report.ok("the API's start-up checks pass")
    report.check(
        env.get("GATE_REQUIRE_ADMIN_2FA", "").lower() == "true",
        "admins must use two-factor sign-in (recommended)",
    )
    report.check(
        env.get("GATE_MAIL_BACKEND", "smtp") != "console",
        "account emails are really sent (not the console backend)",
    )
    try:
        plans = plan_book()
        report.ok(f"configs/plans.yaml loads ({', '.join(plans.plans)})")
    except Exception as err:  # noqa: BLE001  any problem with the file is a failure here
        report.fail(f"configs/plans.yaml: {err}")

    print("Server")
    version = compose_version()
    if version is None:
        report.fail("Docker with the Compose plugin is not available")
    else:
        report.check(
            version >= (2, 24, 0),
            f"Docker Compose {'.'.join(map(str, version))} (2.24 or newer is needed)",
        )
    backups = ROOT / "backups"
    report.check(
        backups.is_dir() and os.access(backups, os.W_OK),
        "./backups exists and is writable (mkdir backups; chown 10001 backups)",
    )


def fetch(url: str, method: str = "GET", follow: bool = True) -> tuple[int, dict[str, str], str]:
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args: Any, **kwargs: Any) -> None:
            return None

    opener = urllib.request.build_opener() if follow else urllib.request.build_opener(NoRedirect)
    request = urllib.request.Request(
        url, method=method, headers={"User-Agent": "gateorchestra-check-deploy"}
    )
    try:
        with opener.open(request, timeout=20) as response:
            return (
                response.status,
                {k.lower(): v for k, v in response.headers.items()},
                response.read().decode("utf-8", "replace"),
            )
    except urllib.error.HTTPError as err:
        return (
            err.code,
            {k.lower(): v for k, v in err.headers.items()},
            err.read().decode("utf-8", "replace"),
        )


def check_live(url: str, allow_http: bool, report: Report) -> None:
    base = url.rstrip("/")
    print(f"Live site ({base})")
    https = base.startswith("https://")
    if not https and not allow_http:
        report.fail("the URL is not https:// (pass --allow-http for a local trial)")
        return
    try:
        status, headers, body = fetch(base + "/")
    except (urllib.error.URLError, OSError) as err:
        report.fail(f"the site cannot be reached: {err}")
        return
    report.check(status == 200 and 'id="root"' in body, "the console loads")
    for name in PAGE_HEADERS:
        report.check(name in headers, f"the page sends {name}")
    if https:
        report.check(
            "strict-transport-security" in headers,
            "the page sends strict-transport-security (HSTS)",
        )
        try:
            plain, plain_headers, _ = fetch(
                "http://" + base.removeprefix("https://") + "/", follow=False
            )
            report.check(
                plain in (301, 302, 307, 308)
                and plain_headers.get("location", "").startswith("https://"),
                "plain HTTP redirects to HTTPS",
            )
        except (urllib.error.URLError, OSError):
            report.fail(
                "plain HTTP did not answer (port 80 must be open for the redirect and certificates)"
            )
    status, _, body = fetch(base + "/api/health")
    report.check(status == 200, "the API answers /api/health")
    status, _, body = fetch(base + "/api/")
    try:
        hosted = json.loads(body).get("hosted") is True
    except json.JSONDecodeError:
        hosted = False
    report.check(status == 200 and hosted, "the API runs in hosted mode")
    status, headers, _ = fetch(base + "/api/auth/me")
    report.check(status == 401, "a request without a session is refused (401)")
    report.check(
        headers.get("cache-control") == "no-store", "account responses are not cached (no-store)"
    )
    status, _, _ = fetch(base + "/api/assistant/stats")
    report.check(status == 403, "operator statistics are admin-only (403)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--env", type=Path, default=ROOT / ".env")
    parser.add_argument("--url", help="check the live site instead of the local configuration")
    parser.add_argument("--allow-http", action="store_true")
    args = parser.parse_args(argv)
    report = Report()
    if args.url:
        check_live(args.url, args.allow_http, report)
    else:
        check_config(args.env, report)
    if report.failures:
        print(f"\n{len(report.failures)} problem(s) to fix before going live.")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

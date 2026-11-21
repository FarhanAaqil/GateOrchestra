"""
Start the API in hosted mode for the browser test (CI and local runs only).

It runs the real app with GATE_HOSTED=true and a locked-down configuration,
but swaps in the offline mock model so no provider is called and no key is
needed, and writes every account email to a file the test reads links from.
Never use this to serve real users.

    E2E_DATA=/tmp/e2e-hosted E2E_ORIGIN=http://127.0.0.1:4174 python scripts/e2e_hosted_server.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = Path(os.getenv("E2E_DATA", tempfile.mkdtemp(prefix="gate-e2e-")))
ORIGIN = os.getenv("E2E_ORIGIN", "http://127.0.0.1:4174")
PORT = int(os.getenv("E2E_PORT", "8001"))
DATA.mkdir(parents=True, exist_ok=True)

os.environ.update(
    {
        "GATE_HOSTED": "true",
        "GATE_CORS_ORIGINS": ORIGIN,
        "GATE_PUBLIC_URL": ORIGIN,
        "GATE_ADMIN_TOKEN": "e2e-admin-token-" + "x" * 20,
        "GATE_ADMIN_EMAILS": "admin@example.com",
        "ASSISTANT_PROVIDERS": "groq,gemini",
        "GATE_LLM_PROVIDER": "mock",
        "GROQ_API_KEY": "gsk_" + "e2e" * 14,  # never used: the service below is the mock
        "GATE_MAIL_BACKEND": "console",
        "GATE_COOKIE_SECURE": "false",
        "GATE_ENCRYPTION_KEY": "ZTJlLW9ubHkta2V5LW5vdC1mb3ItcmVhbC11c2UtMDA=",
        "GATE_PAYMENT_LINK": "https://pay.example.com/pro?prefilled_email={email}",
        "GATE_AUTH_RATE_LIMIT_PER_MINUTE": "1000",
        "GATE_AUTH_EMAIL_LIMIT": "1000",
        "GATE_RATE_LIMIT_PER_MINUTE": "1000",
        "GATE_USER_RATE_LIMIT_PER_MINUTE": "1000",
        "ASSISTANT_DB_PATH": str(DATA / "assistant.db"),
        "ASSISTANT_ENV_PATH": str(DATA / "keys.env"),
        "ASSISTANT_WARMUP": "0",
    }
)

import uvicorn  # noqa: E402

import api.auth_routes  # noqa: E402
from assistant import mailer  # noqa: E402
from assistant.config import AssistantSettings  # noqa: E402
from assistant.service import AssistantService, set_service  # noqa: E402

MAIL_LOG = DATA / "mail.log"
_send = mailer.send_mail


def send_and_log(to: str, subject: str, body: str) -> None:
    _send(to, subject, body)
    with MAIL_LOG.open("a", encoding="utf-8") as log:
        log.write(f"TO {to}\nSUBJECT {subject}\n{body}\n----\n")


mailer.send_mail = send_and_log
api.auth_routes.send_mail = send_and_log
set_service(
    AssistantService(
        AssistantSettings(
            db_path=DATA / "assistant.db", env_path=DATA / "keys.env", providers=["mock"]
        )
    )
)

from api.main import app  # noqa: E402

if __name__ == "__main__":
    print(f"hosted e2e server on 127.0.0.1:{PORT}, data in {DATA}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")

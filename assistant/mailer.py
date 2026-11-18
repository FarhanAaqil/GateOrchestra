"""
assistant/mailer.py
===================
Account emails (verification and password reset).

GATE_MAIL_BACKEND picks how they are sent:

  smtp     (default) through GATE_SMTP_HOST:GATE_SMTP_PORT (587) with STARTTLS,
           signing in with GATE_SMTP_USER / GATE_SMTP_PASSWORD when set, from
           GATE_MAIL_FROM. Port 465 uses implicit TLS instead.
  console  nothing is sent; the message is kept in memory and its subject is
           logged. For local testing only: links in it would sign a user in.

Message bodies are never written to the logs.
"""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
from collections import deque
from email.message import EmailMessage

logger = logging.getLogger(__name__)

# The last messages the console backend "sent" (tests and local trials read them).
OUTBOX: deque[dict[str, str]] = deque(maxlen=50)


class MailError(RuntimeError):
    pass


def mail_backend() -> str:
    return os.getenv("GATE_MAIL_BACKEND", "smtp").strip().lower() or "smtp"


def send_mail(to: str, subject: str, body: str) -> None:
    if mail_backend() == "console":
        OUTBOX.append({"to": to, "subject": subject, "body": body})
        logger.warning("console mail backend: %r kept in memory, not sent", subject)
        return
    message = EmailMessage()
    message["From"] = os.getenv("GATE_MAIL_FROM", "")
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)
    host = os.getenv("GATE_SMTP_HOST", "")
    port = int(os.getenv("GATE_SMTP_PORT", "587"))
    user, password = os.getenv("GATE_SMTP_USER", ""), os.getenv("GATE_SMTP_PASSWORD", "")
    context = ssl.create_default_context()
    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, context=context, timeout=20) as smtp:
                if user:
                    smtp.login(user, password)
                smtp.send_message(message)
        else:
            with smtplib.SMTP(host, port, timeout=20) as smtp:
                smtp.starttls(context=context)
                if user:
                    smtp.login(user, password)
                smtp.send_message(message)
    except (OSError, smtplib.SMTPException) as err:
        logger.error("could not send %r: %s", subject, type(err).__name__)
        raise MailError("The email could not be sent. Try again later.") from err

# Running GateOrchestra as a hosted service

`GATE_HOSTED=true` turns the research console into a multi-user product: accounts,
per-person data, plans with limits, and admin-only operator pages. With it off (the
default), nothing changes: the benchmark, the shared single-tenant chat and every test
behave as before.

This guide covers one VM with Docker, a domain and HTTPS. Budget about an hour.

## What hosted mode does

| Area | Behaviour |
| --- | --- |
| Start-up | The API refuses to start until the settings below are safe (explicit origins, secrets, cloud providers only). The error lists everything missing. |
| Accounts | Email + password (argon2id). Email confirmation before chatting; password reset by email. Links are single-use and expire. |
| Sessions | httpOnly, `SameSite=Strict`, `Secure` cookie; every write also needs the CSRF token. `GATE_API_TOKEN` is not used. |
| Admins | The addresses in `GATE_ADMIN_EMAILS`, once confirmed. Only admins see or can call Admin, Run a Task, Token Accountant, stats, status, metrics, the cache and key endpoints. `X-Admin-Token: $GATE_ADMIN_TOKEN` works for scripts. |
| Data | Conversations, messages, documents, chunks, ratings, events and the answer cache all belong to one person. Anyone else gets 404. |
| Limits | Per plan (`configs/plans.yaml`): answers per day, stored documents, daily tokens, answers in progress. A global daily token cap is your off switch. Plus a per-person rate limit and sandbox rationing. |
| Own keys | People can add their own Groq or Gemini key in Settings. It is encrypted at rest, never returned by the API, and its tokens do not count toward the plan. |
| Public pages | Landing page, Pricing, Benchmark, Results, Baselines, Gate Inspector, About, and the legal drafts. |

## 1. Machine and domain

1. A Linux VM with 2 vCPU and 4 GB RAM is enough. The models run at Groq/Google, not here.
2. Install Docker Engine with the Compose plugin, version 2.24 or newer.
3. Point an `A` (and `AAAA` if you have IPv6) record for your domain at the VM.
4. Open ports 80 and 443 only. Caddy gets and renews the TLS certificate itself.

## 2. Settings

```bash
git clone https://github.com/FarhanAaqil/GateOrchestra.git && cd GateOrchestra
cp .env.example .env
mkdir -p backups && sudo chown 10001 backups   # the containers run as uid 10001
```

Fill in `.env`. These are required; the API will not start without them:

| Setting | Value |
| --- | --- |
| `GATE_DOMAIN` | `gate.example.com` (the compose file derives `GATE_CORS_ORIGINS` and `GATE_PUBLIC_URL` from it) |
| `GATE_ADMIN_TOKEN` | 24+ random characters: `openssl rand -hex 24` |
| `GATE_ADMIN_EMAILS` | your email address(es), comma-separated |
| `GATE_ENCRYPTION_KEY` | `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`. Store it in your password manager: losing it makes people's saved keys unreadable. |
| `GROQ_API_KEY` (and/or `GEMINI_API_KEY`) | the operator's keys; extra keys in `GROQ_API_KEYS` rotate on rate limits |
| `ASSISTANT_PROVIDERS` | `groq,gemini` (no `ollama`, and `ASSISTANT_LOCAL_FALLBACK=false`) |
| `GATE_SMTP_HOST`, `GATE_SMTP_PORT`, `GATE_SMTP_USER`, `GATE_SMTP_PASSWORD`, `GATE_MAIL_FROM` | any SMTP provider. Port 587 uses STARTTLS, 465 implicit TLS. |

Optional: `GATE_PAYMENT_LINK` (see Payments), `GATE_GLOBAL_DAILY_TOKENS`, `SENTRY_DSN`,
`GATE_BACKUP_KEEP_DAYS`. Leave `GATE_API_TOKEN` empty and `GATE_COOKIE_SECURE=true`.

> Check your model providers' terms before charging people. Free-tier keys are often
> not meant for commercial use, and their rate limits will be hit quickly.

## 3. Start

```bash
docker compose -f docker-compose.yml -f deploy/docker-compose.hosted.yml up -d --build
docker compose -f docker-compose.yml -f deploy/docker-compose.hosted.yml logs -f api
```

If a setting is missing, the API log says exactly which, and the container restarts
until you fix `.env`. Then open `https://your-domain`, sign up with an address from
`GATE_ADMIN_EMAILS`, and confirm it from your inbox. You are now an admin, and the
Admin page appears in the menu.

Requests flow Caddy → console (nginx) → API. The API believes `X-Forwarded-For` only
from the Docker network `172.30.0.0/24` (`GATE_TRUSTED_PROXIES`) and reads it from the
right, so a client cannot fake its address to dodge the rate limits.

## 4. Payments (version 1: manual)

1. Create a payment link at your provider (for example a Stripe Payment Link for the
   Pro price) and set `GATE_PAYMENT_LINK` to it. `{email}` in the link is replaced with
   the person's address, for example `https://buy.stripe.com/...?prefilled_email={email}`.
   Check your provider's current documentation for the parameters it supports.
2. A person clicks **Upgrade** on the Pricing page and pays.
3. You see the payment, open **Admin → People and plans**, find the email and press
   **Pro +31 d**. A second payment extends from the end of the current period.
4. When the period ends the person is on Free again automatically.

Version 2 (upgrading automatically from the provider's webhook) is not built yet.

## 5. Backups and the restore test

The `backup` service copies the database every 24 hours into `./backups`, restores each
copy to a temporary file, and checks it (`PRAGMA integrity_check`, every table readable)
before deleting copies older than `GATE_BACKUP_KEEP_DAYS`. The database is everything:
accounts, conversations, uploaded documents (their text), usage and encrypted keys.

- Copy `./backups` off the machine (for example `rclone` or your provider's snapshots).
- Test a copy by hand now and then:
  `docker compose ... run --rm backup "python scripts/backup_db.py --verify /backups/<file>"`
- To restore: stop the API, write the copy over the database, and start it again:
  ```bash
  C="docker compose -f docker-compose.yml -f deploy/docker-compose.hosted.yml"
  $C stop api
  $C run --rm backup "rm -f /app/logs/assistant.db-wal /app/logs/assistant.db-shm && gunzip -c /backups/<file> > /app/logs/assistant.db"
  $C start api
  ```
- Keep `.env` in your password manager, not in the backups.

## 6. Monitoring

- **Uptime:** point an external checker (UptimeRobot, Better Stack, your cloud's monitor)
  at `https://your-domain/api/health` every minute. `docker compose ps` shows the API's
  own health check; Docker restarts it if it exits, not merely when it reports unhealthy.
- **Errors:** set `SENTRY_DSN` and add `sentry-sdk` to the image (`pip install sentry-sdk`).
  Reports carry no request body, cookies, headers, local variables or user details.
- **Logs:** the API writes one JSON line per request: time, request id (shared with
  nginx), user id, method, path, status and duration. It never logs message content,
  documents, query strings or keys. Usage numbers are on the Admin page and at
  `/api/metrics` (with `X-Admin-Token`).
- **Costs:** the global cap (`global_daily_tokens` in `configs/plans.yaml`, or
  `GATE_GLOBAL_DAILY_TOKENS`) stops the service for the day once it is reached.

## 7. The code sandbox

The research agent can run Python in a locked-down container: no network, read-only,
every Linux capability dropped, 256 MB of memory, 15 seconds, and rationed per person in
hosted mode. To start those containers the API needs Docker itself. By default the API
runs inside a container without Docker access, so the Python tool is simply unavailable
and every other feature works.

To enable it, either:

- run the API directly on the VM (`pip install .`, then `uvicorn api.main:app --port 3700`
  as an unprivileged user in the `docker` group, with the same `.env`), point the console's
  `/api/` proxy in `frontend-react/nginx.conf` at the host instead of `api:3700`, and
  `docker pull python:3.12-slim`; or
- mount `/var/run/docker.sock` into the API container. Only do this if you accept that
  anyone who breaks into the API container then controls the VM.

## 8. Plans, legal pages, updates

- Change limits and prices in `configs/plans.yaml`. The file is re-read when it changes.
- Before going live, have the Terms, Privacy and Acceptable-use drafts (`/terms`,
  `/privacy`, `/acceptable-use`) reviewed by a person qualified for your country and
  fill in the bracketed parts. They are marked as drafts until you edit
  `frontend-react/src/pages/Legal.tsx`.
- To update: `git pull` and run the same `up -d --build`. Run `scripts/backup_db.py` first.

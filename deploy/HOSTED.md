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
| Accounts | Email + password (argon2id). Email confirmation before chatting; password reset and email change by emailed link. Links are single-use and expire. |
| Sign-in protection | Optional two-factor sign-in (any authenticator app, with ten one-time recovery codes). After 5 failed passwords or codes in 15 minutes the account's sign-in pauses and the owner is emailed. |
| Sessions | httpOnly, `SameSite=Strict`, `Secure` cookie; every write also needs the CSRF token. `GATE_API_TOKEN` is not used. |
| Admins | The addresses in `GATE_ADMIN_EMAILS`, once confirmed, and with `GATE_REQUIRE_ADMIN_2FA=true` only once they have switched on two-factor sign-in. Only admins see or can call Admin, Run a Task, Token Accountant, stats, status, metrics, the cache and key endpoints. `X-Admin-Token: $GATE_ADMIN_TOKEN` works for scripts. |
| Data | Conversations, messages, documents, chunks, ratings, events and the answer cache all belong to one person. Anyone else gets 404. |
| Limits | Per plan (`configs/plans.yaml`): answers per day, stored documents, daily tokens, answers in progress. A global daily token cap is your off switch. Plus a per-person rate limit and sandbox rationing. All of them live in the database, so they hold across the API's worker processes and restarts. |
| Own keys | People can add their own Groq, Gemini or OpenAI-compatible key in Settings. It is encrypted at rest, never returned by the API, and its tokens do not count toward the plan. |
| Headers | Every page and API response carries security headers, including a strict Content-Security-Policy; anything account-related is sent with `Cache-Control: no-store`. |
| Public pages | Landing page, Pricing, Benchmark, Results, Baselines, Gate Inspector, About, and the legal drafts. |

## 1. Machine and domain

1. A Linux VM with 2 vCPU and 4 GB RAM is enough. The models run at Groq/Google, not here.
2. Install Docker Engine with the Compose plugin, version 2.24 or newer.
3. Point an `A` (and `AAAA` if you have IPv6) record for your domain at the VM.
4. Open ports 80 and 443 only. Caddy gets and renews the TLS certificate itself.

## 2. Settings

```bash
git clone https://github.com/FarhanAaqil/GateOrchestra.git && cd GateOrchestra
pip install cryptography python-dotenv pyyaml   # for the two helper scripts
python scripts/make_env.py --domain gate.example.com --admin-email you@example.com
mkdir -p backups && sudo chown 10001 backups   # the containers run as uid 10001
```

`make_env.py` writes `.env` with every secret generated (admin token, encryption key),
hosted mode on, admin two-factor required and the origins derived from the domain. It
lists what only you can fill in. These are required; the API will not start without them:

| Setting | Value |
| --- | --- |
| `GATE_DOMAIN` | `gate.example.com` (the compose file derives `GATE_CORS_ORIGINS` and `GATE_PUBLIC_URL` from it) |
| `GATE_ADMIN_TOKEN` | 24+ random characters: `openssl rand -hex 24` |
| `GATE_ADMIN_EMAILS` | your email address(es), comma-separated |
| `GATE_ENCRYPTION_KEY` | `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`. Store it in your password manager: losing it makes people's saved keys unreadable. |
| `GROQ_API_KEY` (and/or `GEMINI_API_KEY`, `OPENAI_API_KEY`) | the operator's keys; extra keys in `GROQ_API_KEYS` rotate on rate limits. A Groq key is also what the prompt-injection screen runs on; without one you must set `ASSISTANT_GUARD=false` deliberately. |
| `ASSISTANT_PROVIDERS` | `groq,gemini`, and/or `openai` for any OpenAI-compatible API (`ASSISTANT_OPENAI_BASE`, `ASSISTANT_OPENAI_MODEL`). Never `ollama`. |
| `GATE_SMTP_HOST`, `GATE_SMTP_PORT`, `GATE_SMTP_USER`, `GATE_SMTP_PASSWORD`, `GATE_MAIL_FROM` | any SMTP provider. Port 587 uses STARTTLS, 465 implicit TLS. |

Optional: `GATE_PAYMENT_LINK` (see Payments), `GATE_GLOBAL_DAILY_TOKENS`, `SENTRY_DSN`,
`GATE_BACKUP_KEEP_DAYS`, `GATE_WORKERS` (API processes, default 2). Leave `GATE_API_TOKEN`
empty and `GATE_COOKIE_SECURE=true`.

Then check everything before starting:

```bash
python scripts/check_deploy.py
```

### Choosing a provider and planning capacity

Free tiers are rate-limited and are often not licensed for commercial use, so read your
provider's terms before charging anyone. For a paid service:

- **A paid Groq account** keeps the models the gate's settings were tuned with. Add the
  paid key as `GROQ_API_KEY`; nothing else changes.
- **Any OpenAI-compatible provider** (OpenAI, OpenRouter, Together, Fireworks, DeepInfra,
  or your own vLLM server): set `ASSISTANT_PROVIDERS=openai,groq`, `OPENAI_API_KEY`,
  `ASSISTANT_OPENAI_BASE` and `ASSISTANT_OPENAI_MODEL`. Per-role models go in
  `ASSISTANT_OPENAI_PROBE_MODEL`, `..._FAST_MODEL`, `..._ANSWER_MODEL`, `..._AGENT_MODEL`.
  Keep a Groq key for the prompt-injection screen.
- **Rough capacity:** a chat turn costs a few thousand tokens and the plans cap a Pro
  user at 1,000,000 tokens a day. Multiply your expected daily active users by their
  average daily tokens, compare with your provider's limits and price list, and set
  `global_daily_tokens` in `configs/plans.yaml` (or `GATE_GLOBAL_DAILY_TOKENS`) below
  what you are willing to pay for in a day.

## 3. Start

```bash
docker compose -f docker-compose.yml -f deploy/docker-compose.hosted.yml up -d --build
docker compose -f docker-compose.yml -f deploy/docker-compose.hosted.yml logs -f api
```

If a setting is missing, the API log says exactly which, and the container restarts
until you fix `.env`. Then open `https://your-domain`, sign up with an address from
`GATE_ADMIN_EMAILS`, and confirm it from your inbox. You are now an admin, and the
Admin page appears in the menu.

Once it answers, check it from outside (any computer):

```bash
python scripts/check_deploy.py --url https://gate.example.com
```

It confirms HTTPS and the HTTP redirect, the console and API, hosted mode, the security
headers, and that account and admin endpoints refuse anonymous requests. Then switch on
two-factor sign-in in Settings (admins must when `GATE_REQUIRE_ADMIN_2FA=true`).

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

Day-to-day operations, all done at your payment provider plus one click here:

| Situation | What to do |
| --- | --- |
| Someone paid | Admin → People and plans → find the email on the payment → **Pro +31 d** (or the days they paid for). |
| Paid with a different email | Ask them for their account email, or search by the payment's email; set the plan on the account they sign in with. |
| Refund | Refund at the provider, then press **Free** for that person here (it ends the paid period at once). |
| Invoice or receipt | Your provider issues these (Stripe, Lemon Squeezy and Paddle all send receipts). Turn on automatic receipts there. |
| Sales tax / VAT | Depends on where you and your customers are. A merchant-of-record provider (Lemon Squeezy, Paddle) handles it for you; with Stripe you register yourself (Stripe Tax can calculate it). Ask an accountant. |
| Plan lapsed but they paid late | Press **Pro +31 d** again; it starts from today when the old period has ended. |

Automatic upgrades from the provider's webhook are deliberately not built: you chose
to keep payments manual, which needs no webhook endpoint exposed to the internet.

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
every Linux capability dropped, an unprivileged user, 256 MB of memory, half a CPU and
15 seconds, rationed per person in hosted mode. It is off unless you start the
`sandbox` profile:

```bash
echo "ASSISTANT_SANDBOX_URL=http://sandbox:8090" >> .env   # GATE_SANDBOX_TOKEN is already in .env
docker compose -f docker-compose.yml -f deploy/docker-compose.hosted.yml --profile sandbox up -d --build
```

The sandbox service (`sandbox_service/`) is the only container with the Docker socket.
It accepts exactly one request, "run this code", from the API with `GATE_SANDBOX_TOKEN`,
and can only start the fixed container described above. The API container never gets
Docker access, so a bug in the API cannot be turned into control of the server. The
service is reachable only on the stack's private network. Its own risk is the usual one
for anything holding the Docker socket, so it stays a few dozen lines of standard-library
Python with no other inputs. Without the profile, every other feature works and the agent
simply has no Python tool.

## 8. Security checklist

- Generated secrets (`make_env.py`) and `GATE_REQUIRE_ADMIN_2FA=true` for admin accounts.
- Every admin switches on two-factor sign-in and stores the recovery codes offline.
- `.env` is readable only by you (`chmod 600 .env`); it holds the encryption key and your
  provider keys.
- CI audits the Python runtime dependencies (`pip-audit`) and the shipped console
  (`npm audit --omit=dev`) on every push. The remaining npm advisories are in
  development-only tools (the test browser driver and the Vite dev server), which are
  never deployed.
- Keep the VM updated (`unattended-upgrades` on Ubuntu) and rebuild the images monthly to
  pick up base-image fixes: `docker compose ... build --pull && docker compose ... up -d`.

## 9. Plans, legal pages, updates

- Change limits and prices in `configs/plans.yaml`. The file is re-read when it changes.
- Before going live, have the Terms, Privacy and Acceptable-use drafts (`/terms`,
  `/privacy`, `/acceptable-use`) reviewed by a person qualified for your country and
  fill in the bracketed parts. They are marked as drafts until you edit
  `frontend-react/src/pages/Legal.tsx`.
- To update: `git pull` and run the same `up -d --build`. Run `scripts/backup_db.py` first.

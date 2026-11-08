"""
assistant/sandbox.py
====================
Run model-written Python in a locked-down Docker container.

Each run gets a fresh container: no network, read-only filesystem with a small
/tmp, an unprivileged user, every Linux capability dropped, no privilege
escalation, 256 MB of memory, half a CPU, 64 processes and 15 seconds. The
code is passed on stdin, never written into the image. When Docker is not
running the tool says so instead of falling back to running code on the host.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid

IMAGE = os.getenv("ASSISTANT_SANDBOX_IMAGE", "python:3.12-slim")
TIMEOUT_SECONDS = 15
MAX_OUTPUT = 4000
_STATUS: dict[str, object] = {"checked": 0.0, "ok": False, "why": ""}
STATUS_TTL = 60


def sandbox_status() -> tuple[bool, str]:
    """(available, reason), cached for a minute."""
    if time.time() - float(_STATUS["checked"]) < STATUS_TTL:
        return bool(_STATUS["ok"]), str(_STATUS["why"])
    ok, why = _probe()
    _STATUS.update(checked=time.time(), ok=ok, why=why)
    return ok, why


def _probe() -> tuple[bool, str]:
    if shutil.which("docker") is None:
        return False, "Docker is not installed."
    try:
        info = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, "Docker did not respond."
    if info.returncode != 0:
        return (
            False,
            "Docker is installed but not running. Start Docker Desktop to enable code execution.",
        )
    image = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, timeout=20)
    if image.returncode != 0:
        return False, f"The sandbox image is missing. Run: docker pull {IMAGE}"
    return True, "ready"


def docker_command(name: str) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "-i",
        "--name",
        name,
        "--network",
        "none",
        "--memory",
        "256m",
        "--memory-swap",
        "256m",
        "--cpus",
        "0.5",
        "--pids-limit",
        "64",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,size=16m,noexec",
        "--user",
        "65534:65534",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        IMAGE,
        "python",
        "-I",
        "-",
    ]


def run_python(code: str) -> str:
    """Run ``code`` in the sandbox and return its output (or why it could not run)."""
    if not code.strip():
        return "Error: no code given."
    ok, why = sandbox_status()
    if not ok:
        return f"Code execution is unavailable: {why}"
    name = f"gateorchestra-sbx-{uuid.uuid4().hex[:10]}"
    try:
        done = subprocess.run(
            docker_command(name),
            input=code,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=20)
        return f"Error: the code ran longer than {TIMEOUT_SECONDS}s and was stopped."
    except OSError as err:
        return f"Error: could not start the sandbox ({err})."
    out = (done.stdout or "") + (f"\n[stderr]\n{done.stderr}" if done.stderr.strip() else "")
    out = out.strip() or "(no output)"
    if len(out) > MAX_OUTPUT:
        out = out[:MAX_OUTPUT] + f"\n… output truncated at {MAX_OUTPUT} characters"
    return f"exit code {done.returncode}\n{out}"

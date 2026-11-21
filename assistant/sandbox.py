"""
assistant/sandbox.py
====================
Run model-written Python in a locked-down Docker container.

Each run gets a fresh container: no network, read-only filesystem with a small
/tmp, an unprivileged user, every Linux capability dropped, no privilege
escalation, 256 MB of memory, half a CPU, 64 processes and 15 seconds. The
code is passed on stdin, never written into the image. When Docker is not
running the tool says so instead of falling back to running code on the host.

In hosted mode runs are also rationed: at most GATE_SANDBOX_PER_USER (1) at a
time per person and GATE_SANDBOX_MAX_CONCURRENT (4) across everyone. A run
that cannot get a slot within a few seconds is refused, not queued forever.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
import uuid
from collections import defaultdict
from typing import Any

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


class SandboxLimits:
    """Slots for sandbox runs: a cap per person and one across everyone."""

    def __init__(self, per_user: int, total: int) -> None:
        self.per_user = max(1, per_user)
        self._total = threading.BoundedSemaphore(max(1, total))
        self._running: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def acquire(self, user_id: str, wait: float = 5.0) -> str | None:
        """None when a slot was taken (release it afterwards), else why not."""
        with self._lock:
            if self._running[user_id] >= self.per_user:
                return "you already have code running; wait for it to finish"
            self._running[user_id] += 1
        if not self._total.acquire(timeout=wait):
            with self._lock:
                self._running[user_id] -= 1
            return "the sandbox is busy with other people's code; try again shortly"
        return None

    def release(self, user_id: str) -> None:
        self._total.release()
        with self._lock:
            self._running[user_id] = max(0, self._running[user_id] - 1)


LIMITS = SandboxLimits(
    int(os.getenv("GATE_SANDBOX_PER_USER", "1")), int(os.getenv("GATE_SANDBOX_MAX_CONCURRENT", "4"))
)


class SharedSandboxLimits:
    """The same rationing, kept in the database so every worker process shares it."""

    TTL = TIMEOUT_SECONDS + 45  # a crashed run's slot frees itself

    def __init__(self, per_user: int, total: int, leases: Any = None) -> None:
        from assistant.shared_limits import Leases  # noqa: PLC0415

        self.per_user = max(1, per_user)
        self.total = max(1, total)
        self.leases = leases or Leases()
        self._held: dict[str, list[str]] = {}

    def acquire(self, user_id: str, wait: float = 5.0) -> str | None:
        if self.leases.held(f"sandbox-user:{user_id}") >= self.per_user:
            return "you already have code running; wait for it to finish"
        deadline = time.time() + wait
        while True:
            taken = self.leases.acquire_all(
                [(f"sandbox-user:{user_id}", self.per_user), ("sandbox-all", self.total)], self.TTL
            )
            if taken:
                self._held.setdefault(user_id, []).append(",".join(taken))
                return None
            if time.time() >= deadline:
                return "the sandbox is busy with other people's code; try again shortly"
            time.sleep(0.25)

    def release(self, user_id: str) -> None:
        held = self._held.get(user_id) or []
        if held:
            self.leases.release(*held.pop().split(","))


def default_limits() -> SandboxLimits | SharedSandboxLimits:
    """Shared limits when hosted (several workers), the in-process ones otherwise."""
    from assistant.hosted import hosted_mode  # noqa: PLC0415

    if hosted_mode():
        return SharedSandboxLimits(
            LIMITS.per_user, int(os.getenv("GATE_SANDBOX_MAX_CONCURRENT", "4"))
        )
    return LIMITS


def run_python_for(
    user_id: str, code: str, limits: SandboxLimits | SharedSandboxLimits | None = None
) -> str:
    """``run_python`` within one person's share of the sandbox (hosted mode)."""
    limits = limits or default_limits()
    refused = limits.acquire(user_id)
    if refused:
        return f"Code execution is unavailable right now: {refused}."
    try:
        return run_python(code)
    finally:
        limits.release(user_id)


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

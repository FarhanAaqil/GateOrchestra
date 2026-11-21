"""
GateOrchestra code sandbox service.

The research agent can run short Python programs. Starting a locked-down
container needs the Docker socket, and whoever holds that socket controls the
machine, so the API itself never gets it. This small service does instead, and
it can do exactly one thing: run the given code in the same fixed container
the API would have used (no network, read-only root, every capability dropped,
an unprivileged user, 256 MB, half a CPU, 64 processes, 15 seconds).

    GET  /status   {"ok": true} once Docker and the image are ready
    POST /run      {"code": "..."} with "Authorization: Bearer $GATE_SANDBOX_TOKEN"
                   -> {"output": "exit code 0\\n..."}

Only the API, on the private Docker network, can reach it. Standard library
only: it talks to the Docker Engine API over /var/run/docker.sock directly.
"""

from __future__ import annotations

import hmac
import http.client
import json
import os
import socket
import struct
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import quote

IMAGE = os.getenv("SANDBOX_IMAGE", "python:3.12-slim")
TIMEOUT_SECONDS = int(os.getenv("SANDBOX_TIMEOUT", "15"))
MAX_OUTPUT = 4000
MAX_CODE = 64_000
SOCKET = os.getenv("DOCKER_SOCKET", "/var/run/docker.sock")
SLOTS = threading.BoundedSemaphore(int(os.getenv("SANDBOX_MAX_CONCURRENT", "4")))


def container_spec(code: str) -> dict[str, Any]:
    """The only container this service ever creates."""
    return {
        "Image": IMAGE,
        "Cmd": ["python", "-I", "-c", code],
        "User": "65534:65534",
        "NetworkDisabled": True,
        "HostConfig": {
            "NetworkMode": "none",
            "Memory": 256 * 1024 * 1024,
            "MemorySwap": 256 * 1024 * 1024,
            "NanoCpus": 500_000_000,
            "PidsLimit": 64,
            "ReadonlyRootfs": True,
            "Tmpfs": {"/tmp": "rw,size=16m,noexec"},
            "CapDrop": ["ALL"],
            "SecurityOpt": ["no-new-privileges"],
            "AutoRemove": False,
        },
    }


def demux(raw: bytes) -> tuple[str, str]:
    """Split Docker's multiplexed log stream into (stdout, stderr)."""
    out, err, i = bytearray(), bytearray(), 0
    while i + 8 <= len(raw):
        kind, size = raw[i], struct.unpack(">I", raw[i + 4 : i + 8])[0]
        chunk = raw[i + 8 : i + 8 + size]
        (err if kind == 2 else out).extend(chunk)
        i += 8 + size
    return out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


def format_output(code: int, stdout: str, stderr: str) -> str:
    out = stdout + (f"\n[stderr]\n{stderr}" if stderr.strip() else "")
    out = out.strip() or "(no output)"
    if len(out) > MAX_OUTPUT:
        out = out[:MAX_OUTPUT] + f"\n… output truncated at {MAX_OUTPUT} characters"
    return f"exit code {code}\n{out}"


class _UnixConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self._path)


class Docker:
    """The few Docker Engine API calls the sandbox needs."""

    def __init__(self, path: str = SOCKET) -> None:
        self.path = path

    def call(
        self, method: str, url: str, body: Any = None, timeout: float = 30
    ) -> tuple[int, bytes]:
        conn = _UnixConnection(self.path, timeout)
        try:
            payload = json.dumps(body).encode() if body is not None else None
            headers = {"Content-Type": "application/json"} if payload is not None else {}
            conn.request(method, url, body=payload, headers=headers)
            response = conn.getresponse()
            return response.status, response.read()
        finally:
            conn.close()

    def ready(self) -> tuple[bool, str]:
        try:
            status, _ = self.call("GET", "/_ping", timeout=5)
        except OSError:
            return False, "Docker is not reachable from the sandbox service."
        if status != 200:
            return False, "Docker did not answer."
        name, _, tag = IMAGE.partition(":")
        status, _ = self.call("GET", f"/images/{quote(IMAGE, safe='')}/json")
        if status != 200:
            self.call(
                "POST",
                f"/images/create?fromImage={quote(name)}&tag={quote(tag or 'latest')}",
                timeout=600,
            )
            status, _ = self.call("GET", f"/images/{quote(IMAGE, safe='')}/json")
        return (
            (True, "ready") if status == 200 else (False, f"The sandbox image {IMAGE} is missing.")
        )

    def run(self, code: str) -> str:
        name = f"gateorchestra-sbx-{uuid.uuid4().hex[:10]}"
        status, body = self.call("POST", f"/containers/create?name={name}", container_spec(code))
        if status != 201:
            return f"Error: could not create the sandbox ({status})."
        cid = json.loads(body)["Id"]
        try:
            self.call("POST", f"/containers/{cid}/start")
            try:
                _, waited = self.call("POST", f"/containers/{cid}/wait", timeout=TIMEOUT_SECONDS)
                exit_code = int(json.loads(waited or b"{}").get("StatusCode", -1))
            except TimeoutError:
                self.call("POST", f"/containers/{cid}/kill")
                return f"Error: the code ran longer than {TIMEOUT_SECONDS}s and was stopped."
            _, logs = self.call("GET", f"/containers/{cid}/logs?stdout=1&stderr=1")
            stdout, stderr = demux(logs)
            return format_output(exit_code, stdout, stderr)
        finally:
            self.call("DELETE", f"/containers/{cid}?force=1")


def make_handler(docker: Any, token: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "gateorchestra-sandbox"

        def _reply(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: Any) -> None:  # never log code or output
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/status":
                self._reply(404, {"error": "not found"})
                return
            ok, why = docker.ready()
            self._reply(200, {"ok": ok, "why": why})

        def do_POST(self) -> None:  # noqa: N802
            # Read the (bounded) body before any reply, so the client always receives the
            # answer instead of a connection reset part-way through its upload.
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(min(max(length, 0), 1_000_000))
            if self.path != "/run":
                self._reply(404, {"error": "not found"})
                return
            supplied = self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
            if not token or not supplied or not hmac.compare_digest(supplied, token):
                self._reply(401, {"error": "unauthorised"})
                return
            if length <= 0 or length > MAX_CODE + 1000:
                self._reply(413, {"error": "the code is too long"})
                return
            try:
                code = str(json.loads(body).get("code", ""))
            except (ValueError, AttributeError):
                self._reply(400, {"error": "expected JSON with a code field"})
                return
            if not code.strip():
                self._reply(400, {"error": "no code given"})
                return
            if not SLOTS.acquire(timeout=10):
                self._reply(503, {"error": "busy"})
                return
            try:
                self._reply(200, {"output": docker.run(code)})
            finally:
                SLOTS.release()

    return Handler


def main() -> None:
    token = os.getenv("GATE_SANDBOX_TOKEN", "").strip()
    if len(token) < 24:
        raise SystemExit("GATE_SANDBOX_TOKEN must be set (24+ characters), shared with the API.")
    docker = Docker()
    print("sandbox service: " + docker.ready()[1], flush=True)
    ThreadingHTTPServer(
        ("0.0.0.0", int(os.getenv("SANDBOX_PORT", "8090"))), make_handler(docker, token)
    ).serve_forever()


if __name__ == "__main__":
    main()

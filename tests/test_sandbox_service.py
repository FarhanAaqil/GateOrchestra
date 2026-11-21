"""The sandbox service (sandbox_service/server.py) and the API's client for it."""

from __future__ import annotations

import http.client
import json
import struct
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from assistant import sandbox
from sandbox_service import server

REAL_URLOPEN = urllib.request.urlopen  # captured before the no-network fixture replaces it
TOKEN = "sandbox-token-" + "s" * 20


def frame(kind: int, text: str) -> bytes:
    data = text.encode()
    return bytes([kind, 0, 0, 0]) + struct.pack(">I", len(data)) + data


def test_the_only_container_it_creates_is_locked_down() -> None:
    spec = server.container_spec("print(1)")
    host = spec["HostConfig"]
    assert spec["Cmd"] == ["python", "-I", "-c", "print(1)"]
    assert spec["User"] == "65534:65534" and spec["NetworkDisabled"] is True
    assert host["NetworkMode"] == "none" and host["ReadonlyRootfs"] is True
    assert host["CapDrop"] == ["ALL"] and host["SecurityOpt"] == ["no-new-privileges"]
    assert host["Memory"] == host["MemorySwap"] == 256 * 1024 * 1024
    assert host["PidsLimit"] == 64 and host["NanoCpus"] == 500_000_000
    assert "Binds" not in host and "Privileged" not in host  # nothing from the host is mounted


def test_logs_are_split_and_output_is_capped() -> None:
    out, err = server.demux(frame(1, "hello\n") + frame(2, "oops\n") + frame(1, "world\n"))
    assert out == "hello\nworld\n" and err == "oops\n"
    assert server.format_output(0, out, err) == "exit code 0\nhello\nworld\n\n[stderr]\noops"
    assert server.format_output(1, "", "") == "exit code 1\n(no output)"
    assert "truncated" in server.format_output(0, "x" * 5000, "")


class FakeDocker(server.Docker):
    def __init__(self, wait_times_out: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self.wait_times_out = wait_times_out

    def call(self, method, url, body=None, timeout=30):
        self.calls.append((method, url.split("?")[0]))
        if url.startswith("/containers/create"):
            assert body == server.container_spec("print(6 * 7)")
            return 201, json.dumps({"Id": "abc"}).encode()
        if url.endswith("/wait"):
            if self.wait_times_out:
                raise TimeoutError
            return 200, b'{"StatusCode": 0}'
        if "/logs" in url:
            return 200, frame(1, "42\n")
        return 204, b""


def test_a_run_creates_starts_reads_and_always_removes_the_container() -> None:
    docker = FakeDocker()
    assert docker.run("print(6 * 7)") == "exit code 0\n42"
    assert docker.calls[-1] == ("DELETE", "/containers/abc")
    slow = FakeDocker(wait_times_out=True)
    assert "longer than" in slow.run("print(6 * 7)")
    assert ("POST", "/containers/abc/kill") in slow.calls and slow.calls[-1] == (
        "DELETE",
        "/containers/abc",
    )


class StubDocker:
    def ready(self):
        return True, "ready"

    def run(self, code):
        return f"exit code 0\nran {len(code)} characters"


@pytest.fixture()
def running_service():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(StubDocker(), TOKEN))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", httpd.server_address[1]
    httpd.shutdown()


def post(port: int, body: bytes, token: str | None) -> tuple[int, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    conn.request("POST", "/run", body=body, headers=headers)
    response = conn.getresponse()
    return response.status, json.loads(response.read() or b"{}")


def test_the_service_needs_the_token_and_sane_input(running_service) -> None:
    _, port = running_service
    assert post(port, b'{"code": "print(1)"}', None)[0] == 401
    assert post(port, b'{"code": "print(1)"}', "wrong")[0] == 401
    assert post(port, b'{"code": "   "}', TOKEN)[0] == 400
    assert post(port, b"not json", TOKEN)[0] == 400
    assert post(port, json.dumps({"code": "x" * 70_000}).encode(), TOKEN)[0] == 413
    status, body = post(port, b'{"code": "print(1)"}', TOKEN)
    assert status == 200 and body["output"] == "exit code 0\nran 8 characters"


def test_the_api_uses_the_service_when_configured(
    running_service, monkeypatch: pytest.MonkeyPatch
) -> None:
    url, _ = running_service
    monkeypatch.setattr(urllib.request, "urlopen", REAL_URLOPEN)
    monkeypatch.setenv("ASSISTANT_SANDBOX_URL", url)
    monkeypatch.setenv("GATE_SANDBOX_TOKEN", TOKEN)
    monkeypatch.setitem(sandbox._STATUS, "checked", 0.0)
    assert sandbox.sandbox_status() == (True, "ready")
    assert sandbox.run_python("print(2)") == "exit code 0\nran 8 characters"
    monkeypatch.setenv("GATE_SANDBOX_TOKEN", "wrong")
    assert "answered 401" in sandbox.run_python("print(2)")
    monkeypatch.setenv("ASSISTANT_SANDBOX_URL", "http://127.0.0.1:9")  # nothing listens there
    monkeypatch.setitem(sandbox._STATUS, "checked", 0.0)
    assert sandbox.sandbox_status()[0] is False
    assert "unavailable" in sandbox.run_python("print(2)")


def test_the_service_refuses_to_start_without_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATE_SANDBOX_TOKEN", "short")
    with pytest.raises(SystemExit, match="GATE_SANDBOX_TOKEN"):
        server.main()

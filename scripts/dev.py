"""
scripts/dev.py
==============
Start the API and the console together with one command:

    python scripts/dev.py            # API on the first free port from 8010, console on 5173
    python scripts/dev.py --dry-run  # print what would run

Why it exists:
  * Another local app often holds port 8000. The API gets the first free port
    from 8010 instead, and the console is pointed at it (VITE_API_BASE_URL),
    so the two can never disagree.
  * npm fails on synced folders such as Google Drive (EBADF). When the repo is
    in one, the console is mirrored to a local folder (~/.gateorchestra-build)
    and installed and served from there. Source edits still happen in the repo;
    restart this script to pick up new files.

Ctrl+C stops both servers.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend-react"
LOCAL_MIRROR = Path.home() / ".gateorchestra-build" / "frontend"
SYNCED_FOLDER_MARKERS = ("my drive", "google drive", "onedrive", "dropbox", "icloud")
MIRROR_IGNORE = shutil.ignore_patterns("node_modules", "dist", ".vite")


def free_port(start: int, host: str = "127.0.0.1", tries: int = 50) -> int:
    """First port at or above ``start`` that nothing is listening on."""
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex((host, port)) != 0:
                return port
    raise RuntimeError(f"No free port in {start}-{start + tries - 1}")


def in_synced_folder(path: Path) -> bool:
    return any(marker in str(path).lower() for marker in SYNCED_FOLDER_MARKERS)


def frontend_dir(use_mirror: bool) -> Path:
    """Where to run npm: the repo itself, or a fresh local mirror of it."""
    if not use_mirror:
        return FRONTEND
    LOCAL_MIRROR.mkdir(parents=True, exist_ok=True)
    for item in LOCAL_MIRROR.iterdir():  # replace sources, keep node_modules
        if item.name not in ("node_modules", "dist"):
            shutil.rmtree(item) if item.is_dir() else item.unlink()
    shutil.copytree(FRONTEND, LOCAL_MIRROR, ignore=MIRROR_IGNORE, dirs_exist_ok=True)
    return LOCAL_MIRROR


def npm() -> str:
    found = shutil.which("npm")
    if found is None:
        raise SystemExit("npm was not found. Install Node.js 22+ from https://nodejs.org.")
    return found


def ensure_installed(directory: Path) -> None:
    """Run ``npm ci`` only when package-lock.json changed since the last install."""
    lock_hash = hashlib.sha256((directory / "package-lock.json").read_bytes()).hexdigest()
    stamp = directory / "node_modules" / ".gateorchestra-lock.sha256"
    if stamp.exists() and stamp.read_text(encoding="utf-8").strip() == lock_hash:
        return
    print(f"[dev] installing console dependencies in {directory}", flush=True)
    subprocess.run([npm(), "ci"], cwd=directory, check=True)
    stamp.write_text(lock_hash, encoding="utf-8")


def start(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.Popen:
    """Start a server in its own process group so it can be stopped with its children."""
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        return subprocess.Popen(cmd, cwd=cwd, env=env, creationflags=flags)
    return subprocess.Popen(cmd, cwd=cwd, env=env, start_new_session=True)


def stop(process: subprocess.Popen) -> None:
    """Stop a server and everything it spawned (npm leaves vite running otherwise)."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True, check=False
        )
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GateOrchestra API and console")
    parser.add_argument("--api-port", type=int, default=None, help="default: first free from 8010")
    parser.add_argument("--ui-port", type=int, default=5173)
    parser.add_argument("--no-mirror", action="store_true", help="run npm inside the repo")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api_port = args.api_port or free_port(8010)
    use_mirror = not args.no_mirror and in_synced_folder(FRONTEND)
    api_url = f"http://127.0.0.1:{api_port}"
    api_cmd = [sys.executable, "-m", "uvicorn", "api.main:app", "--port", str(api_port)]
    ui_cmd = [npm(), "run", "dev", "--", "--port", str(args.ui_port), "--strictPort"]

    print(f"[dev] API      {api_url}  ({' '.join(api_cmd[1:])})")
    print(f"[dev] console  http://localhost:{args.ui_port}  (VITE_API_BASE_URL={api_url})")
    if use_mirror:
        print(f"[dev] the repo is in a synced folder; serving the console from {LOCAL_MIRROR}")
    if args.dry_run:
        return

    ui_dir = frontend_dir(use_mirror)
    ensure_installed(ui_dir)
    env = {**os.environ, "VITE_API_BASE_URL": api_url}
    processes = [start(api_cmd, ROOT), start(ui_cmd, ui_dir, env)]
    try:
        while all(p.poll() is None for p in processes):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for process in processes:
            stop(process)


if __name__ == "__main__":
    main()

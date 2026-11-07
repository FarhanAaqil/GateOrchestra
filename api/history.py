"""
api/history.py
==============
Recent /run results, kept on disk so they survive an API restart.

Runs are appended to a JSON-lines file; the newest ``limit`` are held in
memory. The file is compacted when it grows well past ``limit`` lines, so it
never grows without bound. All access is serialised by a lock because FastAPI
runs sync endpoints on a thread pool.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

COMPACT_FACTOR = 4  # rewrite the file once it holds this many times ``limit`` lines


class RunHistory:
    """Newest-first history of API runs, persisted to ``path``."""

    def __init__(self, path: Path, limit: int = 50) -> None:
        self.path = path
        self.limit = limit
        self._lock = threading.Lock()
        self._runs: deque[dict[str, Any]] = deque(maxlen=limit)
        self._lines_on_disk = 0
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as err:
            logger.warning(f"Could not read run history {self.path}: {err}")
            return
        self._lines_on_disk = len(lines)
        for line in reversed(lines):  # newest first, until `limit` valid runs
            if len(self._runs) == self.limit:
                break
            try:
                self._runs.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn line from a crash; skip it

    def add(self, run: dict[str, Any]) -> None:
        with self._lock:
            self._runs.appendleft(run)
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                if self._lines_on_disk >= COMPACT_FACTOR * self.limit:
                    self._rewrite()
                else:
                    with self.path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(run) + "\n")
                    self._lines_on_disk += 1
            except OSError as err:
                logger.warning(f"Could not persist run history: {err}")

    def _rewrite(self) -> None:
        tmp = self.path.with_name(f".{self.path.name}.tmp")
        tmp.write_text(
            "".join(json.dumps(run) + "\n" for run in reversed(self._runs)), encoding="utf-8"
        )
        tmp.replace(self.path)
        self._lines_on_disk = len(self._runs)

    def clear(self) -> None:
        with self._lock:
            self._runs.clear()
            self._lines_on_disk = 0
            try:
                self.path.unlink(missing_ok=True)
            except OSError as err:
                logger.warning(f"Could not delete run history: {err}")

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._runs)

    def __len__(self) -> int:
        return len(self._runs)

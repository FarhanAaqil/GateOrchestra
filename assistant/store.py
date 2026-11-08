"""
assistant/store.py
==================
SQLite storage for the assistant: conversations, messages, feedback, usage
events (for metrics and gate selection), the answer cache and uploaded
documents. Standard library only; one connection per call, WAL mode, so the
API's worker threads can read and write concurrently.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL, content TEXT NOT NULL, meta TEXT NOT NULL DEFAULT '{}', created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);
CREATE TABLE IF NOT EXISTS feedback (
    message_id TEXT PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
    rating INTEGER NOT NULL, gate TEXT, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, kind TEXT NOT NULL, data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_kind_ts ON events(kind, ts);
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY, question TEXT NOT NULL, answer TEXT NOT NULL, meta TEXT NOT NULL,
    created_at REAL NOT NULL, hits INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, chars INTEGER NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL, text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
"""


def new_id() -> str:
    return uuid.uuid4().hex[:16]


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        with self._connect() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=15, check_same_thread=False)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA journal_mode = WAL")
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def _write(self, sql: str, params: tuple = ()) -> None:
        with self._write_lock, self._connect() as db:
            db.execute(sql, params)

    # Conversations and messages

    def create_conversation(self, title: str) -> str:
        cid, now = new_id(), time.time()
        self._write(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (cid, title[:120] or "New chat", now, now),
        )
        return cid

    def conversation_exists(self, cid: str) -> bool:
        with self._connect() as db:
            return (
                db.execute("SELECT 1 FROM conversations WHERE id = ?", (cid,)).fetchone()
                is not None
            )

    def list_conversations(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT id, title, created_at, updated_at FROM conversations "
                "ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def rename_conversation(self, cid: str, title: str) -> None:
        self._write("UPDATE conversations SET title = ? WHERE id = ?", (title[:120], cid))

    def delete_conversation(self, cid: str) -> None:
        self._write("DELETE FROM conversations WHERE id = ?", (cid,))

    def add_message(
        self, cid: str, role: str, content: str, meta: dict[str, Any] | None = None
    ) -> str:
        mid, now = new_id(), time.time()
        with self._write_lock, self._connect() as db:
            db.execute(
                "INSERT INTO messages (id, conversation_id, role, content, meta, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (mid, cid, role, content, json.dumps(meta or {}), now),
            )
            db.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, cid))
        return mid

    def messages(self, cid: str, limit: int | None = None) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT m.id, m.role, m.content, m.meta, m.created_at, f.rating "
                "FROM messages m LEFT JOIN feedback f ON f.message_id = m.id "
                "WHERE m.conversation_id = ? ORDER BY m.created_at",
                (cid,),
            ).fetchall()
        out = [{**dict(r), "meta": json.loads(r["meta"])} for r in rows]
        return out[-limit:] if limit else out

    def delete_message(self, mid: str) -> None:
        self._write("DELETE FROM messages WHERE id = ?", (mid,))

    def message(self, mid: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM messages WHERE id = ?", (mid,)).fetchone()
        return {**dict(row), "meta": json.loads(row["meta"])} if row else None

    # Feedback

    def set_feedback(self, mid: str, rating: int, gate: str | None) -> None:
        self._write(
            "INSERT INTO feedback (message_id, rating, gate, created_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(message_id) DO UPDATE SET rating = excluded.rating, created_at = excluded.created_at",
            (mid, rating, gate, time.time()),
        )

    def feedback_by_gate(self) -> dict[str, dict[str, int]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT gate, SUM(rating > 0) AS up, SUM(rating < 0) AS down FROM feedback "
                "WHERE gate IS NOT NULL GROUP BY gate"
            ).fetchall()
        return {r["gate"]: {"up": int(r["up"] or 0), "down": int(r["down"] or 0)} for r in rows}

    # Events (metrics, gate costs)

    def log_event(self, kind: str, data: dict[str, Any]) -> None:
        self._write(
            "INSERT INTO events (ts, kind, data) VALUES (?, ?, ?)",
            (time.time(), kind, json.dumps(data)),
        )

    def events(self, kind: str, since: float = 0.0, limit: int = 5000) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT ts, data FROM events WHERE kind = ? AND ts >= ? ORDER BY ts DESC LIMIT ?",
                (kind, since, limit),
            ).fetchall()
        return [{"ts": r["ts"], **json.loads(r["data"])} for r in rows]

    # Cache

    def cache_get(self, key: str, max_age: float) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM cache WHERE key = ?", (key,)).fetchone()
        if row is None or time.time() - row["created_at"] > max_age:
            return None
        self._write("UPDATE cache SET hits = hits + 1 WHERE key = ?", (key,))
        return {
            "answer": row["answer"],
            "question": row["question"],
            "meta": json.loads(row["meta"]),
        }

    def cache_put(self, key: str, question: str, answer: str, meta: dict[str, Any]) -> None:
        self._write(
            "INSERT OR REPLACE INTO cache (key, question, answer, meta, created_at, hits) VALUES (?, ?, ?, ?, ?, 0)",
            (key, question, answer, json.dumps(meta), time.time()),
        )

    def cache_entries(self, max_age: float, limit: int = 2000) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT key, question FROM cache WHERE created_at >= ? ORDER BY created_at DESC LIMIT ?",
                (time.time() - max_age, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def cache_clear(self) -> int:
        with self._write_lock, self._connect() as db:
            return db.execute("DELETE FROM cache").rowcount

    # Documents

    def add_document(self, name: str, chunks: list[str]) -> str:
        did, now = new_id(), time.time()
        with self._write_lock, self._connect() as db:
            db.execute(
                "INSERT INTO documents (id, name, chars, created_at) VALUES (?, ?, ?, ?)",
                (did, name[:200], sum(len(c) for c in chunks), now),
            )
            db.executemany(
                "INSERT INTO chunks (document_id, idx, text) VALUES (?, ?, ?)",
                [(did, i, c) for i, c in enumerate(chunks)],
            )
        return did

    def documents(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT d.id, d.name, d.chars, d.created_at, COUNT(c.id) AS chunks FROM documents d "
                "LEFT JOIN chunks c ON c.document_id = d.id GROUP BY d.id ORDER BY d.created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def all_chunks(self, document_ids: list[str] | None = None) -> list[dict[str, Any]]:
        with self._connect() as db:
            if document_ids:
                marks = ",".join("?" * len(document_ids))
                rows = db.execute(
                    f"SELECT c.id, c.document_id, c.idx, c.text, d.name FROM chunks c "
                    f"JOIN documents d ON d.id = c.document_id WHERE c.document_id IN ({marks})",
                    tuple(document_ids),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT c.id, c.document_id, c.idx, c.text, d.name FROM chunks c "
                    "JOIN documents d ON d.id = c.document_id"
                ).fetchall()
        return [dict(r) for r in rows]

    def delete_document(self, did: str) -> None:
        self._write("DELETE FROM documents WHERE id = ?", (did,))

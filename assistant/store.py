"""
assistant/store.py
==================
SQLite storage for the assistant: conversations, messages, feedback, usage
events (for metrics and gate selection), the answer cache and uploaded
documents. Standard library only; one connection per call, WAL mode, so the
API's worker threads can read and write concurrently.

Every row carries a ``user_id``. ``store.scoped(user_id)`` returns a view
whose every query filters on it, so one user can never read, change or
delete another's rows: for that view they do not exist (the API turns that
into a 404). Single-tenant mode uses ``scoped(None)``, which matches the rows
written without a user, i.e. all of them, exactly as before. The unscoped
store (the service's own) sees everything; only operator views and the gate
selector's aggregate numbers use it.
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

SCOPED_TABLES = ("conversations", "messages", "feedback", "events", "cache", "documents", "chunks")
_ALL: Any = object()  # an unscoped store: no user filter


def new_id() -> str:
    return uuid.uuid4().hex[:16]


class Store:
    def __init__(self, path: Path, *, _parent: Store | None = None, _user: Any = _ALL) -> None:
        self.path = path
        self._user = _user
        if _parent is not None:
            self._write_lock = _parent._write_lock
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        with self._connect() as db:
            db.executescript(SCHEMA)
            self._migrate(db)

    @staticmethod
    def _migrate(db: sqlite3.Connection) -> None:
        """Databases from before accounts get a user_id column (NULL: single-tenant rows)."""
        for table in SCOPED_TABLES:
            columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
            if "user_id" not in columns:
                db.execute(f"ALTER TABLE {table} ADD COLUMN user_id TEXT")
        db.executescript(
            "CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id, updated_at);"
            "CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id);"
            "CREATE INDEX IF NOT EXISTS idx_chunks_user ON chunks(user_id);"
            "CREATE INDEX IF NOT EXISTS idx_events_user ON events(user_id, ts);"
        )

    def scoped(self, user_id: str | None) -> Store:
        """A view of one user's rows (None: the single-tenant rows)."""
        return Store(self.path, _parent=self, _user=user_id)

    @property
    def user_id(self) -> str | None:
        """The owner written on new rows (None when unscoped or single-tenant)."""
        return None if self._user is _ALL else self._user

    def _who(self, alias: str = "") -> tuple[str, tuple]:
        """SQL condition limiting a query to this view's rows."""
        if self._user is _ALL:
            return "1 = 1", ()
        return f"{alias + '.' if alias else ''}user_id IS ?", (self._user,)

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

    @contextmanager
    def transaction(self):
        """One serialised write transaction (for modules that share this database)."""
        with self._write_lock, self._connect() as db:
            yield db

    @contextmanager
    def read(self):
        with self._connect() as db:
            yield db

    def _write(self, sql: str, params: tuple = ()) -> int:
        with self._write_lock, self._connect() as db:
            return db.execute(sql, params).rowcount

    # Conversations and messages

    def create_conversation(self, title: str) -> str:
        cid, now = new_id(), time.time()
        self._write(
            "INSERT INTO conversations (id, title, created_at, updated_at, user_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (cid, title[:120] or "New chat", now, now, self.user_id),
        )
        return cid

    def conversation(self, cid: str) -> dict[str, Any] | None:
        who, args = self._who()
        with self._connect() as db:
            row = db.execute(
                "SELECT id, title, created_at, updated_at FROM conversations "
                f"WHERE id = ? AND {who}",
                (cid, *args),
            ).fetchone()
        return dict(row) if row else None

    def conversation_exists(self, cid: str) -> bool:
        return self.conversation(cid) is not None

    def list_conversations(self, limit: int = 50) -> list[dict[str, Any]]:
        who, args = self._who()
        with self._connect() as db:
            rows = db.execute(
                f"SELECT id, title, created_at, updated_at FROM conversations WHERE {who} "
                "ORDER BY updated_at DESC LIMIT ?",
                (*args, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def rename_conversation(self, cid: str, title: str) -> bool:
        who, args = self._who()
        changed = self._write(
            f"UPDATE conversations SET title = ? WHERE id = ? AND {who}", (title[:120], cid, *args)
        )
        return changed > 0

    def delete_conversation(self, cid: str) -> bool:
        who, args = self._who()
        return self._write(f"DELETE FROM conversations WHERE id = ? AND {who}", (cid, *args)) > 0

    def add_message(
        self, cid: str, role: str, content: str, meta: dict[str, Any] | None = None
    ) -> str:
        if not self.conversation_exists(cid):
            raise LookupError("No such conversation.")
        mid, now = new_id(), time.time()
        with self._write_lock, self._connect() as db:
            db.execute(
                "INSERT INTO messages "
                "(id, conversation_id, role, content, meta, created_at, user_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (mid, cid, role, content, json.dumps(meta or {}), now, self.user_id),
            )
            db.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, cid))
        return mid

    def messages(self, cid: str, limit: int | None = None) -> list[dict[str, Any]]:
        who, args = self._who("m")
        with self._connect() as db:
            rows = db.execute(
                "SELECT m.id, m.role, m.content, m.meta, m.created_at, f.rating "
                "FROM messages m LEFT JOIN feedback f ON f.message_id = m.id "
                f"WHERE m.conversation_id = ? AND {who} ORDER BY m.created_at",
                (cid, *args),
            ).fetchall()
        out = [{**dict(r), "meta": json.loads(r["meta"])} for r in rows]
        return out[-limit:] if limit else out

    def delete_message(self, mid: str) -> None:
        who, args = self._who()
        self._write(f"DELETE FROM messages WHERE id = ? AND {who}", (mid, *args))

    def message(self, mid: str) -> dict[str, Any] | None:
        who, args = self._who()
        with self._connect() as db:
            row = db.execute(
                f"SELECT * FROM messages WHERE id = ? AND {who}", (mid, *args)
            ).fetchone()
        return {**dict(row), "meta": json.loads(row["meta"])} if row else None

    # Feedback

    def set_feedback(self, mid: str, rating: int, gate: str | None) -> None:
        if self.message(mid) is None:
            raise LookupError("No such answer.")
        self._write(
            "INSERT INTO feedback (message_id, rating, gate, created_at, user_id) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT(message_id) DO UPDATE SET "
            "rating = excluded.rating, created_at = excluded.created_at",
            (mid, rating, gate, time.time(), self.user_id),
        )

    def clear_feedback(self, mid: str) -> None:
        who, args = self._who()
        self._write(f"DELETE FROM feedback WHERE message_id = ? AND {who}", (mid, *args))

    def rated_turns(self, limit: int = 5000) -> list[dict[str, Any]]:
        """Each rated answer's rating with the route it took and every gate's verdict."""
        who, args = self._who("f")
        with self._connect() as db:
            rows = db.execute(
                "SELECT f.rating, m.meta FROM feedback f JOIN messages m ON m.id = f.message_id "
                f"WHERE {who} ORDER BY f.created_at DESC LIMIT ?",
                (*args, limit),
            ).fetchall()
        out = []
        for row in rows:
            meta = json.loads(row["meta"])
            if meta.get("decision") and meta.get("verdicts") and row["rating"]:
                out.append(
                    {
                        "rating": row["rating"],
                        "decision": meta["decision"],
                        "verdicts": meta["verdicts"],
                    }
                )
        return out

    def feedback_by_gate(self) -> dict[str, dict[str, int]]:
        who, args = self._who()
        with self._connect() as db:
            rows = db.execute(
                "SELECT gate, SUM(rating > 0) AS up, SUM(rating < 0) AS down FROM feedback "
                f"WHERE gate IS NOT NULL AND {who} GROUP BY gate",
                args,
            ).fetchall()
        return {r["gate"]: {"up": int(r["up"] or 0), "down": int(r["down"] or 0)} for r in rows}

    # Events (metrics, gate costs)

    def log_event(self, kind: str, data: dict[str, Any]) -> None:
        self._write(
            "INSERT INTO events (ts, kind, data, user_id) VALUES (?, ?, ?, ?)",
            (time.time(), kind, json.dumps(data), self.user_id),
        )

    def events(self, kind: str, since: float = 0.0, limit: int = 5000) -> list[dict[str, Any]]:
        who, args = self._who()
        with self._connect() as db:
            rows = db.execute(
                f"SELECT ts, data FROM events WHERE kind = ? AND ts >= ? AND {who} "
                "ORDER BY ts DESC LIMIT ?",
                (kind, since, *args, limit),
            ).fetchall()
        return [{"ts": r["ts"], **json.loads(r["data"])} for r in rows]

    # Cache: one per user, so nobody is ever served an answer to someone else's question

    def _cache_key(self, key: str) -> str:
        return key if self.user_id is None else f"user:{self.user_id}:{key}"

    def cache_get(self, key: str, max_age: float) -> dict[str, Any] | None:
        who, args = self._who()
        stored = self._cache_key(key)
        with self._connect() as db:
            row = db.execute(
                f"SELECT * FROM cache WHERE key = ? AND {who}", (stored, *args)
            ).fetchone()
        if row is None or time.time() - row["created_at"] > max_age:
            return None
        self._write("UPDATE cache SET hits = hits + 1 WHERE key = ?", (stored,))
        return {
            "answer": row["answer"],
            "question": row["question"],
            "meta": json.loads(row["meta"]),
        }

    def cache_put(self, key: str, question: str, answer: str, meta: dict[str, Any]) -> None:
        self._write(
            "INSERT OR REPLACE INTO cache "
            "(key, question, answer, meta, created_at, hits, user_id) VALUES (?, ?, ?, ?, ?, 0, ?)",
            (self._cache_key(key), question, answer, json.dumps(meta), time.time(), self.user_id),
        )

    def cache_entries(self, max_age: float, limit: int = 2000) -> list[dict[str, Any]]:
        who, args = self._who()
        with self._connect() as db:
            rows = db.execute(
                f"SELECT key, question FROM cache WHERE created_at >= ? AND {who} "
                "ORDER BY created_at DESC LIMIT ?",
                (time.time() - max_age, *args, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def cache_clear(self) -> int:
        who, args = self._who()
        return self._write(f"DELETE FROM cache WHERE {who}", args)

    # Documents

    def add_document(self, name: str, chunks: list[str]) -> str:
        did, now = new_id(), time.time()
        with self._write_lock, self._connect() as db:
            db.execute(
                "INSERT INTO documents (id, name, chars, created_at, user_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (did, name[:200], sum(len(c) for c in chunks), now, self.user_id),
            )
            db.executemany(
                "INSERT INTO chunks (document_id, idx, text, user_id) VALUES (?, ?, ?, ?)",
                [(did, i, c, self.user_id) for i, c in enumerate(chunks)],
            )
        return did

    def documents(self) -> list[dict[str, Any]]:
        who, args = self._who("d")
        with self._connect() as db:
            rows = db.execute(
                "SELECT d.id, d.name, d.chars, d.created_at, COUNT(c.id) AS chunks "
                f"FROM documents d LEFT JOIN chunks c ON c.document_id = d.id WHERE {who} "
                "GROUP BY d.id ORDER BY d.created_at DESC",
                args,
            ).fetchall()
        return [dict(r) for r in rows]

    def all_chunks(self, document_ids: list[str] | None = None) -> list[dict[str, Any]]:
        """Chunks of this view's own documents (optionally only ``document_ids``)."""
        chunk_who, chunk_args = self._who("c")
        doc_who, doc_args = self._who("d")
        sql = (
            "SELECT c.id, c.document_id, c.idx, c.text, d.name FROM chunks c "
            f"JOIN documents d ON d.id = c.document_id WHERE {chunk_who} AND {doc_who}"
        )
        params: tuple = (*chunk_args, *doc_args)
        if document_ids:
            sql += f" AND c.document_id IN ({','.join('?' * len(document_ids))})"
            params += tuple(document_ids)
        with self._connect() as db:
            rows = db.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def delete_document(self, did: str) -> bool:
        who, args = self._who()
        return self._write(f"DELETE FROM documents WHERE id = ? AND {who}", (did, *args)) > 0

    # A user's data as a whole (export and account deletion)

    def export(self) -> dict[str, Any]:
        """This view's conversations with their messages, and its documents' names."""
        conversations = [
            {**c, "messages": self.messages(c["id"])} for c in self.list_conversations(1_000_000)
        ]
        return {"conversations": conversations, "documents": self.documents()}

    def delete_everything(self) -> None:
        """Remove every row of this view's user (refused on unscoped and single-tenant views)."""
        uid = self.user_id
        if uid is None:
            raise ValueError("Only one user's own view can be wiped.")
        with self._write_lock, self._connect() as db:
            for table in ("feedback", "messages", "conversations", "chunks", "documents", "cache"):
                db.execute(f"DELETE FROM {table} WHERE user_id = ?", (uid,))
            # Usage events hold no content (gate, tokens, timings). They stay in the
            # operator's totals but no longer point at the person.
            db.execute("UPDATE events SET user_id = 'deleted' WHERE user_id = ?", (uid,))

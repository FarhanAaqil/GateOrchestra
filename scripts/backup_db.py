"""
Back up the assistant's SQLite database, and prove the backup restores.

The database holds everything a hosted server keeps: accounts, sessions,
conversations, uploaded documents (as their extracted text chunks), the answer
cache, usage counts and people's encrypted API keys. Nothing else needs a
backup except your .env, which holds secrets and belongs in a password manager.

    python scripts/backup_db.py --db logs/assistant.db --out backups --keep 14
    python scripts/backup_db.py --verify backups/assistant-20261120-031500.db.gz

A backup uses SQLite's online backup API, so it is consistent even while the
API is writing. Every backup is then restored to a temporary file and checked
(``PRAGMA integrity_check``, every table readable, row counts printed) before
older backups beyond ``--keep`` days are removed. A failed check exits with
status 1 and deletes nothing.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

PREFIX = "assistant-"
SUFFIX = ".db.gz"


def backup(db_path: Path, out_dir: Path, now: float | None = None) -> Path:
    """A gzipped, consistent copy of ``db_path`` in ``out_dir``."""
    if not db_path.exists():
        raise FileNotFoundError(f"No database at {db_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(now or time.time()))
    target = out_dir / f"{PREFIX}{stamp}{SUFFIX}"
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "copy.db"
        source = sqlite3.connect(db_path)
        dest = sqlite3.connect(copy)
        try:
            source.backup(dest)
        finally:
            dest.close()
            source.close()
        with copy.open("rb") as raw, gzip.open(target, "wb", compresslevel=6) as packed:
            shutil.copyfileobj(raw, packed)
    return target


def verify(backup_path: Path) -> dict[str, int]:
    """Restore ``backup_path`` to a temporary file and check it. Returns row counts."""
    with tempfile.TemporaryDirectory() as tmp:
        restored = Path(tmp) / "restored.db"
        with gzip.open(backup_path, "rb") as packed, restored.open("wb") as raw:
            shutil.copyfileobj(packed, raw)
        db = sqlite3.connect(restored)
        try:
            check = db.execute("PRAGMA integrity_check").fetchone()[0]
            if check != "ok":
                raise RuntimeError(f"integrity check failed: {check}")
            tables = [
                r[0]
                for r in db.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            ]
            if not tables:
                raise RuntimeError("the backup holds no tables")
            return {t: db.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in tables}
        finally:
            db.close()


def prune(out_dir: Path, keep_days: int, now: float | None = None) -> list[Path]:
    """Delete backups older than ``keep_days`` (never the newest one)."""
    cutoff = (now or time.time()) - keep_days * 86400
    backups = sorted(out_dir.glob(f"{PREFIX}*{SUFFIX}"))
    removed = []
    for path in backups[:-1]:
        if path.stat().st_mtime < cutoff:
            path.unlink()
            removed.append(path)
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", type=Path, default=Path("logs/assistant.db"))
    parser.add_argument("--out", type=Path, default=Path("backups"))
    parser.add_argument("--keep", type=int, default=14, help="days of backups to keep")
    parser.add_argument("--verify", type=Path, help="only check an existing backup")
    args = parser.parse_args(argv)
    try:
        if args.verify:
            counts = verify(args.verify)
            print(f"OK {args.verify}: " + ", ".join(f"{t}={n}" for t, n in counts.items()))
            return 0
        made = backup(args.db, args.out)
        counts = verify(made)
        print(f"OK {made}: " + ", ".join(f"{t}={n}" for t, n in counts.items()))
        for path in prune(args.out, args.keep):
            print(f"removed {path}")
        return 0
    except (OSError, RuntimeError, sqlite3.DatabaseError, gzip.BadGzipFile) as err:
        print(f"BACKUP FAILED: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

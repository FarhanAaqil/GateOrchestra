"""
dataset/repository.py
=====================
Repository Pattern — Storage Abstraction Layer for GateOrchestra Dataset.

Week 5 deliverable (Person 1 — Dataset Engineer).

Provides a clean interface between the rest of the codebase and however
the dataset is physically stored (local JSONL files today, Supabase / cloud
PostgreSQL tomorrow).  Other team members should program against
``TaskRepository`` — they should never import from the concrete adapters.

Architecture:
    TaskRepository          <- abstract interface (contract)
        JSONLTaskRepository   <- default filesystem backend (Week 5)
        SupabaseTaskRepository (stub)  <- future cloud adapter (Future)

Usage::

    from dataset.repository import JSONLTaskRepository

    repo = JSONLTaskRepository()          # uses DATASET_DIR from shared.config
    task = repo.get_by_id("hotpot_001")
    train = repo.list_by_split("train")
    hard  = repo.filter(depth=4)
    repo.save(new_task, split="train")    # persists to JSONL

Person 1 owns this file.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from shared.config import DATASET_DIR
from shared.schemas import Task

logger = logging.getLogger(__name__)

_VALID_SPLITS = ("train", "val", "test")

# ─────────────────────────────────────────────────────────────────────────────
# Abstract Interface
# ─────────────────────────────────────────────────────────────────────────────


class TaskRepository(ABC):
    """Abstract storage interface for Task objects.

    All concrete adapters must implement every method.  Program against
    *this* class -- not against any concrete subclass -- so that swapping
    backends (JSONL -> Supabase) requires zero changes to callers.
    """

    # -- Read ------------------------------------------------------------------

    @abstractmethod
    def get_by_id(self, task_id: str) -> Task | None:
        """Retrieve a single Task by its unique ID.

        Args:
            task_id: The unique task identifier, e.g. ``'hotpot_001'``.

        Returns:
            The matching ``Task`` object, or ``None`` if not found.
        """

    @abstractmethod
    def list_by_split(self, split: str) -> list[Task]:
        """Return all Tasks belonging to a given dataset split.

        Args:
            split: One of ``'train'``, ``'val'``, ``'test'``.

        Returns:
            List of ``Task`` objects for that split.

        Raises:
            ValueError: if *split* is not a recognised name.
        """

    @abstractmethod
    def filter(
        self,
        source: str | None = None,
        depth: int | None = None,
        parallel: int | None = None,
        split: str | None = None,
    ) -> list[Task]:
        """Return Tasks matching all supplied (non-None) criteria.

        Args:
            source:   Filter by ``Task.source_dataset``.
            depth:    Filter by ``Task.depth_score``.
            parallel: Filter by ``Task.parallel_score``.
            split:    Restrict search to this split only; searches all
                      splits when ``None``.

        Returns:
            List of matching ``Task`` objects (may be empty).
        """

    @abstractmethod
    def all(self) -> list[Task]:
        """Return every Task across all splits."""

    # -- Write -----------------------------------------------------------------

    @abstractmethod
    def save(self, task: Task, split: str) -> None:
        """Persist a Task to the specified split.

        If a task with the same ``task_id`` already exists in the split it
        will be **overwritten**.

        Args:
            task:  The ``Task`` object to persist.
            split: The target split: ``'train'``, ``'val'``, or ``'test'``.

        Raises:
            ValueError: if *split* is not recognised.
        """

    # -- Streaming -------------------------------------------------------------

    @abstractmethod
    def stream(self, split: str) -> Iterator[Task]:
        """Yield Tasks one at a time for memory-efficient iteration.

        Args:
            split: One of ``'train'``, ``'val'``, ``'test'``.

        Yields:
            ``Task`` objects for the split, lazily.
        """

    # -- Metadata --------------------------------------------------------------

    @abstractmethod
    def count(self, split: str | None = None) -> int:
        """Return the number of tasks (optionally restricted to one split).

        Args:
            split: When provided, count only tasks in that split.
                   When ``None``, count all tasks across every split.

        Returns:
            Integer task count.
        """

    @abstractmethod
    def split_names(self) -> list[str]:
        """Return the split names that have data on this backend."""


# ─────────────────────────────────────────────────────────────────────────────
# JSONL Filesystem Adapter (Default / Week 5)
# ─────────────────────────────────────────────────────────────────────────────


class JSONLTaskRepository(TaskRepository):
    """Filesystem-backed ``TaskRepository`` using JSONL split files.

    Reads from and writes to ``dataset/masbench_mini/{split}.jsonl``.
    This is the default local backend used throughout Weeks 1-8.

    The repository maintains an in-memory cache that is populated lazily
    on the first access to each split.  Call ``invalidate_cache()`` if you
    write to the JSONL files externally and need to force a re-read.

    Args:
        dataset_dir: Root directory containing ``{split}.jsonl`` files.
                     Defaults to ``shared.config.DATASET_DIR``.
    """

    def __init__(self, dataset_dir: Path | None = None) -> None:
        self._root = Path(dataset_dir) if dataset_dir is not None else DATASET_DIR
        # {split: list[Task]} -- populated lazily
        self._cache: dict[str, list[Task]] = {}

    # -- Internal helpers ------------------------------------------------------

    def _split_path(self, split: str) -> Path:
        """Resolve the JSONL file path for *split*."""
        flat = self._root / f"{split}.jsonl"
        if flat.exists():
            return flat
        # Legacy nested layout: dataset/masbench_mini/train/train.jsonl
        nested = self._root / split / f"{split}.jsonl"
        if nested.exists():
            return nested
        return flat  # Return flat path even if missing (for save())

    def _load_split(self, split: str) -> list[Task]:
        """Load tasks from disk for *split*, populating the cache."""
        if split in self._cache:
            return self._cache[split]

        path = self._split_path(split)
        tasks: list[Task] = []

        if not path.exists():
            logger.warning(
                "[Repository] Split file not found: %s -- returning empty list.", path
            )
            self._cache[split] = tasks
            return tasks

        with path.open(encoding="utf-8") as fh:
            for line_num, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    tasks.append(Task(**json.loads(raw)))
                except Exception as exc:
                    logger.error(
                        "[Repository] Parse error in %s:%d -- %s",
                        path.name, line_num, exc,
                    )

        self._cache[split] = tasks
        logger.debug(
            "[Repository] Loaded %d tasks from split='%s' (%s)", len(tasks), split, path
        )
        return tasks

    def _load_all(self) -> list[Task]:
        """Return a flat list of every task across all available splits."""
        tasks: list[Task] = []
        for split in self.split_names():
            tasks.extend(self._load_split(split))
        return tasks

    def invalidate_cache(self, split: str | None = None) -> None:
        """Drop cached data so the next read hits disk.

        Args:
            split: Drop only this split's cache; when ``None`` drops all.
        """
        if split is not None:
            self._cache.pop(split, None)
        else:
            self._cache.clear()
        logger.debug(
            "[Repository] Cache invalidated for split='%s'.", split or "ALL"
        )

    # -- TaskRepository interface -----------------------------------------------

    def get_by_id(self, task_id: str) -> Task | None:
        """O(n) linear scan across all splits.  Suitable for small datasets."""
        for task in self._load_all():
            if task.task_id == task_id:
                return task
        logger.debug("[Repository] task_id='%s' not found.", task_id)
        return None

    def list_by_split(self, split: str) -> list[Task]:
        if split not in _VALID_SPLITS:
            raise ValueError(
                f"split must be one of {_VALID_SPLITS}, got {split!r}"
            )
        return list(self._load_split(split))  # return a copy

    def filter(
        self,
        source: str | None = None,
        depth: int | None = None,
        parallel: int | None = None,
        split: str | None = None,
    ) -> list[Task]:
        pool: list[Task]
        if split is not None:
            if split not in _VALID_SPLITS:
                raise ValueError(
                    f"split must be one of {_VALID_SPLITS}, got {split!r}"
                )
            pool = self._load_split(split)
        else:
            pool = self._load_all()

        results: list[Task] = []
        for task in pool:
            if source is not None and task.source_dataset != source:
                continue
            if depth is not None and task.depth_score != depth:
                continue
            if parallel is not None and task.parallel_score != parallel:
                continue
            results.append(task)

        logger.debug(
            "[Repository] filter(source=%r, depth=%s, parallel=%s, split=%r) -> %d tasks",
            source, depth, parallel, split, len(results),
        )
        return results

    def all(self) -> list[Task]:
        return self._load_all()

    def save(self, task: Task, split: str) -> None:
        if split not in _VALID_SPLITS:
            raise ValueError(
                f"split must be one of {_VALID_SPLITS}, got {split!r}"
            )

        path = self._split_path(split)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing tasks, replace if task_id already exists
        existing = self._load_split(split)
        updated = [t for t in existing if t.task_id != task.task_id]
        replaced = len(existing) - len(updated)
        updated.append(task)

        with path.open("w", encoding="utf-8") as fh:
            for t in updated:
                fh.write(t.model_dump_json() + "\n")

        # Invalidate cache so next read is fresh
        self.invalidate_cache(split)

        action = "Updated" if replaced else "Saved"
        logger.info(
            "[Repository] %s task '%s' in split='%s' (%s)", action, task.task_id, split, path
        )

    def stream(self, split: str) -> Iterator[Task]:
        if split not in _VALID_SPLITS:
            raise ValueError(
                f"split must be one of {_VALID_SPLITS}, got {split!r}"
            )

        path = self._split_path(split)
        if not path.exists():
            logger.warning("[Repository] stream: Split file not found: %s", path)
            return

        with path.open(encoding="utf-8") as fh:
            for line_num, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    yield Task(**json.loads(raw))
                except Exception as exc:
                    logger.error(
                        "[Repository] stream parse error %s:%d -- %s",
                        path.name, line_num, exc,
                    )

    def count(self, split: str | None = None) -> int:
        if split is not None:
            if split not in _VALID_SPLITS:
                raise ValueError(
                    f"split must be one of {_VALID_SPLITS}, got {split!r}"
                )
            return len(self._load_split(split))
        return len(self._load_all())

    def split_names(self) -> list[str]:
        """Return only the splits that actually have data on disk."""
        available = []
        for split in _VALID_SPLITS:
            if self._split_path(split).exists():
                available.append(split)
        return available

    # -- Batch loading ---------------------------------------------------------

    def batch(self, split: str, batch_size: int = 16) -> Iterator[list[Task]]:
        """Yield tasks in fixed-size batches for batched inference.

        Args:
            split:      The dataset split to batch over.
            batch_size: Number of tasks per batch (default 16).

        Yields:
            Lists of up to *batch_size* ``Task`` objects.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        buf: list[Task] = []
        for task in self.stream(split):
            buf.append(task)
            if len(buf) == batch_size:
                yield buf
                buf = []
        if buf:
            yield buf

    def __repr__(self) -> str:
        splits = self.split_names()
        counts = {s: self.count(s) for s in splits}
        return f"JSONLTaskRepository(root={self._root}, splits={counts})"


# ─────────────────────────────────────────────────────────────────────────────
# Future: Supabase Adapter Stub
# ─────────────────────────────────────────────────────────────────────────────


class SupabaseTaskRepository(TaskRepository):
    """Supabase / PostgreSQL-backed repository adapter.

    **NOT YET IMPLEMENTED** -- this class is a forward-looking stub that
    documents the interface contract for the future cloud deployment phase.

    Once implemented it will sync tasks to a ``tasks`` table in Supabase
    (with ``task_id`` as primary key) and will enable live telemetry,
    multi-user query routing, and cross-team experiment tracking.

    To activate:
        1.  ``pip install supabase``
        2.  Set ``SUPABASE_URL`` and ``SUPABASE_KEY`` env vars.
        3.  Implement each abstract method against the Supabase client.

    Args:
        url:   Supabase project URL (overrides ``SUPABASE_URL`` env var).
        key:   Supabase anon/service key (overrides ``SUPABASE_KEY`` env var).
        table: Table name to read/write (default ``'tasks'``).
    """

    def __init__(
        self,
        url: str | None = None,
        key: str | None = None,
        table: str = "tasks",
    ) -> None:
        import os

        self._url = url or os.getenv("SUPABASE_URL", "")
        self._key = key or os.getenv("SUPABASE_KEY", "")
        self._table = table

        if not self._url or not self._key:
            raise EnvironmentError(
                "SupabaseTaskRepository requires SUPABASE_URL and SUPABASE_KEY "
                "to be set (as env vars or constructor args)."
            )
        raise NotImplementedError(
            "SupabaseTaskRepository is not yet implemented.  "
            "Use JSONLTaskRepository for local development."
        )

    # -- Stubs (satisfy ABC) ---------------------------------------------------

    def get_by_id(self, task_id: str) -> Task | None:  # type: ignore[return]
        raise NotImplementedError

    def list_by_split(self, split: str) -> list[Task]:
        raise NotImplementedError

    def filter(
        self,
        source: str | None = None,
        depth: int | None = None,
        parallel: int | None = None,
        split: str | None = None,
    ) -> list[Task]:
        raise NotImplementedError

    def all(self) -> list[Task]:
        raise NotImplementedError

    def save(self, task: Task, split: str) -> None:
        raise NotImplementedError

    def stream(self, split: str) -> Iterator[Task]:  # type: ignore[return]
        raise NotImplementedError

    def count(self, split: str | None = None) -> int:
        raise NotImplementedError

    def split_names(self) -> list[str]:
        raise NotImplementedError

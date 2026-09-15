"""
dataset/loader.py
==================
Public API for loading the GateOrchestra dataset.

Other team members should use ONLY this module — do not import
from dataset.generation or dataset.labeling directly.

Usage:
    from dataset import load_dataset, load_all_tasks

    train_tasks = load_dataset("train")     # list[Task]
    val_tasks   = load_dataset("val")       # list[Task]
    test_tasks  = load_dataset("test")      # list[Task]
    all_tasks   = load_all_tasks()          # list[Task] (all splits)
    df          = load_features_df()        # pandas DataFrame (if available)

Paths are resolved relative to repo root (shared/config.py DATASET_DIR).
The loader reads from dataset/masbench_mini/{split}.jsonl files.

Person 1 owns this file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterator

from shared.config import DATASET_DIR
from shared.schemas import Task

logger = logging.getLogger(__name__)

_VALID_SPLITS = {"train", "val", "test"}

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def load_dataset(
    split: str,
    *,
    filter_func: Callable[[Task], bool] | None = None,
) -> list[Task]:
    """Load tasks for a given split, with optional filtering.

    Args:
        split: One of 'train', 'val', 'test'
        filter_func: Optional predicate function taking a Task and returning bool

    Returns:
        List of Task objects for the requested split.

    Raises:
        ValueError: if split is not recognized
        FileNotFoundError: if the split file does not exist yet
    """
    if split not in _VALID_SPLITS:
        raise ValueError(f"split must be one of {_VALID_SPLITS}, got {split!r}")

    # Check both flat (dataset/masbench_mini/train.jsonl) and nested (dataset/masbench_mini/train/train.jsonl)
    split_file = DATASET_DIR / f"{split}.jsonl"
    if not split_file.exists():
        split_file = DATASET_DIR / split / f"{split}.jsonl"

    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found for split '{split}' in {DATASET_DIR}\n"
            f"Run `python scripts/create_splits.py` or `python scripts/build_dataset.py` first."
        )

    tasks = _load_jsonl_tasks(split_file)
    if filter_func is not None:
        tasks = [t for t in tasks if filter_func(t)]

    logger.info(f"[Loader] Loaded {len(tasks)} tasks from split={split!r}")
    return tasks


def load_all_splits() -> dict[str, list[Task]]:
    """Load all three splits as a dictionary mapping split name to list of Tasks."""
    return {split: load_dataset(split) for split in ["train", "val", "test"]}


def load_all_tasks() -> list[Task]:
    """Load all tasks across all splits (train + val + test).

    Returns:
        Combined list of Task objects from all splits.
    """
    all_tasks: list[Task] = []
    for split in ["train", "val", "test"]:
        try:
            all_tasks.extend(load_dataset(split))
        except FileNotFoundError:
            logger.warning(f"[Loader] Split '{split}' not found, skipping")
    logger.info(f"[Loader] Total tasks loaded: {len(all_tasks)}")
    return all_tasks


def load_batches(split: str, batch_size: int = 32) -> Iterator[list[Task]]:
    """Load dataset in batches of specified size.

    Args:
        split: One of 'train', 'val', 'test'
        batch_size: Number of tasks per batch

    Yields:
        Lists of Task objects.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    tasks = load_dataset(split)
    for i in range(0, len(tasks), batch_size):
        yield tasks[i : i + batch_size]


def get_dataset_metadata() -> dict[str, Any]:
    """Extract metadata for all splits, including task counts and categories."""
    metadata = {}
    for split in _VALID_SPLITS:
        try:
            tasks = load_dataset(split)
            categories = {}
            for t in tasks:
                cat = getattr(t, "category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1
            metadata[split] = {
                "count": len(tasks),
                "categories": categories,
            }
        except FileNotFoundError:
            metadata[split] = {"count": 0, "categories": {}}
    return metadata


def load_task_by_id(task_id: str) -> Task | None:
    """Retrieve a specific task by its ID across all splits (repository-backed operation)."""
    for task in load_all_tasks():
        if getattr(task, "task_id", None) == task_id or getattr(task, "id", None) == task_id:
            return task
    return None


def load_features_df():
    """Load the intermediate feature matrix as a pandas DataFrame.

    Returns:
        pandas.DataFrame with one row per task, columns = features + labels.
        Returns None if pandas is not installed or file doesn't exist.
    """
    features_file = DATASET_DIR.parent / "processed" / "features.jsonl"
    if not features_file.exists():
        logger.warning(f"[Loader] Features file not found: {features_file}")
        return None

    try:
        import pandas as pd  # noqa: PLC0415
    except ImportError:
        logger.warning("[Loader] pandas not installed — cannot load features DataFrame")
        return None

    rows = []
    with features_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    logger.info(f"[Loader] Loaded features DataFrame: {len(df)} rows, {len(df.columns)} cols")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _load_jsonl_tasks(path: Path) -> list[Task]:
    """Read a JSONL file and parse each line into a Task object."""
    tasks: list[Task] = []
    with path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                task = Task(**data)
                tasks.append(task)
            except Exception as e:
                logger.error(f"  Error parsing line {line_num} in {path.name}: {e}")
    return tasks

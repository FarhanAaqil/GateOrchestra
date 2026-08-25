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

from shared.config import DATASET_DIR
from shared.schemas import Task

logger = logging.getLogger(__name__)

_VALID_SPLITS = {"train", "val", "test"}

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def load_dataset(split: str) -> list[Task]:
    """Load tasks for a given split.

    Args:
        split: One of 'train', 'val', 'test'

    Returns:
        List of Task objects for the requested split.

    Raises:
        ValueError: if split is not recognized
        FileNotFoundError: if the split file does not exist yet
    """
    if split not in _VALID_SPLITS:
        raise ValueError(f"split must be one of {_VALID_SPLITS}, got {split!r}")

    split_file = DATASET_DIR / f"{split}.jsonl"

    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}\n"
            f"Run `python scripts/build_dataset.py` first to generate the dataset."
        )

    tasks = _load_jsonl_tasks(split_file)
    logger.info(f"[Loader] Loaded {len(tasks)} tasks from split={split!r}")
    return tasks


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
    with open(features_file, encoding="utf-8") as f:
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
    with open(path, encoding="utf-8") as f:
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

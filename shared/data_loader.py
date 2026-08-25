"""
shared/data_loader.py
======================
Day 2 -- Loads and validates MASBench-mini tasks from .jsonl files.

Usage:
    from shared.data_loader import load_split, load_all

    train_tasks = load_split("train")   # list[Task]
    val_tasks   = load_split("val")
    test_tasks  = load_split("test")

Run as script:
    python shared/data_loader.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import DATASET_DIR
from shared.schemas import Task

SplitName = Literal["train", "val", "test"]


def load_split(split: SplitName) -> list[Task]:
    """Load and validate all tasks from a named split.

    Args:
        split: One of 'train', 'val', 'test'.

    Returns:
        List of validated Task objects.

    Raises:
        FileNotFoundError: If the split file doesn't exist.
        ValueError: If any row fails Pydantic validation.
    """
    path = DATASET_DIR / split / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}\n"
            f"Run: python dataset/generation/build_dataset.py  (Day 1)"
        )

    tasks: list[Task] = []
    errors: list[str] = []

    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                tasks.append(Task(**data))
            except Exception as e:
                errors.append(f"  Line {lineno}: {e}")

    if errors:
        raise ValueError(
            f"Validation errors in {path}:\n" + "\n".join(errors[:5])
        )

    return tasks


def load_all() -> dict[SplitName, list[Task]]:
    """Load all three splits. Returns dict with keys 'train', 'val', 'test'."""
    return {
        "train": load_split("train"),
        "val": load_split("val"),
        "test": load_split("test"),
    }


def split_stats(tasks: list[Task]) -> dict:
    """Compute basic statistics for a list of tasks."""
    depths = [t.depth_score for t in tasks if t.depth_score is not None]
    pars = [t.parallel_score for t in tasks if t.parallel_score is not None]
    sources = {}
    for t in tasks:
        src = t.source_dataset or "unknown"
        sources[src] = sources.get(src, 0) + 1
    return {
        "n": len(tasks),
        "avg_depth": round(sum(depths) / len(depths), 2) if depths else None,
        "avg_parallel": round(sum(pars) / len(pars), 2) if pars else None,
        "sources": sources,
        "has_ground_truth": sum(1 for t in tasks if t.ground_truth),
        "has_context": sum(1 for t in tasks if t.context),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  GateOrchestra -- Data Loader (Day 2)")
    print("=" * 65)

    splits = load_all()
    total = sum(len(v) for v in splits.values())
    print(f"\n  Loaded {total} tasks across 3 splits\n")

    # Stats table header
    print(f"  {'Split':<8} {'Count':>6}  {'Avg Depth':>10}  {'Avg Parallel':>12}  {'Has GT':>7}")
    print(f"  {'-'*8} {'-'*6}  {'-'*10}  {'-'*12}  {'-'*7}")

    for name, tasks in splits.items():
        s = split_stats(tasks)
        gt = s["has_ground_truth"]
        print(
            f"  {name:<8} {s['n']:>6}  {str(s['avg_depth']):>10}  "
            f"{str(s['avg_parallel']):>12}  {gt:>7}"
        )

    print()

    # Source breakdown for train
    train_stats = split_stats(splits["train"])
    print("  Source breakdown (train):")
    for src, cnt in sorted(train_stats["sources"].items()):
        bar = "#" * (cnt // 2)
        pct = cnt / train_stats["n"] * 100
        print(f"    {src:>25}: {cnt:3d} ({pct:.0f}%)  {bar}")

    print()

    # Show 5 sample tasks from val
    print("  Sample val tasks:")
    print(f"  {'task_id':<20} {'D':>3} {'P':>3}  {'question'}")
    print(f"  {'-'*20} {'-'*3} {'-'*3}  {'-'*45}")
    for t in splits["val"][:5]:
        d = t.depth_score or "-"
        p = t.parallel_score or "-"
        q = t.question[:50] + ("..." if len(t.question) > 50 else "")
        print(f"  {t.task_id:<20} {str(d):>3} {str(p):>3}  {q!r}")

    print()
    print("[OK] Day 2 complete. Data loader working on real data.")

"""
dataset/review/sampler.py
==========================
Stratified 20% sample selector for manual dataset review.

Week 6 deliverable (Person 1 — Dataset Engineer).

Selects a reproducible, stratified ~20% sample of tasks across all splits
for human manual review.  Stratification is performed jointly on
``(source_dataset, depth_score)`` so that the review sample preserves the
overall dataset distribution — avoiding the reviewer over-sampling easy or
common tasks.

Design decisions:
- Deterministic: fixed seed means re-running always produces the same sample.
- Stratified: proportional allocation per stratum, with deficit rounding.
- Cross-split: samples from ALL three splits (train/val/test), not just train,
  to catch labeling artefacts introduced across the pipeline.
- Non-destructive: produces a list of task IDs only — no files are modified.

Usage::

    from dataset.review.sampler import ReviewSampler, select_review_sample

    # Quick helper — returns list of Task objects (20% sample)
    sample = select_review_sample()

    # Full control via class
    sampler = ReviewSampler(sample_rate=0.20, seed=42)
    sample = sampler.sample_all_splits()
    sampler.print_summary(sample)

Person 1 owns this file.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Sequence

from shared.schemas import Task

logger = logging.getLogger(__name__)

_DEFAULT_RATE = 0.20
_DEFAULT_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Sampler class
# ─────────────────────────────────────────────────────────────────────────────


class ReviewSampler:
    """Stratified sampler for manual dataset review.

    Selects a reproducible ~``sample_rate`` fraction of tasks from a
    collection, preserving the joint distribution of
    ``(source_dataset, depth_score)``.

    Args:
        sample_rate: Fraction of tasks to select (default 0.20 = 20%).
        seed:        Random seed for reproducibility (default 42).
    """

    def __init__(self, sample_rate: float = _DEFAULT_RATE, seed: int = _DEFAULT_SEED) -> None:
        if not 0 < sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")
        self.sample_rate = sample_rate
        self.seed = seed

    # ------------------------------------------------------------------
    # Core sampling logic
    # ------------------------------------------------------------------

    def sample(self, tasks: Sequence[Task], split_label: str = "all") -> list[Task]:
        """Return a stratified random sample from *tasks*.

        Stratification key: ``(source_dataset, depth_score)``.
        Deficit-based proportional allocation ensures the target count is
        met exactly (or as close as the stratum sizes allow).

        Args:
            tasks:       Pool of Task objects to sample from.
            split_label: Label used only for logging (e.g. ``'train'``).

        Returns:
            Sampled list of Task objects.
        """
        if not tasks:
            logger.warning("[Sampler] Empty task pool for split=%r — returning [].", split_label)
            return []

        target_n = max(1, round(len(tasks) * self.sample_rate))

        # Group by stratum key
        strata: dict[tuple, list[Task]] = defaultdict(list)
        for task in tasks:
            key = (task.source_dataset or "unknown", task.depth_score or 0)
            strata[key].append(task)

        # Proportional allocation with deficit rounding
        allocations: dict[tuple, int] = {}
        total_allocated = 0
        stratum_sizes = {k: len(v) for k, v in strata.items()}

        for key, size in stratum_sizes.items():
            proportion = size / len(tasks)
            exact = proportion * target_n
            floor_val = int(exact)
            allocations[key] = floor_val
            total_allocated += floor_val

        # Distribute remaining slots by largest fractional remainder
        remainder = target_n - total_allocated
        if remainder > 0:
            fractional = {
                k: (stratum_sizes[k] / len(tasks)) * target_n - allocations[k]
                for k in allocations
            }
            for key in sorted(fractional, key=lambda k: -fractional[k])[:remainder]:
                allocations[key] += 1

        # Sample from each stratum
        rng = random.Random(self.seed)
        selected: list[Task] = []
        for key, alloc in allocations.items():
            pool = strata[key]
            n = min(alloc, len(pool))
            selected.extend(rng.sample(pool, n))

        # Shuffle final list so it's not sorted by stratum
        rng.shuffle(selected)

        logger.info(
            "[Sampler] split=%-6s  pool=%d  target=%d  selected=%d  (rate=%.0f%%)",
            split_label, len(tasks), target_n, len(selected), self.sample_rate * 100,
        )
        return selected

    def sample_all_splits(self, dataset_dir=None) -> list[Task]:
        """Sample ~``sample_rate`` tasks from every available split.

        Uses ``JSONLTaskRepository`` to load tasks, then applies ``sample()``
        per split.  The results are combined into a single list.

        Args:
            dataset_dir: Optional path override for the repository root.
                         Defaults to ``shared.config.DATASET_DIR``.

        Returns:
            Combined list of sampled Task objects across all splits.
        """
        from dataset.repository import JSONLTaskRepository

        repo = JSONLTaskRepository(dataset_dir=dataset_dir)
        sampled: list[Task] = []

        for split in repo.split_names():
            split_tasks = repo.list_by_split(split)
            split_sample = self.sample(split_tasks, split_label=split)
            sampled.extend(split_sample)

        logger.info(
            "[Sampler] Total sampled across all splits: %d tasks (%.0f%% of %d)",
            len(sampled), self.sample_rate * 100, repo.count(),
        )
        return sampled

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def print_summary(self, sample: list[Task]) -> None:
        """Print a human-readable summary table of the review sample."""
        from collections import Counter

        source_counts = Counter(t.source_dataset or "unknown" for t in sample)
        depth_counts = Counter(t.depth_score or 0 for t in sample)
        parallel_counts = Counter(t.parallel_score or 0 for t in sample)

        print(f"\n{'='*55}")
        print(f"  Review Sample Summary  ({len(sample)} tasks, {self.sample_rate*100:.0f}% rate)")
        print(f"{'='*55}")
        print("\n  By source_dataset:")
        for src, cnt in source_counts.most_common():
            print(f"    {src:<30} {cnt:>4} tasks")
        print("\n  By depth_score:")
        for d, cnt in sorted(depth_counts.items()):
            bar = "#" * cnt
            print(f"    depth={d}  {bar:<20}  {cnt:>4}")
        print("\n  By parallel_score:")
        for p, cnt in sorted(parallel_counts.items()):
            bar = "#" * cnt
            print(f"    parallel={p}  {bar:<18}  {cnt:>4}")
        print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience helper
# ─────────────────────────────────────────────────────────────────────────────


def select_review_sample(
    sample_rate: float = _DEFAULT_RATE,
    seed: int = _DEFAULT_SEED,
    dataset_dir=None,
) -> list[Task]:
    """One-call helper: load all splits, sample ~20%, return list of Tasks.

    Args:
        sample_rate: Fraction to select (default 0.20).
        seed:        RNG seed (default 42).
        dataset_dir: Optional path override for ``JSONLTaskRepository``.

    Returns:
        List of sampled ``Task`` objects.

    Example::

        from dataset.review.sampler import select_review_sample
        sample = select_review_sample()
        print(f"Selected {len(sample)} tasks for review")
    """
    sampler = ReviewSampler(sample_rate=sample_rate, seed=seed)
    return sampler.sample_all_splits(dataset_dir=dataset_dir)

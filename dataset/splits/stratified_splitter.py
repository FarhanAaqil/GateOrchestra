"""
dataset/splits/stratified_splitter.py
======================================
Stratified dataset partitioning for GateOrchestra MASBench-Mini.

Splits tasks into train, val, and test subsets while preserving joint
distributions across source_dataset, depth_score, and parallel_score.

Person 1 owns this file.
"""

from __future__ import annotations

import collections
import logging
import random
from dataclasses import dataclass, field
from typing import Any

from shared.schemas import Task

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """Container for partitioned dataset splits and metadata."""

    train: list[Task] = field(default_factory=list)
    val: list[Task] = field(default_factory=list)
    test: list[Task] = field(default_factory=list)
    seed: int = 42
    ratios: dict[str, float] = field(
        default_factory=lambda: {"train": 0.60, "val": 0.20, "test": 0.20}
    )

    @property
    def total_count(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)

    def as_dict(self) -> dict[str, list[Task]]:
        return {"train": self.train, "val": self.val, "test": self.test}


class StratifiedSplitter:
    """Partitions Task objects into stratified train/val/test splits."""

    def __init__(
        self,
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
        test_ratio: float = 0.20,
        seed: int = 42,
    ) -> None:
        total = round(train_ratio + val_ratio + test_ratio, 6)
        if total != 1.0:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {train_ratio} + {val_ratio} + {test_ratio} = {total}"
            )
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split(self, tasks: list[Task]) -> SplitResult:
        """Perform stratified split across (source_dataset, depth_score, parallel_score)."""
        if not tasks:
            return SplitResult(
                train=[],
                val=[],
                test=[],
                seed=self.seed,
                ratios={"train": self.train_ratio, "val": self.val_ratio, "test": self.test_ratio},
            )

        rng = random.Random(self.seed)

        # Group tasks into stratification strata
        strata: dict[tuple[str, int, int], list[Task]] = collections.defaultdict(list)
        for t in tasks:
            key = (
                t.source_dataset or "unknown",
                t.depth_score if t.depth_score is not None else 0,
                t.parallel_score if t.parallel_score is not None else 0,
            )
            strata[key].append(t)

        # Exact target counts
        total_tasks = len(tasks)
        target_train = int(round(total_tasks * self.train_ratio))
        target_val = int(round(total_tasks * self.val_ratio))
        target_test = total_tasks - target_train - target_val

        targets = {"train": target_train, "val": target_val, "test": target_test}
        splits: dict[str, list[Task]] = {"train": [], "val": [], "test": []}

        # Sort keys for deterministic order
        sorted_keys = sorted(strata.keys(), key=lambda k: (str(k[0]), int(k[1]), int(k[2])))

        for key in sorted_keys:
            group = list(strata[key])
            rng.shuffle(group)

            for task in group:
                # Find available split with highest remaining deficit
                eligible = [s for s in ["train", "val", "test"] if len(splits[s]) < targets[s]]
                if not eligible:
                    eligible = ["train", "val", "test"]

                best_split = max(
                    eligible,
                    key=lambda s: (targets[s] - len(splits[s])) / max(1, targets[s]),
                )
                splits[best_split].append(task)

        # Sort tasks within each split by task_id for stable determinism
        for s in splits:
            splits[s].sort(key=lambda t: t.task_id)

        result = SplitResult(
            train=splits["train"],
            val=splits["val"],
            test=splits["test"],
            seed=self.seed,
            ratios={"train": self.train_ratio, "val": self.val_ratio, "test": self.test_ratio},
        )

        logger.info(
            f"Stratified split complete: total={result.total_count} "
            f"(train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])})"
        )
        return result


def compute_split_statistics(splits: dict[str, list[Task]]) -> dict[str, Any]:
    """Compute distribution statistics across all splits for verification."""
    stats: dict[str, Any] = {}
    for split_name, task_list in splits.items():
        sources: dict[str, int] = collections.defaultdict(int)
        depths: dict[int, int] = collections.defaultdict(int)
        parallels: dict[int, int] = collections.defaultdict(int)

        for t in task_list:
            if t.source_dataset:
                sources[t.source_dataset] += 1
            if t.depth_score is not None:
                depths[t.depth_score] += 1
            if t.parallel_score is not None:
                parallels[t.parallel_score] += 1

        stats[split_name] = {
            "count": len(task_list),
            "sources": dict(sorted(sources.items())),
            "depth_distribution": dict(sorted(depths.items())),
            "parallel_distribution": dict(sorted(parallels.items())),
        }
    return stats


def split_dataset(
    tasks: list[Task],
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> SplitResult:
    """Convenience function for stratified dataset splitting."""
    splitter = StratifiedSplitter(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    return splitter.split(tasks)

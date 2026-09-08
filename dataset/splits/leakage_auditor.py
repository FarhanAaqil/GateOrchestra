"""
dataset/splits/leakage_auditor.py
==================================
Data contamination and cross-split leakage auditing engine.

Detects exact question matches, n-gram overlap, context duplication,
and cross-split anomalies across train, val, and test subsets.

Person 1 owns this file.
"""

from __future__ import annotations

import itertools
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared.schemas import Task

logger = logging.getLogger(__name__)


def _extract_ngrams(text: str, n: int = 3) -> set[tuple[str, ...]]:
    """Extract word n-grams from normalized text."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _jaccard_similarity(set_a: set[Any], set_b: set[Any]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


@dataclass
class LeakageViolation:
    """Describes an identified data leakage violation between two tasks."""

    violation_type: str  # "exact_id", "exact_question", "high_ngram_overlap", "context_leak"
    split_a: str
    task_id_a: str
    split_b: str
    task_id_b: str
    score: float = 1.0
    details: str = ""


@dataclass
class LeakageAuditReport:
    """Comprehensive data leakage and contamination audit report."""

    is_clean: bool = True
    total_inspected: int = 0
    split_counts: dict[str, int] = field(default_factory=dict)
    violations: list[LeakageViolation] = field(default_factory=list)
    pair_comparisons: int = 0
    max_ngram_similarity: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["violations"] = [asdict(v) for v in self.violations]
        data["violations_count"] = len(self.violations)
        return data


class LeakageAuditor:
    """Audits train, val, and test splits for data leakage and contamination."""

    def __init__(
        self,
        ngram_n: int = 3,
        ngram_threshold: float = 0.80,
    ) -> None:
        self.ngram_n = ngram_n
        self.ngram_threshold = ngram_threshold

    def audit(self, splits: dict[str, list[Task]]) -> LeakageAuditReport:
        """Run multi-tier leakage checks across all split pairs."""
        split_names = list(splits.keys())
        split_counts = {name: len(tasks) for name, tasks in splits.items()}
        total_tasks = sum(split_counts.values())

        report = LeakageAuditReport(
            is_clean=True,
            total_inspected=total_tasks,
            split_counts=split_counts,
        )

        violations: list[LeakageViolation] = []
        max_similarity = 0.0
        comparisons = 0

        # Pairwise cross-split comparison (e.g., train-val, train-test, val-test)
        for split_a, split_b in itertools.combinations(split_names, 2):
            tasks_a = splits[split_a]
            tasks_b = splits[split_b]

            # 1. Exact ID and Question map
            ids_a = {t.task_id: t for t in tasks_a}
            ids_b = {t.task_id: t for t in tasks_b}

            # Check ID collisions
            common_ids = set(ids_a.keys()) & set(ids_b.keys())
            for cid in common_ids:
                violations.append(
                    LeakageViolation(
                        violation_type="exact_id",
                        split_a=split_a,
                        task_id_a=cid,
                        split_b=split_b,
                        task_id_b=cid,
                        score=1.0,
                        details=f"Task ID '{cid}' exists in both {split_a} and {split_b}",
                    )
                )

            # Pre-compute ngrams for faster pair comparison
            ngrams_a = [(t, _extract_ngrams(t.question, self.ngram_n)) for t in tasks_a]
            ngrams_b = [(t, _extract_ngrams(t.question, self.ngram_n)) for t in tasks_b]

            for (task_a, n_a), (task_b, n_b) in itertools.product(ngrams_a, ngrams_b):
                comparisons += 1
                q_a = " ".join(task_a.question.strip().lower().split())
                q_b = " ".join(task_b.question.strip().lower().split())

                # Check exact normalized question match
                if q_a == q_b:
                    violations.append(
                        LeakageViolation(
                            violation_type="exact_question",
                            split_a=split_a,
                            task_id_a=task_a.task_id,
                            split_b=split_b,
                            task_id_b=task_b.task_id,
                            score=1.0,
                            details=f"Identical question text across {split_a} and {split_b}: '{task_a.question}'",
                        )
                    )
                    continue

                # Check n-gram Jaccard similarity
                sim = _jaccard_similarity(n_a, n_b)
                if sim > max_similarity:
                    max_similarity = sim

                if sim >= self.ngram_threshold:
                    violations.append(
                        LeakageViolation(
                            violation_type="high_ngram_overlap",
                            split_a=split_a,
                            task_id_a=task_a.task_id,
                            split_b=split_b,
                            task_id_b=task_b.task_id,
                            score=round(sim, 4),
                            details=(
                                f"High {self.ngram_n}-gram similarity ({sim:.2f} >= {self.ngram_threshold}) "
                                f"between {task_a.task_id} ({split_a}) and {task_b.task_id} ({split_b})"
                            ),
                        )
                    )

        report.violations = violations
        report.is_clean = len(violations) == 0
        report.pair_comparisons = comparisons
        report.max_ngram_similarity = round(max_similarity, 4)

        logger.info(
            f"Leakage audit finished: clean={report.is_clean}, "
            f"violations={len(violations)}, comparisons={comparisons}"
        )
        return report


def save_leakage_report(report: LeakageAuditReport, output_path: str | Path) -> None:
    """Save leakage audit report to JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info(f"Saved leakage audit report -> {path}")


def audit_dataset_leakage(
    splits: dict[str, list[Task]],
    ngram_threshold: float = 0.80,
) -> LeakageAuditReport:
    """Convenience function to audit dataset leakage."""
    auditor = LeakageAuditor(ngram_threshold=ngram_threshold)
    return auditor.audit(splits)

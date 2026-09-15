"""
dataset/review/reviewer.py
===========================
Review log schema and read/write logic for manual task review.

Week 6 deliverable (Person 1 — Dataset Engineer).

Provides:
    ReviewEntry   — Pydantic model for a single human review record.
    ReviewLog     — Manages the persistent JSONL review log file at
                    ``dataset/reviewed/review_log.jsonl``.

Each entry records the reviewer's verdict on one task:
    - approve        — task is correct as-is
    - fix_answer     — ground_truth needs updating (reviewer supplies fix)
    - fix_question   — question wording is bad / ambiguous
    - flag_remove    — task should be dropped from the dataset
    - skip           — reviewer declined to judge

The log is append-only during an interactive session; duplicate task_id
entries are allowed (last-write wins when loading).

Usage::

    from dataset.review.reviewer import ReviewEntry, ReviewLog

    log = ReviewLog()
    entry = ReviewEntry(
        task_id="hotpot_001",
        verdict="approve",
        reviewer="umar",
        notes="Correct and well-formed.",
    )
    log.append(entry)

    # Query
    reviewed_ids = log.reviewed_ids()
    approved = log.filter_by_verdict("approve")

Person 1 owns this file.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Default output path — relative to repo root
_DEFAULT_LOG_PATH = (
    Path(__file__).resolve().parent.parent.parent  # repo root
    / "dataset"
    / "reviewed"
    / "review_log.jsonl"
)

Verdict = Literal["approve", "fix_answer", "fix_question", "flag_remove", "skip"]

VERDICT_LABELS: dict[str, str] = {
    "approve": "Approve        — task is correct as-is",
    "fix_answer": "Fix answer     — ground_truth needs correction",
    "fix_question": "Fix question   — question wording is ambiguous/bad",
    "flag_remove": "Flag remove    — task should be dropped",
    "skip": "Skip           — cannot judge right now",
}


# ─────────────────────────────────────────────────────────────────────────────
# ReviewEntry schema
# ─────────────────────────────────────────────────────────────────────────────


class ReviewEntry(BaseModel):
    """A single human review record for one dataset task.

    Attributes:
        task_id:          The unique task identifier being reviewed.
        verdict:          Reviewer's decision (approve/fix_answer/fix_question/
                          flag_remove/skip).
        reviewer:         Name or handle of the reviewer (for audit trail).
        notes:            Free-text reviewer notes (optional).
        suggested_answer: Corrected ground_truth (required if verdict is
                          ``'fix_answer'``).
        reviewed_at:      UTC ISO-8601 timestamp of review (auto-set).
        split:            Which dataset split this task came from (optional,
                          for traceability).
    """

    task_id: str = Field(..., description="Unique task identifier being reviewed")
    verdict: Verdict = Field(..., description="Reviewer's decision on this task")
    reviewer: str = Field(default="anonymous", description="Reviewer name or handle")
    notes: str = Field(default="", description="Free-text reviewer notes")
    suggested_answer: str | None = Field(
        default=None,
        description="Corrected ground_truth; required when verdict='fix_answer'",
    )
    reviewed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC ISO-8601 timestamp of review",
    )
    split: str | None = Field(
        default=None,
        description="Dataset split this task came from (train/val/test)",
    )

    model_config = {"frozen": False}


# ─────────────────────────────────────────────────────────────────────────────
# ReviewLog — JSONL persistence layer
# ─────────────────────────────────────────────────────────────────────────────


class ReviewLog:
    """Manages the persistent JSONL review log.

    The log file is append-only during live review sessions.  Entries are
    written one-per-line in JSONL format.  When loading, the *last* entry
    for a given ``task_id`` takes precedence (allows reviewers to correct
    earlier decisions).

    Args:
        log_path: Path to the JSONL log file.
                  Defaults to ``dataset/reviewed/review_log.jsonl``.
    """

    def __init__(self, log_path: Path | None = None) -> None:
        self.log_path = Path(log_path) if log_path is not None else _DEFAULT_LOG_PATH
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def append(self, entry: ReviewEntry) -> None:
        """Append a single ReviewEntry to the log file.

        Args:
            entry: The ``ReviewEntry`` to persist.
        """
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json() + "\n")
        logger.debug(
            "[ReviewLog] Appended entry: task_id='%s' verdict='%s'",
            entry.task_id,
            entry.verdict,
        )

    def append_many(self, entries: list[ReviewEntry]) -> None:
        """Append multiple ReviewEntry objects in one write.

        Args:
            entries: List of ``ReviewEntry`` objects to persist.
        """
        with self.log_path.open("a", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(entry.model_dump_json() + "\n")
        logger.info("[ReviewLog] Appended %d entries to %s", len(entries), self.log_path)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_all(self) -> list[ReviewEntry]:
        """Load every ReviewEntry from the log (preserving duplicates).

        Returns:
            List of all ``ReviewEntry`` objects in append order.
        """
        if not self.log_path.exists():
            return []

        entries: list[ReviewEntry] = []
        with self.log_path.open(encoding="utf-8") as fh:
            for line_num, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entries.append(ReviewEntry(**json.loads(raw)))
                except Exception as exc:
                    logger.error("[ReviewLog] Parse error at line %d: %s", line_num, exc)
        return entries

    def load_latest(self) -> dict[str, ReviewEntry]:
        """Load entries as a dict keyed by task_id (last write wins).

        Returns:
            ``{task_id: ReviewEntry}`` with only the most recent entry per task.
        """
        latest: dict[str, ReviewEntry] = {}
        for entry in self.load_all():
            latest[entry.task_id] = entry
        return latest

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def reviewed_ids(self) -> set[str]:
        """Return the set of task_ids that have at least one review entry."""
        return set(self.load_latest().keys())

    def filter_by_verdict(self, verdict: Verdict) -> list[ReviewEntry]:
        """Return the latest review entry per task filtered by *verdict*.

        Args:
            verdict: One of the valid Verdict literals.

        Returns:
            List of matching ``ReviewEntry`` objects.
        """
        return [e for e in self.load_latest().values() if e.verdict == verdict]

    def pending_ids(self, all_task_ids: set[str]) -> set[str]:
        """Return task IDs from *all_task_ids* that have NOT been reviewed.

        Args:
            all_task_ids: Full set of task IDs in the review sample.

        Returns:
            Set of un-reviewed task IDs.
        """
        return all_task_ids - self.reviewed_ids()

    def summary(self) -> dict:
        """Return a dict summarising review progress.

        Returns:
            Dict with keys: ``total``, ``by_verdict`` (counts per verdict),
            ``reviewers`` (unique reviewer names).
        """
        entries = list(self.load_latest().values())
        from collections import Counter

        verdict_counts = Counter(e.verdict for e in entries)
        reviewers = sorted({e.reviewer for e in entries})

        return {
            "total_reviewed": len(entries),
            "by_verdict": dict(verdict_counts),
            "reviewers": reviewers,
            "log_path": str(self.log_path),
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Pretty-print the review progress to stdout."""
        s = self.summary()
        print(f"\n{'='*50}")
        print("  Review Log Summary")
        print(f"  Log: {s['log_path']}")
        print(f"{'='*50}")
        print(f"  Total reviewed : {s['total_reviewed']}")
        print("  By verdict:")
        for verdict, count in sorted(s["by_verdict"].items()):
            label = VERDICT_LABELS.get(verdict, verdict)
            print(f"    {label:<42} {count:>4}")
        print(f"  Reviewers: {', '.join(s['reviewers']) or 'none'}")
        print(f"{'='*50}\n")

    def export_flagged_csv(self, out_path: Path) -> Path:
        """Export tasks flagged for fixing or removal to a CSV for offline editing.

        Args:
            out_path: Destination CSV file path.

        Returns:
            The path to the written CSV file.
        """
        import csv

        flagged_verdicts = {"fix_answer", "fix_question", "flag_remove"}
        flagged = [e for e in self.load_latest().values() if e.verdict in flagged_verdicts]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(ReviewEntry.model_fields.keys())

        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for entry in flagged:
                writer.writerow(entry.model_dump())

        logger.info("[ReviewLog] Exported %d flagged entries to %s", len(flagged), out_path)
        return out_path

    def __repr__(self) -> str:
        s = self.summary()
        return f"ReviewLog(path={self.log_path}, " f"reviewed={s['total_reviewed']})"

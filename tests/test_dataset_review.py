"""
tests/test_dataset_review.py
==============================
Unit & Integration Tests for Week 6 — Manual Review Tooling.

Tests cover:
    ReviewSampler:
        - Correct 20% sample count
        - Stratified distribution preservation
        - Deterministic reproducibility (fixed seed)
        - Edge cases: empty pool, 100% rate, single-task pool

    ReviewEntry:
        - Schema field validation
        - Default timestamp generation
        - Model serialization round-trip

    ReviewLog:
        - append() and load_all()
        - load_latest() last-write-wins deduplication
        - reviewed_ids() and pending_ids()
        - filter_by_verdict()
        - summary() structure and counts
        - export_flagged_csv() correctness
        - Graceful handling of missing / corrupted log file

    Integration:
        - Full sample -> review -> query pipeline
        - Resume mode (skip already-reviewed tasks)

Person 1 owns this file.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from shared.schemas import Task
from dataset.review.sampler import ReviewSampler, select_review_sample
from dataset.review.reviewer import ReviewEntry, ReviewLog, VERDICT_LABELS


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_task(
    task_id: str,
    source: str = "hotpotqa",
    depth: int = 2,
    parallel: int = 1,
    question: str | None = None,
) -> Task:
    return Task(
        task_id=task_id,
        question=question or f"Question for {task_id}?",
        ground_truth=f"Answer for {task_id}",
        depth_score=depth,
        parallel_score=parallel,
        source_dataset=source,
    )


def _make_tasks(n: int, source: str = "hotpotqa", depth: int = 2, offset: int = 0) -> list[Task]:
    return [_make_task(f"{source}_{i+offset:03d}", source=source, depth=depth) for i in range(n)]


# A diverse pool mirroring the real dataset distribution — all IDs unique across sources
POOL = (
    _make_tasks(40, source="hotpotqa", depth=2, offset=0)
    + _make_tasks(30, source="musique", depth=4, offset=0)
    + _make_tasks(20, source="template_arithmetic", depth=1, offset=0)
    + _make_tasks(10, source="template_comparison", depth=3, offset=0)
)  # 100 tasks total, all unique task_ids


@pytest.fixture
def log_file(tmp_path: Path) -> Path:
    return tmp_path / "test_review_log.jsonl"


@pytest.fixture
def review_log(log_file: Path) -> ReviewLog:
    return ReviewLog(log_path=log_file)


@pytest.fixture
def sample_entry() -> ReviewEntry:
    return ReviewEntry(
        task_id="hotpot_001",
        verdict="approve",
        reviewer="tester",
        notes="Looks good.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. ReviewSampler
# ─────────────────────────────────────────────────────────────────────────────


class TestReviewSampler:
    def test_sample_count_approximately_20_percent(self):
        sampler = ReviewSampler(sample_rate=0.20, seed=42)
        sample = sampler.sample(POOL)
        expected = round(len(POOL) * 0.20)
        # Allow +/- 1 for rounding
        assert abs(len(sample) - expected) <= 1

    def test_sample_rate_10_percent(self):
        sampler = ReviewSampler(sample_rate=0.10, seed=42)
        sample = sampler.sample(POOL)
        expected = round(len(POOL) * 0.10)
        assert abs(len(sample) - expected) <= 1

    def test_sample_rate_100_percent(self):
        """100% rate should return all (or nearly all) tasks."""
        pool = _make_tasks(10)
        sampler = ReviewSampler(sample_rate=1.0, seed=42)
        sample = sampler.sample(pool)
        assert len(sample) == len(pool)

    def test_reproducibility_same_seed(self):
        """Same seed must always produce the same sample."""
        sampler = ReviewSampler(sample_rate=0.20, seed=42)
        s1 = sampler.sample(POOL)
        s2 = sampler.sample(POOL)
        assert [t.task_id for t in s1] == [t.task_id for t in s2]

    def test_different_seeds_different_samples(self):
        s1 = ReviewSampler(sample_rate=0.20, seed=42).sample(POOL)
        s2 = ReviewSampler(sample_rate=0.20, seed=99).sample(POOL)
        assert [t.task_id for t in s1] != [t.task_id for t in s2]

    def test_stratified_sources_all_represented(self):
        """Every source_dataset should appear in the sample."""
        sampler = ReviewSampler(sample_rate=0.20, seed=42)
        sample = sampler.sample(POOL)
        sources_in_sample = {t.source_dataset for t in sample}
        sources_in_pool = {t.source_dataset for t in POOL}
        assert sources_in_pool == sources_in_sample

    def test_no_task_sampled_twice(self):
        sampler = ReviewSampler(sample_rate=0.20, seed=42)
        sample = sampler.sample(POOL)
        ids = [t.task_id for t in sample]
        assert len(ids) == len(set(ids))

    def test_empty_pool_returns_empty(self):
        sampler = ReviewSampler(sample_rate=0.20, seed=42)
        result = sampler.sample([])
        assert result == []

    def test_single_task_pool(self):
        task = _make_task("solo_001")
        sampler = ReviewSampler(sample_rate=0.20, seed=42)
        sample = sampler.sample([task])
        # 20% of 1 = 0.2 -> rounds to 0; but max(1, round) -> 1
        assert len(sample) == 1

    def test_invalid_rate_raises(self):
        with pytest.raises(ValueError):
            ReviewSampler(sample_rate=0.0)
        with pytest.raises(ValueError):
            ReviewSampler(sample_rate=1.5)

    def test_all_sampled_tasks_are_from_pool(self):
        sampler = ReviewSampler(sample_rate=0.20, seed=42)
        sample = sampler.sample(POOL)
        pool_ids = {t.task_id for t in POOL}
        for task in sample:
            assert task.task_id in pool_ids

    def test_distribution_roughly_proportional(self):
        """Hotpotqa (40%) should make up roughly 40% of the sample."""
        sampler = ReviewSampler(sample_rate=0.20, seed=42)
        sample = sampler.sample(POOL)
        hotpot_count = sum(1 for t in sample if t.source_dataset == "hotpotqa")
        fraction = hotpot_count / len(sample)
        # Allow generous tolerance: 20% to 60%
        assert 0.20 <= fraction <= 0.60


class TestSelectReviewSampleHelper:
    def test_helper_uses_defaults(self, tmp_path):
        """select_review_sample() should work with a real JSONL dir."""
        from dataset.repository import JSONLTaskRepository

        # Write tasks to a temp dir
        train_tasks = _make_tasks(20, source="hotpotqa", depth=2)
        val_tasks = _make_tasks(10, source="musique", depth=3)
        for split, tasks in [("train", train_tasks), ("val", val_tasks)]:
            path = tmp_path / f"{split}.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for t in tasks:
                    fh.write(t.model_dump_json() + "\n")

        sample = select_review_sample(sample_rate=0.20, seed=42, dataset_dir=tmp_path)
        assert len(sample) >= 1
        assert all(isinstance(t, Task) for t in sample)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ReviewEntry schema
# ─────────────────────────────────────────────────────────────────────────────


class TestReviewEntry:
    def test_required_fields(self, sample_entry):
        assert sample_entry.task_id == "hotpot_001"
        assert sample_entry.verdict == "approve"

    def test_default_reviewer(self):
        entry = ReviewEntry(task_id="x", verdict="skip")
        assert entry.reviewer == "anonymous"

    def test_default_notes_empty(self):
        entry = ReviewEntry(task_id="x", verdict="approve")
        assert entry.notes == ""

    def test_auto_timestamp_set(self):
        entry = ReviewEntry(task_id="x", verdict="approve")
        assert entry.reviewed_at != ""
        assert "T" in entry.reviewed_at  # ISO format has T separator

    def test_suggested_answer_optional(self):
        entry = ReviewEntry(task_id="x", verdict="fix_answer", suggested_answer="Paris")
        assert entry.suggested_answer == "Paris"

    def test_suggested_answer_none_by_default(self):
        entry = ReviewEntry(task_id="x", verdict="approve")
        assert entry.suggested_answer is None

    def test_split_field_optional(self):
        entry = ReviewEntry(task_id="x", verdict="approve", split="train")
        assert entry.split == "train"

    def test_serialization_round_trip(self, sample_entry):
        dumped = sample_entry.model_dump_json()
        loaded = ReviewEntry(**json.loads(dumped))
        assert loaded.task_id == sample_entry.task_id
        assert loaded.verdict == sample_entry.verdict
        assert loaded.reviewer == sample_entry.reviewer

    def test_all_verdicts_valid(self):
        for verdict in ("approve", "fix_answer", "fix_question", "flag_remove", "skip"):
            entry = ReviewEntry(task_id="x", verdict=verdict)  # type: ignore[arg-type]
            assert entry.verdict == verdict

    def test_verdict_labels_dict_complete(self):
        for verdict in ("approve", "fix_answer", "fix_question", "flag_remove", "skip"):
            assert verdict in VERDICT_LABELS


# ─────────────────────────────────────────────────────────────────────────────
# 3. ReviewLog — persistence
# ─────────────────────────────────────────────────────────────────────────────


class TestReviewLogAppendLoad:
    def test_append_creates_file(self, review_log, log_file, sample_entry):
        assert not log_file.exists() or log_file.stat().st_size == 0
        review_log.append(sample_entry)
        assert log_file.exists()

    def test_load_all_returns_all_entries(self, review_log, sample_entry):
        review_log.append(sample_entry)
        review_log.append(ReviewEntry(task_id="hotpot_002", verdict="skip"))
        entries = review_log.load_all()
        assert len(entries) == 2

    def test_load_all_preserves_order(self, review_log):
        ids = [f"task_{i}" for i in range(5)]
        for tid in ids:
            review_log.append(ReviewEntry(task_id=tid, verdict="approve"))
        loaded_ids = [e.task_id for e in review_log.load_all()]
        assert loaded_ids == ids

    def test_append_many(self, review_log):
        entries = [ReviewEntry(task_id=f"t_{i}", verdict="approve") for i in range(5)]
        review_log.append_many(entries)
        assert len(review_log.load_all()) == 5

    def test_entries_are_review_entry_instances(self, review_log, sample_entry):
        review_log.append(sample_entry)
        for entry in review_log.load_all():
            assert isinstance(entry, ReviewEntry)

    def test_missing_log_returns_empty(self, log_file):
        log = ReviewLog(log_path=log_file / "nonexistent" / "log.jsonl")
        assert log.load_all() == []

    def test_corrupted_line_skipped(self, review_log, log_file, sample_entry):
        review_log.append(sample_entry)
        # Inject a bad line
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write("{this is not valid json\n")
        review_log.append(ReviewEntry(task_id="hotpot_002", verdict="skip"))

        entries = review_log.load_all()
        # Corrupted line skipped; 2 valid entries remain
        assert len(entries) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 4. ReviewLog — load_latest (last-write-wins)
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadLatest:
    def test_last_write_wins_for_same_task_id(self, review_log):
        review_log.append(ReviewEntry(task_id="t1", verdict="approve"))
        review_log.append(ReviewEntry(task_id="t1", verdict="flag_remove"))

        latest = review_log.load_latest()
        assert latest["t1"].verdict == "flag_remove"

    def test_unique_task_ids_all_present(self, review_log):
        for tid in ["a", "b", "c"]:
            review_log.append(ReviewEntry(task_id=tid, verdict="approve"))
        latest = review_log.load_latest()
        assert set(latest.keys()) == {"a", "b", "c"}

    def test_empty_log_returns_empty_dict(self, review_log):
        assert review_log.load_latest() == {}


# ─────────────────────────────────────────────────────────────────────────────
# 5. ReviewLog — query helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestReviewLogQueryHelpers:
    def _populate(self, log: ReviewLog) -> None:
        entries = [
            ReviewEntry(task_id="t1", verdict="approve"),
            ReviewEntry(task_id="t2", verdict="fix_answer", suggested_answer="Paris"),
            ReviewEntry(task_id="t3", verdict="fix_question"),
            ReviewEntry(task_id="t4", verdict="flag_remove"),
            ReviewEntry(task_id="t5", verdict="skip"),
        ]
        log.append_many(entries)

    def test_reviewed_ids(self, review_log):
        self._populate(review_log)
        assert review_log.reviewed_ids() == {"t1", "t2", "t3", "t4", "t5"}

    def test_pending_ids_all_unreviewed(self, review_log):
        all_ids = {"t1", "t2", "t3", "t4", "t5", "t6"}
        self._populate(review_log)
        pending = review_log.pending_ids(all_ids)
        assert pending == {"t6"}

    def test_pending_ids_none_remaining(self, review_log):
        all_ids = {"t1", "t2", "t3", "t4", "t5"}
        self._populate(review_log)
        pending = review_log.pending_ids(all_ids)
        assert pending == set()

    def test_filter_by_verdict_approve(self, review_log):
        self._populate(review_log)
        approved = review_log.filter_by_verdict("approve")
        assert len(approved) == 1
        assert approved[0].task_id == "t1"

    def test_filter_by_verdict_fix_answer(self, review_log):
        self._populate(review_log)
        fixes = review_log.filter_by_verdict("fix_answer")
        assert len(fixes) == 1
        assert fixes[0].suggested_answer == "Paris"

    def test_filter_by_verdict_empty_result(self, review_log):
        review_log.append(ReviewEntry(task_id="t1", verdict="approve"))
        result = review_log.filter_by_verdict("flag_remove")
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# 6. ReviewLog — summary()
# ─────────────────────────────────────────────────────────────────────────────


class TestReviewLogSummary:
    def test_summary_total_count(self, review_log):
        for i in range(5):
            review_log.append(ReviewEntry(task_id=f"t{i}", verdict="approve"))
        s = review_log.summary()
        assert s["total_reviewed"] == 5

    def test_summary_by_verdict_counts(self, review_log):
        review_log.append(ReviewEntry(task_id="t1", verdict="approve"))
        review_log.append(ReviewEntry(task_id="t2", verdict="approve"))
        review_log.append(ReviewEntry(task_id="t3", verdict="skip"))
        s = review_log.summary()
        assert s["by_verdict"]["approve"] == 2
        assert s["by_verdict"]["skip"] == 1

    def test_summary_reviewers(self, review_log):
        review_log.append(ReviewEntry(task_id="t1", verdict="approve", reviewer="alice"))
        review_log.append(ReviewEntry(task_id="t2", verdict="skip", reviewer="bob"))
        s = review_log.summary()
        assert "alice" in s["reviewers"]
        assert "bob" in s["reviewers"]

    def test_summary_has_log_path(self, review_log):
        s = review_log.summary()
        assert "log_path" in s

    def test_empty_log_summary(self, review_log):
        s = review_log.summary()
        assert s["total_reviewed"] == 0
        assert s["by_verdict"] == {}


# ─────────────────────────────────────────────────────────────────────────────
# 7. ReviewLog — export_flagged_csv()
# ─────────────────────────────────────────────────────────────────────────────


class TestExportFlaggedCSV:
    def _populate_mixed(self, log: ReviewLog) -> None:
        log.append_many([
            ReviewEntry(task_id="t1", verdict="approve"),
            ReviewEntry(task_id="t2", verdict="fix_answer", suggested_answer="Berlin"),
            ReviewEntry(task_id="t3", verdict="fix_question", notes="Ambiguous"),
            ReviewEntry(task_id="t4", verdict="flag_remove", notes="Duplicate"),
            ReviewEntry(task_id="t5", verdict="skip"),
        ])

    def test_creates_csv_file(self, review_log, tmp_path):
        self._populate_mixed(review_log)
        out = tmp_path / "flagged.csv"
        review_log.export_flagged_csv(out)
        assert out.exists()

    def test_csv_excludes_approve_and_skip(self, review_log, tmp_path):
        self._populate_mixed(review_log)
        out = tmp_path / "flagged.csv"
        review_log.export_flagged_csv(out)
        with out.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        task_ids = {row["task_id"] for row in rows}
        assert "t1" not in task_ids  # approve
        assert "t5" not in task_ids  # skip

    def test_csv_includes_flagged_tasks(self, review_log, tmp_path):
        self._populate_mixed(review_log)
        out = tmp_path / "flagged.csv"
        review_log.export_flagged_csv(out)
        with out.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        task_ids = {row["task_id"] for row in rows}
        assert {"t2", "t3", "t4"} == task_ids

    def test_csv_has_all_fields(self, review_log, tmp_path):
        self._populate_mixed(review_log)
        out = tmp_path / "flagged.csv"
        review_log.export_flagged_csv(out)
        with out.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = set(reader.fieldnames or [])
        expected = {"task_id", "verdict", "reviewer", "notes", "reviewed_at"}
        assert expected.issubset(fieldnames)

    def test_no_flagged_tasks_creates_empty_csv(self, review_log, tmp_path):
        review_log.append(ReviewEntry(task_id="t1", verdict="approve"))
        out = tmp_path / "flagged.csv"
        review_log.export_flagged_csv(out)
        with out.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert rows == []


# ─────────────────────────────────────────────────────────────────────────────
# 8. Integration: full sample -> review -> query pipeline
# ─────────────────────────────────────────────────────────────────────────────


class TestFullPipeline:
    def test_sample_review_query(self, review_log):
        """Sample from pool -> review all -> query results."""
        pool = _make_tasks(20)
        sampler = ReviewSampler(sample_rate=0.20, seed=42)
        sample = sampler.sample(pool)

        # Simulate reviewing each task
        for task in sample:
            entry = ReviewEntry(
                task_id=task.task_id,
                verdict="approve",
                reviewer="integration_test",
            )
            review_log.append(entry)

        # All sampled tasks should now be in reviewed_ids
        sampled_ids = {t.task_id for t in sample}
        reviewed = review_log.reviewed_ids()
        assert sampled_ids == reviewed

    def test_pending_ids_empty_after_full_review(self, review_log):
        pool = _make_tasks(10)
        sampler = ReviewSampler(sample_rate=1.0, seed=42)
        sample = sampler.sample(pool)

        all_ids = {t.task_id for t in sample}
        for task in sample:
            review_log.append(ReviewEntry(task_id=task.task_id, verdict="approve"))

        assert review_log.pending_ids(all_ids) == set()

    def test_resume_skips_already_reviewed(self, review_log):
        """Simulates --resume: tasks reviewed in session 1 are skipped in session 2."""
        pool = _make_tasks(20)
        sampler = ReviewSampler(sample_rate=0.50, seed=42)
        sample = sampler.sample(pool)

        # Session 1: review half
        session_1 = sample[: len(sample) // 2]
        for task in session_1:
            review_log.append(ReviewEntry(task_id=task.task_id, verdict="approve"))

        # Session 2: compute what's still pending
        already_reviewed = review_log.reviewed_ids()
        pending = [t for t in sample if t.task_id not in already_reviewed]

        assert len(pending) == len(sample) - len(session_1)

    def test_verdict_distribution_in_summary(self, review_log):
        entries = [
            ReviewEntry(task_id="t1", verdict="approve"),
            ReviewEntry(task_id="t2", verdict="approve"),
            ReviewEntry(task_id="t3", verdict="fix_answer", suggested_answer="X"),
            ReviewEntry(task_id="t4", verdict="flag_remove"),
        ]
        review_log.append_many(entries)
        s = review_log.summary()
        assert s["by_verdict"]["approve"] == 2
        assert s["by_verdict"]["fix_answer"] == 1
        assert s["by_verdict"]["flag_remove"] == 1
        assert s["total_reviewed"] == 4

"""
tests/test_dataset_splits.py
=============================
Unit and integration tests for dataset stratified splitting and leakage auditing (Week 4).

Tests:
- StratifiedSplitter ratios, distributions, and determinism
- LeakageAuditor sensitivity against exact and n-gram contamination
- Dataset loader split integration
"""

from __future__ import annotations

import json

import pytest

from dataset.loader import load_all_splits, load_all_tasks, load_dataset
from dataset.splits.leakage_auditor import (
    LeakageAuditor,
    audit_dataset_leakage,
    save_leakage_report,
)
from dataset.splits.stratified_splitter import (
    StratifiedSplitter,
    compute_split_statistics,
    split_dataset,
)
from shared.schemas import Task


def _make_dummy_task(
    task_id: str,
    question: str,
    source: str = "hotpotqa_style",
    depth: int = 3,
    parallel: int = 1,
    ground_truth: str = "Answer",
) -> Task:
    return Task(
        task_id=task_id,
        question=question,
        context=f"Context for {task_id}",
        depth_score=depth,
        parallel_score=parallel,
        ground_truth=ground_truth,
        source_dataset=source,
    )


@pytest.fixture
def synthetic_task_pool() -> list[Task]:
    """Generate 60 diverse synthetic tasks for fast unit testing."""
    tasks: list[Task] = []
    sources = ["hotpotqa_style", "musique_style", "template_arithmetic", "template_comparison"]
    for i in range(60):
        src = sources[i % len(sources)]
        depth = (i % 5) + 1
        parallel = (i % 4) + 1
        tasks.append(
            _make_dummy_task(
                task_id=f"synth_{i:03d}",
                question=f"Synthetic unique reasoning question number {i} for source {src}?",
                source=src,
                depth=depth,
                parallel=parallel,
                ground_truth=f"Gold Answer {i}",
            )
        )
    return tasks


class TestStratifiedSplitter:
    def test_split_ratios_and_count_preservation(self, synthetic_task_pool):
        splitter = StratifiedSplitter(train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, seed=42)
        result = splitter.split(synthetic_task_pool)

        assert result.total_count == 60
        # 60% of 60 = 36, 20% = 12, 20% = 12
        assert len(result.train) == 36
        assert len(result.val) == 12
        assert len(result.test) == 12

        # Check unique task IDs
        all_ids = [t.task_id for t in result.train + result.val + result.test]
        assert len(set(all_ids)) == 60

    def test_distribution_preservation(self, synthetic_task_pool):
        result = split_dataset(synthetic_task_pool, seed=42)
        stats = compute_split_statistics(result.as_dict())

        # Check that all sources appear across train, val, and test
        for split_name in ["train", "val", "test"]:
            assert "hotpotqa_style" in stats[split_name]["sources"]
            assert "musique_style" in stats[split_name]["sources"]
            assert "template_arithmetic" in stats[split_name]["sources"]
            assert "template_comparison" in stats[split_name]["sources"]

    def test_deterministic_reproducibility(self, synthetic_task_pool):
        res_a = split_dataset(synthetic_task_pool, seed=42)
        res_b = split_dataset(synthetic_task_pool, seed=42)
        res_c = split_dataset(synthetic_task_pool, seed=99)

        ids_a = [t.task_id for t in res_a.train]
        ids_b = [t.task_id for t in res_b.train]
        ids_c = [t.task_id for t in res_c.train]

        assert ids_a == ids_b
        assert ids_a != ids_c

    def test_invalid_ratios_raise(self):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            StratifiedSplitter(train_ratio=0.70, val_ratio=0.20, test_ratio=0.20)

    def test_empty_task_list(self):
        res = split_dataset([], seed=42)
        assert res.total_count == 0
        assert len(res.train) == 0
        assert len(res.val) == 0
        assert len(res.test) == 0


class TestLeakageAuditor:
    def test_clean_splits_pass(self, synthetic_task_pool):
        res = split_dataset(synthetic_task_pool, seed=42)
        auditor = LeakageAuditor(ngram_n=3, ngram_threshold=0.80)
        report = auditor.audit(res.as_dict())

        assert report.is_clean is True
        assert len(report.violations) == 0
        assert report.total_inspected == 60
        assert report.pair_comparisons > 0

    def test_detect_exact_id_leakage(self, synthetic_task_pool):
        res = split_dataset(synthetic_task_pool, seed=42)
        splits = res.as_dict()
        # Inject duplicate ID in test
        splits["test"].append(splits["train"][0])

        report = audit_dataset_leakage(splits)
        assert report.is_clean is False
        id_violations = [v for v in report.violations if v.violation_type == "exact_id"]
        assert len(id_violations) >= 1
        assert id_violations[0].task_id_a == splits["train"][0].task_id

    def test_detect_exact_question_leakage(self):
        task_a = _make_dummy_task("id_001", "What is the capital of France?")
        task_b = _make_dummy_task("id_002", "what is the capital of france?  ")

        splits = {
            "train": [task_a],
            "val": [task_b],
            "test": [],
        }
        report = audit_dataset_leakage(splits)
        assert report.is_clean is False
        q_violations = [v for v in report.violations if v.violation_type == "exact_question"]
        assert len(q_violations) == 1

    def test_detect_high_ngram_overlap(self):
        task_a = _make_dummy_task(
            "id_101",
            "What is the official capital city of the European country named France?",
        )
        task_b = _make_dummy_task(
            "id_102",
            "What is the official capital city of the European country named France today?",
        )

        splits = {
            "train": [task_a],
            "val": [task_b],
            "test": [],
        }
        auditor = LeakageAuditor(ngram_n=3, ngram_threshold=0.70)
        report = auditor.audit(splits)
        assert report.is_clean is False
        overlap_violations = [
            v for v in report.violations if v.violation_type == "high_ngram_overlap"
        ]
        assert len(overlap_violations) == 1

    def test_save_and_load_leakage_report(self, tmp_path):
        auditor = LeakageAuditor()
        report = auditor.audit({"train": [_make_dummy_task("t1", "Q1?")], "val": [], "test": []})
        out_file = tmp_path / "leakage_report.json"
        save_leakage_report(report, out_file)

        assert out_file.exists()
        with out_file.open(encoding="utf-8") as f:
            data = json.load(f)
        assert data["is_clean"] is True
        assert data["total_inspected"] == 1


class TestDatasetLoaderIntegration:
    def test_loader_functions(self):
        splits = load_all_splits()
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

        train_tasks = load_dataset("train")
        assert len(train_tasks) > 0
        assert all(isinstance(t, Task) for t in train_tasks)

        all_tasks = load_all_tasks()
        assert len(all_tasks) == len(splits["train"]) + len(splits["val"]) + len(splits["test"])

    def test_invalid_split_raises(self):
        with pytest.raises(ValueError, match="split must be one of"):
            load_dataset("unknown_split")

"""
tests/test_dataset_repository.py
==================================
Unit & Integration Tests for Week 5 — Public API & Storage Abstraction.

Tests cover:
    - JSONLTaskRepository interface contracts
    - Filter accuracy (source, depth, parallel, split)
    - Batch iteration correctness
    - Stream iteration
    - save() / overwrite semantics
    - Cache invalidation
    - get_by_id() across splits
    - count() and split_names()
    - TaskRepository ABC enforcement
    - Loader Week 5 additions (load_batches, get_dataset_metadata, load_task_by_id)
    - CSV export validity (scripts/export_dataset.py helpers)

Person 1 owns this file.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from shared.schemas import Task

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

# Sample tasks used across tests
SAMPLE_TASKS = [
    Task(
        task_id="hotpot_001",
        question="Who was the president of France in 2020?",
        ground_truth="Emmanuel Macron",
        depth_score=2,
        parallel_score=1,
        source_dataset="hotpotqa",
    ),
    Task(
        task_id="hotpot_002",
        question="What is the capital of Germany?",
        ground_truth="Berlin",
        depth_score=1,
        parallel_score=1,
        source_dataset="hotpotqa",
    ),
    Task(
        task_id="musique_001",
        question="Who founded the company that built the Eiffel Tower?",
        ground_truth="Gustave Eiffel",
        depth_score=4,
        parallel_score=3,
        source_dataset="musique",
    ),
    Task(
        task_id="musique_002",
        question="What year was the Louvre museum originally built?",
        ground_truth="1546",
        depth_score=3,
        parallel_score=2,
        source_dataset="musique",
    ),
    Task(
        task_id="arith_001",
        question="If Alice has 3 apples and Bob has 5 times as many, how many does Bob have?",
        ground_truth="15",
        depth_score=2,
        parallel_score=2,
        source_dataset="template_arithmetic",
    ),
]

# Split assignment for fixtures
TRAIN_TASKS = SAMPLE_TASKS[:3]
VAL_TASKS = SAMPLE_TASKS[3:4]
TEST_TASKS = SAMPLE_TASKS[4:]


def _write_jsonl(path: Path, tasks: list[Task]) -> None:
    """Helper to write tasks to a JSONL file."""
    with path.open("w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(t.model_dump_json() + "\n")


@pytest.fixture
def tmp_repo_dir(tmp_path: Path):
    """Create a temporary directory with JSONL split files."""
    _write_jsonl(tmp_path / "train.jsonl", TRAIN_TASKS)
    _write_jsonl(tmp_path / "val.jsonl", VAL_TASKS)
    _write_jsonl(tmp_path / "test.jsonl", TEST_TASKS)
    return tmp_path


@pytest.fixture
def repo(tmp_repo_dir):
    """Return a JSONLTaskRepository pointing at the temp dir."""
    from dataset.repository import JSONLTaskRepository

    return JSONLTaskRepository(dataset_dir=tmp_repo_dir)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Abstract Interface Enforcement
# ─────────────────────────────────────────────────────────────────────────────


class TestTaskRepositoryABC:
    def test_cannot_instantiate_abstract_class(self):
        """TaskRepository is abstract and cannot be instantiated directly."""
        from dataset.repository import TaskRepository

        with pytest.raises(TypeError):
            TaskRepository()  # type: ignore[abstract]

    def test_concrete_class_satisfies_abc(self, repo):
        """JSONLTaskRepository satisfies all abstract methods."""
        from dataset.repository import TaskRepository

        assert isinstance(repo, TaskRepository)


# ─────────────────────────────────────────────────────────────────────────────
# 2. list_by_split() — basic contract
# ─────────────────────────────────────────────────────────────────────────────


class TestListBySplit:
    def test_train_returns_correct_tasks(self, repo):
        tasks = repo.list_by_split("train")
        assert len(tasks) == len(TRAIN_TASKS)
        ids = {t.task_id for t in tasks}
        assert ids == {t.task_id for t in TRAIN_TASKS}

    def test_val_returns_correct_tasks(self, repo):
        tasks = repo.list_by_split("val")
        assert len(tasks) == len(VAL_TASKS)

    def test_test_returns_correct_tasks(self, repo):
        tasks = repo.list_by_split("test")
        assert len(tasks) == len(TEST_TASKS)

    def test_all_tasks_are_task_instances(self, repo):
        for split in ("train", "val", "test"):
            for task in repo.list_by_split(split):
                assert isinstance(task, Task)

    def test_invalid_split_raises_value_error(self, repo):
        with pytest.raises(ValueError, match="split must be one of"):
            repo.list_by_split("bogus")

    def test_returns_copy_not_reference(self, repo):
        """list_by_split should return a copy so mutation does not affect cache."""
        tasks = repo.list_by_split("train")
        original_len = len(tasks)
        tasks.clear()
        # Fetch again — should still return original count
        tasks2 = repo.list_by_split("train")
        assert len(tasks2) == original_len


# ─────────────────────────────────────────────────────────────────────────────
# 3. get_by_id()
# ─────────────────────────────────────────────────────────────────────────────


class TestGetById:
    def test_returns_correct_task(self, repo):
        task = repo.get_by_id("hotpot_001")
        assert task is not None
        assert task.task_id == "hotpot_001"
        assert task.ground_truth == "Emmanuel Macron"

    def test_finds_task_across_splits(self, repo):
        """get_by_id should search all splits, not just train."""
        # arith_001 is in test split
        task = repo.get_by_id("arith_001")
        assert task is not None
        assert task.task_id == "arith_001"

    def test_returns_none_for_missing_id(self, repo):
        result = repo.get_by_id("nonexistent_999")
        assert result is None

    def test_returns_task_instance(self, repo):
        task = repo.get_by_id("musique_001")
        assert isinstance(task, Task)


# ─────────────────────────────────────────────────────────────────────────────
# 4. filter()
# ─────────────────────────────────────────────────────────────────────────────


class TestFilter:
    def test_filter_by_source(self, repo):
        results = repo.filter(source="hotpotqa")
        assert all(t.source_dataset == "hotpotqa" for t in results)
        assert len(results) == 2  # hotpot_001 and hotpot_002

    def test_filter_by_depth(self, repo):
        results = repo.filter(depth=2)
        assert all(t.depth_score == 2 for t in results)
        # hotpot_001 (depth=2) and arith_001 (depth=2)
        assert len(results) == 2

    def test_filter_by_parallel(self, repo):
        results = repo.filter(parallel=1)
        assert all(t.parallel_score == 1 for t in results)

    def test_filter_combined_source_and_depth(self, repo):
        results = repo.filter(source="musique", depth=4)
        assert len(results) == 1
        assert results[0].task_id == "musique_001"

    def test_filter_restricted_to_split(self, repo):
        """filter(split=...) should only search that split."""
        # musique_001 is in train, musique_002 is in val
        results = repo.filter(source="musique", split="train")
        assert all(t.source_dataset == "musique" for t in results)
        ids = {t.task_id for t in results}
        assert "musique_002" not in ids  # in val, not train

    def test_filter_no_criteria_returns_all(self, repo):
        """filter() with no arguments returns all tasks."""
        all_tasks = repo.filter()
        assert len(all_tasks) == len(SAMPLE_TASKS)

    def test_filter_no_match_returns_empty(self, repo):
        results = repo.filter(source="nonexistent_dataset")
        assert results == []

    def test_filter_invalid_split_raises(self, repo):
        with pytest.raises(ValueError):
            repo.filter(split="invalid_split")

    def test_filter_returns_task_instances(self, repo):
        for task in repo.filter(source="hotpotqa"):
            assert isinstance(task, Task)


# ─────────────────────────────────────────────────────────────────────────────
# 5. all()
# ─────────────────────────────────────────────────────────────────────────────


class TestAll:
    def test_all_returns_every_task(self, repo):
        all_tasks = repo.all()
        assert len(all_tasks) == len(SAMPLE_TASKS)

    def test_all_ids_are_unique(self, repo):
        ids = [t.task_id for t in repo.all()]
        assert len(ids) == len(set(ids))


# ─────────────────────────────────────────────────────────────────────────────
# 6. count()
# ─────────────────────────────────────────────────────────────────────────────


class TestCount:
    def test_total_count(self, repo):
        assert repo.count() == len(SAMPLE_TASKS)

    def test_per_split_count(self, repo):
        assert repo.count("train") == len(TRAIN_TASKS)
        assert repo.count("val") == len(VAL_TASKS)
        assert repo.count("test") == len(TEST_TASKS)

    def test_counts_sum_to_total(self, repo):
        total = sum(repo.count(s) for s in ("train", "val", "test"))
        assert total == repo.count()

    def test_invalid_split_raises(self, repo):
        with pytest.raises(ValueError):
            repo.count("garbage_split")


# ─────────────────────────────────────────────────────────────────────────────
# 7. split_names()
# ─────────────────────────────────────────────────────────────────────────────


class TestSplitNames:
    def test_returns_all_three_splits(self, repo):
        names = repo.split_names()
        assert set(names) == {"train", "val", "test"}

    def test_returns_only_existing_splits(self, tmp_path):
        """When only train exists, split_names returns only ['train']."""
        from dataset.repository import JSONLTaskRepository

        _write_jsonl(tmp_path / "train.jsonl", TRAIN_TASKS)
        repo = JSONLTaskRepository(dataset_dir=tmp_path)
        assert repo.split_names() == ["train"]

    def test_empty_dir_returns_empty_list(self, tmp_path):
        from dataset.repository import JSONLTaskRepository

        repo = JSONLTaskRepository(dataset_dir=tmp_path)
        assert repo.split_names() == []


# ─────────────────────────────────────────────────────────────────────────────
# 8. save()
# ─────────────────────────────────────────────────────────────────────────────


class TestSave:
    def test_save_new_task_increases_count(self, repo, tmp_repo_dir):
        new_task = Task(
            task_id="new_task_999",
            question="What is the airspeed velocity of an unladen swallow?",
            ground_truth="African or European?",
            depth_score=5,
            parallel_score=4,
            source_dataset="hotpotqa",
        )
        before = repo.count("train")
        repo.save(new_task, split="train")
        after = repo.count("train")
        assert after == before + 1

    def test_save_overwrites_existing_task(self, repo, tmp_repo_dir):
        """Saving a task with an existing task_id should overwrite it."""
        original = repo.get_by_id("hotpot_001")
        assert original is not None

        updated = Task(
            task_id="hotpot_001",
            question="UPDATED QUESTION",
            ground_truth="Updated Answer",
            depth_score=5,
            parallel_score=4,
            source_dataset="hotpotqa",
        )
        before = repo.count("train")
        repo.save(updated, split="train")
        after = repo.count("train")

        # Count should be the same (overwrite, not append)
        assert after == before

        # The fetched task should reflect the update
        fetched = repo.get_by_id("hotpot_001")
        assert fetched is not None
        assert fetched.question == "UPDATED QUESTION"

    def test_save_persists_to_disk(self, repo, tmp_repo_dir):
        """After save(), the JSONL file on disk should contain the new task."""
        new_task = Task(
            task_id="disk_persist_001",
            question="Does save write to disk?",
            ground_truth="Yes",
            depth_score=1,
            parallel_score=1,
            source_dataset="hotpotqa",
        )
        repo.save(new_task, split="train")

        # Read disk directly (bypassing the repo cache)
        raw = (tmp_repo_dir / "train.jsonl").read_text(encoding="utf-8")
        assert "disk_persist_001" in raw

    def test_save_invalid_split_raises(self, repo):
        task = Task(
            task_id="x",
            question="q?",
            ground_truth="a",
            depth_score=1,
            parallel_score=1,
            source_dataset="hotpotqa",
        )
        with pytest.raises(ValueError):
            repo.save(task, split="invalid")


# ─────────────────────────────────────────────────────────────────────────────
# 9. stream()
# ─────────────────────────────────────────────────────────────────────────────


class TestStream:
    def test_stream_yields_all_tasks(self, repo):
        tasks = list(repo.stream("train"))
        assert len(tasks) == len(TRAIN_TASKS)

    def test_stream_yields_task_instances(self, repo):
        for task in repo.stream("val"):
            assert isinstance(task, Task)

    def test_stream_lazy_iteration(self, repo):
        """stream() should be an iterator/generator, not a list."""
        gen = repo.stream("train")
        # Should support next() without loading all at once
        first = next(gen)
        assert isinstance(first, Task)

    def test_stream_invalid_split_raises(self, repo):
        with pytest.raises(ValueError):
            list(repo.stream("not_a_split"))

    def test_stream_missing_file_yields_nothing(self, tmp_path):
        """stream() on a missing file should yield nothing (not crash)."""
        from dataset.repository import JSONLTaskRepository

        repo = JSONLTaskRepository(dataset_dir=tmp_path)
        tasks = list(repo.stream("train"))
        assert tasks == []


# ─────────────────────────────────────────────────────────────────────────────
# 10. batch()
# ─────────────────────────────────────────────────────────────────────────────


class TestBatch:
    def test_batches_cover_all_tasks(self, repo):
        all_batched = []
        for batch in repo.batch("train", batch_size=2):
            all_batched.extend(batch)
        assert len(all_batched) == len(TRAIN_TASKS)

    def test_batch_size_respected(self, repo):
        """All batches except possibly the last should have exactly batch_size tasks."""
        batches = list(repo.batch("train", batch_size=2))
        for batch in batches[:-1]:
            assert len(batch) == 2
        # Last batch may be smaller
        assert len(batches[-1]) >= 1

    def test_batch_size_1(self, repo):
        batches = list(repo.batch("train", batch_size=1))
        assert len(batches) == len(TRAIN_TASKS)
        for b in batches:
            assert len(b) == 1

    def test_batch_size_larger_than_split(self, repo):
        """batch_size > split size should yield a single batch."""
        batches = list(repo.batch("val", batch_size=1000))
        assert len(batches) == 1
        assert len(batches[0]) == len(VAL_TASKS)

    def test_batch_invalid_size_raises(self, repo):
        with pytest.raises(ValueError):
            list(repo.batch("train", batch_size=0))

    def test_batch_yields_task_instances(self, repo):
        for batch in repo.batch("test", batch_size=2):
            for task in batch:
                assert isinstance(task, Task)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Cache invalidation
# ─────────────────────────────────────────────────────────────────────────────


class TestCacheInvalidation:
    def test_invalidate_specific_split_clears_only_that_split(self, repo):
        # Warm the cache
        _ = repo.list_by_split("train")
        _ = repo.list_by_split("val")
        assert "train" in repo._cache
        assert "val" in repo._cache

        repo.invalidate_cache("train")
        assert "train" not in repo._cache
        assert "val" in repo._cache  # val should still be cached

    def test_invalidate_all_clears_cache(self, repo):
        _ = repo.list_by_split("train")
        _ = repo.list_by_split("val")
        repo.invalidate_cache()
        assert repo._cache == {}

    def test_save_invalidates_split_cache(self, repo, tmp_repo_dir):
        """save() must invalidate the cache for the affected split."""
        _ = repo.list_by_split("train")  # warm cache
        assert "train" in repo._cache

        new_task = Task(
            task_id="cache_bust_001",
            question="Cache busting test?",
            ground_truth="Yes",
            depth_score=1,
            parallel_score=1,
            source_dataset="hotpotqa",
        )
        repo.save(new_task, split="train")
        # After save, train cache must be cleared
        assert "train" not in repo._cache


# ─────────────────────────────────────────────────────────────────────────────
# 12. Loader Week 5 API
# ─────────────────────────────────────────────────────────────────────────────


class TestLoaderWeek5:
    def test_load_batches_covers_all_tasks(self, tmp_repo_dir, monkeypatch):
        """load_batches() should yield batches covering all tasks."""
        from dataset import loader as ldr

        # Patch DATASET_DIR to point at tmp_repo_dir
        monkeypatch.setattr(ldr, "DATASET_DIR", tmp_repo_dir)

        all_batched = []
        for batch in ldr.load_batches("train", batch_size=2):
            all_batched.extend(batch)
        assert len(all_batched) == len(TRAIN_TASKS)

    def test_load_batches_invalid_size_raises(self, tmp_repo_dir, monkeypatch):
        from dataset import loader as ldr

        monkeypatch.setattr(ldr, "DATASET_DIR", tmp_repo_dir)
        with pytest.raises(ValueError):
            list(ldr.load_batches("train", batch_size=0))

    def test_get_dataset_metadata_keys(self, tmp_repo_dir, monkeypatch):
        """get_dataset_metadata returns expected top-level keys."""
        from dataset import loader as ldr

        monkeypatch.setattr(ldr, "DATASET_DIR", tmp_repo_dir)
        meta = ldr.get_dataset_metadata()
        assert "train" in meta
        assert "count" in meta["train"]

    def test_load_task_by_id_found(self, tmp_repo_dir, monkeypatch):
        from dataset import loader as ldr

        monkeypatch.setattr(ldr, "DATASET_DIR", tmp_repo_dir)
        task = ldr.load_task_by_id("hotpot_001")
        assert task is not None
        assert task.task_id == "hotpot_001"

    def test_load_task_by_id_not_found(self, tmp_repo_dir, monkeypatch):
        from dataset import loader as ldr

        monkeypatch.setattr(ldr, "DATASET_DIR", tmp_repo_dir)
        task = ldr.load_task_by_id("nonexistent_xyz_999")
        assert task is None


# ─────────────────────────────────────────────────────────────────────────────
# 13. CSV Export Validity (export_dataset.py helpers)
# ─────────────────────────────────────────────────────────────────────────────


class TestCSVExport:
    def test_export_csv_per_split_creates_files(self, repo, tmp_path):
        from scripts.export_dataset import export_csv_per_split

        out_dir = tmp_path / "csv_out"
        exported = export_csv_per_split(repo, ["train", "val", "test"], out_dir)

        assert "train" in exported
        assert exported["train"].exists()
        assert exported["val"].exists()
        assert exported["test"].exists()

    def test_csv_has_correct_row_count(self, repo, tmp_path):
        from scripts.export_dataset import export_csv_per_split

        out_dir = tmp_path / "csv_out"
        exported = export_csv_per_split(repo, ["train"], out_dir)

        with exported["train"].open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == len(TRAIN_TASKS)

    def test_csv_contains_expected_columns(self, repo, tmp_path):
        from scripts.export_dataset import _TASK_FIELDS, export_csv_per_split

        out_dir = tmp_path / "csv_out"
        exported = export_csv_per_split(repo, ["train"], out_dir)

        with exported["train"].open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames

        assert fieldnames is not None
        for field in _TASK_FIELDS:
            assert field in fieldnames

    def test_csv_task_ids_match_source(self, repo, tmp_path):
        from scripts.export_dataset import export_csv_per_split

        out_dir = tmp_path / "csv_out"
        exported = export_csv_per_split(repo, ["train"], out_dir)

        with exported["train"].open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            exported_ids = {row["task_id"] for row in reader}

        expected_ids = {t.task_id for t in TRAIN_TASKS}
        assert exported_ids == expected_ids

    def test_export_csv_combined_has_split_column(self, repo, tmp_path):
        from scripts.export_dataset import export_csv_combined

        out_dir = tmp_path / "combined_out"
        out_path = export_csv_combined(repo, ["train", "val", "test"], out_dir)

        assert out_path.exists()
        with out_path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            assert "split" in (reader.fieldnames or [])
            rows = list(reader)

        assert len(rows) == len(SAMPLE_TASKS)

    def test_export_csv_combined_split_values_correct(self, repo, tmp_path):
        from scripts.export_dataset import export_csv_combined

        out_dir = tmp_path / "combined_out"
        out_path = export_csv_combined(repo, ["train", "val"], out_dir)

        with out_path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            split_values = {row["split"] for row in reader}

        assert split_values == {"train", "val"}

    def test_export_metadata_json_structure(self, repo, tmp_path):
        from scripts.export_dataset import export_metadata_json

        out_path = export_metadata_json(repo, ["train", "val", "test"], tmp_path)
        assert out_path.exists()

        with out_path.open(encoding="utf-8") as fh:
            meta = json.load(fh)

        assert "total" in meta
        assert "splits" in meta
        assert meta["total"] == len(SAMPLE_TASKS)
        assert "train" in meta["splits"]
        assert meta["splits"]["train"]["count"] == len(TRAIN_TASKS)

    def test_export_metadata_includes_distributions(self, repo, tmp_path):
        from scripts.export_dataset import export_metadata_json

        out_path = export_metadata_json(repo, ["train"], tmp_path)
        with out_path.open(encoding="utf-8") as fh:
            meta = json.load(fh)

        train_meta = meta["splits"]["train"]
        assert "source_distribution" in train_meta
        assert "depth_distribution" in train_meta
        assert "parallel_distribution" in train_meta


# ─────────────────────────────────────────────────────────────────────────────
# 14. Integration: repr and edge cases
# ─────────────────────────────────────────────────────────────────────────────


class TestRepositoryIntegration:
    def test_repr_contains_root_and_counts(self, repo):
        r = repr(repo)
        assert "JSONLTaskRepository" in r
        assert "train" in r

    def test_empty_repo_returns_empty_lists(self, tmp_path):
        from dataset.repository import JSONLTaskRepository

        repo = JSONLTaskRepository(dataset_dir=tmp_path)
        assert repo.all() == []
        assert repo.count() == 0
        assert repo.split_names() == []

    def test_missing_split_file_returns_empty_list(self, tmp_path):
        from dataset.repository import JSONLTaskRepository

        _write_jsonl(tmp_path / "train.jsonl", TRAIN_TASKS)
        repo = JSONLTaskRepository(dataset_dir=tmp_path)
        # val does not exist
        result = repo.list_by_split("val")
        assert result == []

    def test_malformed_jsonl_line_is_skipped(self, tmp_path):
        """Corrupted JSONL lines should be skipped with a log error, not crash."""
        from dataset.repository import JSONLTaskRepository

        path = tmp_path / "train.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            fh.write(TRAIN_TASKS[0].model_dump_json() + "\n")
            fh.write("{this is not valid JSON\n")
            fh.write(TRAIN_TASKS[1].model_dump_json() + "\n")

        repo = JSONLTaskRepository(dataset_dir=tmp_path)
        tasks = repo.list_by_split("train")
        # 2 valid lines, 1 corrupted line skipped
        assert len(tasks) == 2

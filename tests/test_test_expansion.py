"""The 96-task test split, its provenance and the dataset review record."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset.generation.test_expansion import build_aliases, build_new_tasks
from scripts.expand_test_split import _supported_by_context, check_tasks
from shared.data_loader import dataset_sha256, load_split
from shared.grading import grade_task

REVIEW = Path(__file__).resolve().parent.parent / "dataset" / "masbench_mini" / "REVIEW.json"


def _ids(split: str) -> set[str]:
    return {t.task_id for t in load_split(split)}  # type: ignore[arg-type]


def test_split_sizes_and_no_overlap() -> None:
    train, val, test = _ids("train"), _ids("val"), _ids("test")
    assert (len(train), len(val), len(test)) == (97, 32, 96)
    assert not (train & val) and not (train & test) and not (val & test)


def test_new_tasks_live_only_in_the_test_split() -> None:
    new = {t["task_id"] for t in build_new_tasks()}
    assert len(new) == 64 and new <= _ids("test")
    assert not (new & (_ids("train") | _ids("val")))


def test_every_new_gold_and_alias_grades_correct() -> None:
    tasks = {t.task_id: t for t in load_split("test")}  # type: ignore[arg-type]
    for new in build_new_tasks():
        task = tasks[new["task_id"]]
        assert grade_task(task.ground_truth, task) is True
        for alias in task.answer_aliases:
            assert grade_task(alias, task) is True, (task.task_id, alias)
    assert set(build_aliases()) <= set(tasks)


def test_new_tasks_pass_the_expansion_checks() -> None:
    assert check_tasks(build_new_tasks(), build_aliases()) == []


def test_checks_reject_duplicates_missing_context_and_unsupported_answers() -> None:
    template = dict(build_new_tasks()[20])  # a bridge task with context
    existing = next(t for t in load_split("train") if t.question)  # type: ignore[arg-type]
    duplicate = {**template, "task_id": "bridge_900", "question": existing.question}
    assert any("repeats" in p for p in check_tasks([duplicate], {}))
    no_context = {**template, "task_id": "bridge_901", "context": None}
    assert any("needs a context" in p for p in check_tasks([no_context], {}))
    unsupported = {**template, "task_id": "bridge_902", "ground_truth": "Zanzibar"}
    assert any("not supported" in p for p in check_tasks([unsupported], {}))


def test_derived_numeric_answers_are_checked_against_the_context() -> None:
    assert _supported_by_context("20 years", "It happened in 1969. Then again in 1989.")
    assert not _supported_by_context("21 years", "It happened in 1969. Then again in 1989.")
    assert _supported_by_context("2", "Mars has two moons.")
    assert not _supported_by_context("3", "Mars has two moons.")
    assert _supported_by_context("Danube", "The Danube is about 2,850 km long.")


def test_arithmetic_golds_are_computed_not_typed() -> None:
    by_id = {t["task_id"]: t for t in build_new_tasks()}
    assert by_id["arith_041"]["ground_truth"] == "$27.25"
    assert by_id["arith_057"]["ground_truth"] == "$5,832.00"
    assert by_id["arith_058"]["ground_truth"] == "2 hours"


def test_review_record_matches_the_dataset() -> None:
    record = json.loads(REVIEW.read_text(encoding="utf-8"))
    added = record["assistant_reviewed"]
    reviewed, by_assistant = set(record["reviewed_task_ids"]), set(added["task_ids"])
    everything = _ids("train") | _ids("val") | _ids("test")
    assert record["status"] == "reviewed" and record["reviewed_by"] and record["reviewed_on"]
    assert added["reviewed_by"] and added["reviewed_on"] and added["findings"]
    assert not (reviewed & by_assistant)
    assert reviewed | by_assistant == everything
    assert by_assistant == {t["task_id"] for t in build_new_tasks()}


def test_review_goes_stale_when_the_dataset_changes() -> None:
    """Editing any task changes the hash, which forces someone to look at REVIEW.json again."""
    record = json.loads(REVIEW.read_text(encoding="utf-8"))
    assert (
        record["dataset_sha256"] == dataset_sha256()
    ), "The dataset changed since it was reviewed. Re-review it, then update REVIEW.json."


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_every_task_has_the_fields_the_benchmark_needs(split: str) -> None:
    for task in load_split(split):  # type: ignore[arg-type]
        assert task.question.strip() and task.ground_truth and task.source_dataset
        assert 1 <= task.depth_score <= 5 and 1 <= task.parallel_score <= 4

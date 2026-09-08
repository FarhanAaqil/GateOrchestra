"""
tests/test_dataset_validation.py
================================
Unit and integration tests for dataset validation and structural checks (Week 3):
  - field constraints and length boundaries
  - collection integrity and ID collision detection
  - DatasetValidator suite and ValidationReport generation
  - file validation and error handling
"""

import json
from pathlib import Path

from dataset.validation.structural_checks import (
    check_collection_integrity,
    check_task_fields,
)
from dataset.validation.validator import DatasetValidator, ValidationReport
from shared.schemas import Task


class TestTaskStructuralChecks:
    def test_valid_task_passes(self):
        task = Task(
            task_id="bridge_001",
            question="Which director won an Academy Award for Inception?",
            ground_truth="Christopher Nolan",
            source_dataset="hotpotqa_style",
            depth_score=2,
            parallel_score=1,
        )
        errors = check_task_fields(task)
        assert len(errors) == 0

    def test_task_id_errors(self):
        # Pydantic schema validation rejects spaces
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Task(
                task_id="bridge 001",
                question="What is the capital of Japan?",
                ground_truth="Tokyo",
                source_dataset="hotpotqa_style",
                depth_score=1,
                parallel_score=1,
            )

    def test_question_length_boundaries(self):
        # Too short (< 10 chars)
        task_short = Task(
            task_id="short_01",
            question="Short?",
            ground_truth="Answer",
            source_dataset="hotpotqa_style",
            depth_score=1,
            parallel_score=1,
        )
        errors = check_task_fields(task_short, min_q_len=10)
        assert any("too short" in e for e in errors)

    def test_invalid_source_dataset(self):
        task_bad_src = Task(
            task_id="src_01",
            question="What is the distance between the Earth and the Moon?",
            ground_truth="384,400 km",
            source_dataset="unauthorized_source",
            depth_score=1,
            parallel_score=1,
        )
        errors = check_task_fields(task_bad_src)
        assert any("Unknown source_dataset" in e for e in errors)

    def test_score_out_of_bounds(self):
        task_bad_scores = Task(
            task_id="score_01",
            question="What is the chemical symbol for Helium?",
            ground_truth="He",
            source_dataset="hotpotqa_style",
            depth_score=5,
            parallel_score=4,
        )
        # Check custom bounds
        errors = check_task_fields(
            task_bad_scores,
            depth_range=(1, 3),
            parallel_range=(1, 2),
        )
        assert any("depth_score" in e for e in errors)
        assert any("parallel_score" in e for e in errors)


class TestCollectionIntegrity:
    def test_empty_collection(self):
        violations = check_collection_integrity([])
        assert len(violations) == 1
        assert violations[0]["type"] == "empty_dataset"

    def test_task_id_collision_detected(self):
        t1 = Task(
            task_id="dup_id",
            question="What is the capital of Italy?",
            ground_truth="Rome",
            source_dataset="hotpotqa_style",
            depth_score=1,
            parallel_score=1,
        )
        t2 = Task(
            task_id="dup_id",
            question="What is the capital of Spain?",
            ground_truth="Madrid",
            source_dataset="hotpotqa_style",
            depth_score=1,
            parallel_score=1,
        )
        violations = check_collection_integrity([t1, t2])
        assert any(v["type"] == "task_id_collision" for v in violations)

    def test_duplicate_question_detected(self):
        t1 = Task(
            task_id="id_1",
            question="What is the capital of France?",
            ground_truth="Paris",
            source_dataset="hotpotqa_style",
            depth_score=1,
            parallel_score=1,
        )
        t2 = Task(
            task_id="id_2",
            question="What is the capital of France?",
            ground_truth="Paris",
            source_dataset="hotpotqa_style",
            depth_score=1,
            parallel_score=1,
        )
        violations = check_collection_integrity([t1, t2])
        assert any(v["type"] == "duplicate_question" for v in violations)


class TestDatasetValidatorSuite:
    def _make_dummy_dataset(self, n: int = 155) -> list[Task]:
        tasks = []
        sources = [
            "hotpotqa_style",
            "musique_style",
            "template_arithmetic",
            "template_comparison",
        ]
        for i in range(n):
            tasks.append(
                Task(
                    task_id=f"task_{i:03d}",
                    question=f"Question number {i} asking something very interesting and informative?",
                    ground_truth=f"Answer {i}",
                    source_dataset=sources[i % len(sources)],
                    depth_score=(i % 5) + 1,
                    parallel_score=(i % 4) + 1,
                )
            )
        return tasks

    def test_validator_success(self):
        tasks = self._make_dummy_dataset(160)
        validator = DatasetValidator(
            config={
                "min_acceptable_size": 150,
                "sources": [
                    "hotpotqa_style",
                    "musique_style",
                    "template_arithmetic",
                    "template_comparison",
                ],
                "validation": {
                    "min_question_length": 10,
                    "max_question_length": 600,
                    "min_answer_length": 1,
                    "max_answer_length": 200,
                    "valid_depth_range": [1, 5],
                    "valid_parallel_range": [1, 4],
                },
            }
        )
        report = validator.validate(tasks)
        assert report.is_valid is True
        assert report.passed_count == 160
        assert report.failed_count == 0
        assert report.min_size_met is True
        assert "PASS" in report.summary_text()

    def test_validator_under_sized_fails(self):
        tasks = self._make_dummy_dataset(20)
        validator = DatasetValidator(
            config={
                "min_acceptable_size": 150,
                "sources": ["hotpotqa_style", "musique_style"],
                "validation": {},
            }
        )
        report = validator.validate(tasks)
        assert report.is_valid is False
        assert report.min_size_met is False
        assert any(e["type"] == "insufficient_dataset_size" for e in report.errors)

    def test_save_and_load_report(self, tmp_path: Path):
        report = ValidationReport(
            is_valid=True,
            total_tasks=160,
            passed_count=160,
            failed_count=0,
            min_size_met=True,
            statistics={"source_distribution": {"hotpotqa_style": 160}},
        )
        out_file = tmp_path / "report.json"
        DatasetValidator.save_report(report, out_file)
        assert out_file.exists()

        with out_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["is_valid"] is True
        assert data["total_tasks"] == 160

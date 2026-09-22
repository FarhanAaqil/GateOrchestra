"""
tests/test_dataset_error_analysis.py
====================================
Unit and integration tests for error taxonomy, failure classification,
and diagnostic profiling engine (Week 8).

Person 1 owns this test suite.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from dataset.error_labels import (
    ERROR_DESCRIPTIONS,
    ErrorAnalyzer,
    ErrorAnnotation,
    ErrorType,
)
from shared.schemas import Task


@pytest.fixture
def sample_tasks() -> list[Task]:
    """Sample diverse tasks for error analysis testing."""
    return [
        Task(
            task_id="err_001",
            question="Calculate 12 * 8.",
            ground_truth="96",
            source_dataset="template_arithmetic",
            depth_score=2,
            parallel_score=1,
            context="12 multiplied by 8 is 96.",
        ),
        Task(
            task_id="err_002",
            question="What are the capitals of France and Germany?",
            ground_truth="Paris, Berlin",
            source_dataset="hotpotqa_style",
            depth_score=3,
            parallel_score=2,
            context="Paris is the capital of France and Berlin is the capital of Germany.",
        ),
        Task(
            task_id="err_003",
            question="Which river is longer: Nile or Amazon?",
            ground_truth="Nile",
            source_dataset="template_comparison",
            depth_score=2,
            parallel_score=1,
            context="The Nile is longer than the Amazon.",
        ),
        Task(
            task_id="err_004",
            question="Who was the maternal grandfather of King George III?",
            ground_truth="Frederick II, Duke of Saxe-Gotha-Altenburg",
            source_dataset="musique_style",
            depth_score=4,
            parallel_score=1,
            context="Princess Augusta was mother of George III. Her father was Frederick II.",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 1. ErrorType Taxonomy Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestErrorType:
    def test_error_types_enumeration(self):
        assert ErrorType.REASONING_ERROR.value == "reasoning_error"
        assert ErrorType.CALCULATION_ERROR.value == "calculation_error"
        assert ErrorType.FACTUAL_ERROR.value == "factual_error"
        assert ErrorType.CONCURRENCY_ERROR.value == "concurrency_error"
        assert ErrorType.FORMAT_ERROR.value == "format_error"
        assert ErrorType.CONTEXT_MISALIGNMENT.value == "context_misalignment"
        assert ErrorType.UNKNOWN.value == "unknown"

    def test_all_types_helper(self):
        all_types = ErrorType.all_types()
        assert len(all_types) == 7
        assert "calculation_error" in all_types

    def test_error_descriptions_completeness(self):
        for e_type in ErrorType:
            assert e_type in ERROR_DESCRIPTIONS
            assert len(ERROR_DESCRIPTIONS[e_type]) > 10


# ─────────────────────────────────────────────────────────────────────────────
# 2. ErrorAnnotation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestErrorAnnotation:
    def test_defaults_and_serialization(self):
        ann = ErrorAnnotation(
            task_id="t_01",
            error_type=ErrorType.CALCULATION_ERROR,
            predicted_answer="95",
            ground_truth="96",
        )
        assert ann.severity == "MAJOR"
        assert ann.rationale == ""
        d = ann.to_dict()
        assert isinstance(d, dict)
        assert d["error_type"] == "calculation_error"
        assert d["predicted_answer"] == "95"
        assert d["ground_truth"] == "96"


# ─────────────────────────────────────────────────────────────────────────────
# 3. ErrorAnalyzer Diagnosis Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestErrorAnalyzerDiagnosis:
    def test_exact_match_variants(self):
        assert ErrorAnalyzer.is_exact_match("Paris", "paris")
        assert ErrorAnalyzer.is_exact_match(" Paris. ", "'paris'")
        assert not ErrorAnalyzer.is_exact_match("London", "Paris")

    def test_calculation_error_heuristic(self, sample_tasks: list[Task]):
        analyzer = ErrorAnalyzer()
        t = sample_tasks[0]  # arithmetic task
        ann = analyzer.diagnose_error(t, predicted_answer="92")
        assert ann.error_type == ErrorType.CALCULATION_ERROR
        assert ann.task_id == "err_001"

    def test_concurrency_error_heuristic(self, sample_tasks: list[Task]):
        analyzer = ErrorAnalyzer()
        t = sample_tasks[1]  # multi-part Paris, Berlin
        ann = analyzer.diagnose_error(t, predicted_answer="Paris")
        assert ann.error_type == ErrorType.CONCURRENCY_ERROR

    def test_format_error_heuristic(self, sample_tasks: list[Task]):
        analyzer = ErrorAnalyzer()
        t = sample_tasks[2]
        ann_empty = analyzer.diagnose_error(t, predicted_answer="")
        assert ann_empty.error_type == ErrorType.FORMAT_ERROR

        ann_none = analyzer.diagnose_error(t, predicted_answer="None")
        assert ann_none.error_type == ErrorType.FORMAT_ERROR

    def test_override_error_type(self, sample_tasks: list[Task]):
        analyzer = ErrorAnalyzer()
        t = sample_tasks[0]
        ann = analyzer.diagnose_error(
            t,
            predicted_answer="92",
            error_type_override=ErrorType.FACTUAL_ERROR,
            severity="CRITICAL",
            rationale="Custom factual override",
        )
        assert ann.error_type == ErrorType.FACTUAL_ERROR
        assert ann.severity == "CRITICAL"
        assert ann.rationale == "Custom factual override"


# ─────────────────────────────────────────────────────────────────────────────
# 4. ErrorBreakdown and Reporting Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestErrorBreakdownAndReporting:
    def test_compute_breakdown(self, sample_tasks: list[Task]):
        analyzer = ErrorAnalyzer()
        ann1 = analyzer.diagnose_error(sample_tasks[0], "90")
        ann2 = analyzer.diagnose_error(sample_tasks[1], "Paris")

        breakdown = analyzer.compute_breakdown([ann1, ann2])
        assert breakdown.total_errors == 2
        assert breakdown.by_error_type["calculation_error"] == 1
        assert breakdown.by_error_type["concurrency_error"] == 1
        assert "template_arithmetic" in breakdown.by_source
        assert "depth_2" in breakdown.by_depth

    def test_analyze_predictions_all_correct(self, sample_tasks: list[Task]):
        analyzer = ErrorAnalyzer()
        preds = {t.task_id: str(t.ground_truth) for t in sample_tasks}
        report = analyzer.analyze_predictions(sample_tasks, preds)
        assert report.total_evaluated == 4
        assert report.total_errors == 0
        assert report.error_rate_pct == 0.0
        assert len(report.recommendations) > 0

    def test_analyze_predictions_partial_failures(self, sample_tasks: list[Task]):
        analyzer = ErrorAnalyzer()
        preds = {
            "err_001": "95",  # wrong
            "err_002": "Paris, Berlin",  # correct
            "err_003": "Nile",  # correct
            "err_004": "Wrong king",  # wrong
        }
        report = analyzer.analyze_predictions(sample_tasks, preds)
        assert report.total_evaluated == 4
        assert report.total_errors == 2
        assert report.error_rate_pct == 50.0
        assert len(report.annotations) == 2

    def test_render_ascii_summary(self, sample_tasks: list[Task]):
        analyzer = ErrorAnalyzer()
        preds = {"err_001": "95"}
        report = analyzer.analyze_predictions(sample_tasks, preds)
        ascii_text = analyzer.render_ascii_summary(report)
        assert "GATEORCHESTRA ERROR ANALYSIS" in ascii_text
        assert "ERROR CATEGORY BREAKDOWN" in ascii_text

    def test_render_markdown_report(self, sample_tasks: list[Task]):
        analyzer = ErrorAnalyzer()
        preds = {"err_001": "95"}
        report = analyzer.analyze_predictions(sample_tasks, preds)
        md_text = analyzer.render_markdown_report(report)
        assert "# GateOrchestra — Error Analysis & Diagnostic Report" in md_text
        assert "## 1. Failure Modes by Error Taxonomy" in md_text

    def test_export_report_files(self, sample_tasks: list[Task], tmp_path: Path):
        analyzer = ErrorAnalyzer()
        preds = {t.task_id: str(t.ground_truth) for t in sample_tasks}
        preds["err_001"] = "95"  # 1 error out of 4
        report = analyzer.analyze_predictions(sample_tasks, preds)

        j_out = tmp_path / "err.json"
        m_out = tmp_path / "err.md"
        analyzer.export_report(report, json_path=j_out, md_path=m_out)

        assert j_out.exists()
        assert m_out.exists()
        with j_out.open("r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["total_evaluated"] == 4
            assert data["total_errors"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLI Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCLIAnalyzeErrors:
    def test_cli_help(self):
        res = subprocess.run(
            [sys.executable, "scripts/analyze_errors.py", "--help"],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        assert "--predictions-file" in res.stdout
        assert "--output-json" in res.stdout

    def test_cli_diagnostic_run(self):
        res = subprocess.run(
            [sys.executable, "scripts/analyze_errors.py", "--split", "test", "--no-save"],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        assert "GATEORCHESTRA ERROR ANALYSIS" in res.stdout

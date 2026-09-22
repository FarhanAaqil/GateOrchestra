"""tests/test_error_analysis.py
=============================
Unit and integration tests for GateOrchestra Error Analysis (Person 2).

Covers:
  - ErrorCategory enum and dataclass representations
  - Error classification logic (False STOP, Failed ESCALATE, Budget Exhaustion, etc.)
  - analyze_errors_from_records with synthetic and EvalResult records
  - Gate error breakdown rates (False STOP rate, Failed ESCALATE rate)
  - analyze_errors_from_file on real evaluation traces (logs/week2_baseline_results.jsonl)
  - scripts/analyze_errors.py CLI entrypoint
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.error_analysis import (
    AgentErrorBreakdown,
    ErrorAnalysisReport,
    ErrorCategory,
    ErrorRecord,
    GateErrorBreakdown,
    analyze_errors_from_file,
    analyze_errors_from_records,
    classify_error,
)
from scripts.analyze_errors import main as cli_main
from shared.schemas import EvalResult, GateDecision

# ─────────────────────────────────────────────────────────────────────────────
# 1. Tests for Data Models & Serialization
# ─────────────────────────────────────────────────────────────────────────────


class TestErrorDataModels:
    def test_error_category_enum_values(self):
        assert ErrorCategory.FALSE_STOP.value == "false_stop"
        assert ErrorCategory.FALSE_ESCALATE_FAILED.value == "false_escalate_failed"
        assert ErrorCategory.BUDGET_EXHAUSTION.value == "budget_exhaustion"
        assert ErrorCategory.EMPTY_OR_UNPARSED.value == "empty_or_unparsed"
        assert ErrorCategory.ARITHMETIC_ERROR.value == "arithmetic_error"

    def test_error_record_to_dict(self):
        rec = ErrorRecord(
            task_id="t1",
            method="GateOrchestra",
            predicted_answer="Wrong Answer",
            ground_truth="Right Answer",
            category=ErrorCategory.FALSE_STOP,
            gate_decision="STOP",
            confidence=0.95,
            tokens_spent=120,
            explanation="Premature exit",
        )
        d = rec.to_dict()
        assert d["task_id"] == "t1"
        assert d["category"] == "false_stop"
        assert d["confidence"] == 0.95

    def test_report_serialization(self):
        gate_breakdown = GateErrorBreakdown(
            total_decisions=10,
            total_stop=6,
            total_escalate=4,
            correct_stop=5,
            false_stop=1,
            correct_escalate=3,
            failed_escalate=1,
            false_stop_rate=16.67,
            failed_escalate_rate=25.0,
        )
        agent_breakdown = {
            "GateOrchestra": AgentErrorBreakdown(
                agent_or_method="GateOrchestra",
                total_evaluations=10,
                error_count=2,
                error_rate=20.0,
                category_counts={"false_stop": 1, "failed_escalate": 1},
                avg_tokens_on_error=250.0,
            )
        }
        report = ErrorAnalysisReport(
            total_tasks_evaluated=10,
            total_errors=2,
            overall_error_rate=20.0,
            gate_errors=gate_breakdown,
            agent_breakdowns=agent_breakdown,
            category_counts={"false_stop": 1, "failed_escalate": 1},
            category_percentages={"false_stop": 50.0, "failed_escalate": 50.0},
            source_info="unit_test",
        )

        md = report.to_markdown()
        assert "GateOrchestra -- Error Analysis Report" in md
        assert "False STOP (Under-routing)" in md
        assert "16.7% of STOP" in md
        assert "GateOrchestra" in md

        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed["total_tasks_evaluated"] == 10
        assert parsed["gate_errors"]["false_stop"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tests for Error Classification
# ─────────────────────────────────────────────────────────────────────────────


class TestClassifyError:
    def test_classify_false_stop(self):
        cat, expl = classify_error(
            predicted="Berlin",
            ground_truth="Paris",
            gate_decision="STOP",
            confidence=0.98,
        )
        assert cat == ErrorCategory.FALSE_STOP
        assert "Gate issued STOP" in expl
        assert "0.98" in expl

    def test_classify_failed_escalate(self):
        cat, expl = classify_error(
            predicted="Berlin",
            ground_truth="Paris",
            gate_decision="ESCALATE",
            tokens_spent=300,
            budget_cap=1000,
        )
        assert cat == ErrorCategory.FALSE_ESCALATE_FAILED
        assert "multi-agent orchestrator failed" in expl

    def test_classify_budget_exhaustion(self):
        cat, expl = classify_error(
            predicted="Step 3: ...",
            ground_truth="Paris",
            gate_decision="ESCALATE",
            tokens_spent=500,
            budget_cap=500,
        )
        assert cat == ErrorCategory.BUDGET_EXHAUSTION
        assert "full budget cap" in expl

    def test_classify_empty_or_defeatist(self):
        cat1, _ = classify_error(predicted="", ground_truth="Paris")
        assert cat1 == ErrorCategory.EMPTY_OR_UNPARSED

        cat2, _ = classify_error(predicted="Unable to determine", ground_truth="Paris")
        assert cat2 == ErrorCategory.EMPTY_OR_UNPARSED

        cat3, _ = classify_error(predicted="I don't know", ground_truth="Paris")
        assert cat3 == ErrorCategory.EMPTY_OR_UNPARSED

    def test_classify_arithmetic_error(self):
        cat, expl = classify_error(
            predicted="42",
            ground_truth="144",
        )
        assert cat == ErrorCategory.ARITHMETIC_ERROR
        assert "Numerical mismatch" in expl

    def test_classify_reasoning_gap(self):
        cat, _ = classify_error(
            predicted="The treaty was signed in Paris by regional delegates",
            ground_truth="The convention was approved in London by council representatives",
        )
        assert cat == ErrorCategory.REASONING_GAP

    def test_classify_other_fallback(self):
        cat, _ = classify_error(predicted="Tokyo", ground_truth="Kyoto")
        assert cat == ErrorCategory.OTHER_INCORRECT


# ─────────────────────────────────────────────────────────────────────────────
# 3. Tests for Record Diagnostics & Gate Error Rates
# ─────────────────────────────────────────────────────────────────────────────


class TestAnalyzeErrorsFromRecords:
    def test_empty_records(self):
        report = analyze_errors_from_records([])
        assert report.total_tasks_evaluated == 0
        assert report.total_errors == 0
        assert report.gate_errors.total_decisions == 0

    def test_synthetic_mixed_batch(self):
        records = [
            # 1. Correct STOP
            {
                "task_id": "t1",
                "method": "GateOrchestra",
                "predicted_answer": "Paris",
                "ground_truth": "Paris",
                "is_correct": True,
                "tokens_spent": 100,
                "gate_decision": {"decision": "STOP", "confidence": 0.99},
            },
            # 2. False STOP (error)
            {
                "task_id": "t2",
                "method": "GateOrchestra",
                "predicted_answer": "Unable to determine",
                "ground_truth": "Tokyo",
                "is_correct": False,
                "tokens_spent": 120,
                "gate_decision": {"decision": "STOP", "confidence": 0.95},
            },
            # 3. Correct ESCALATE
            {
                "task_id": "t3",
                "method": "GateOrchestra",
                "predicted_answer": "Euro",
                "ground_truth": "Euro",
                "is_correct": True,
                "tokens_spent": 400,
                "gate_decision": {
                    "decision": "ESCALATE",
                    "confidence": 0.88,
                    "token_budget_cap": 800,
                },
            },
            # 4. Failed ESCALATE (error)
            {
                "task_id": "t4",
                "method": "GateOrchestra",
                "predicted_answer": "Yen",
                "ground_truth": "Euro",
                "is_correct": False,
                "tokens_spent": 450,
                "gate_decision": {
                    "decision": "ESCALATE",
                    "confidence": 0.90,
                    "token_budget_cap": 800,
                },
            },
        ]

        report = analyze_errors_from_records(records, source_name="synthetic")
        assert report.total_tasks_evaluated == 4
        assert report.total_errors == 2
        assert report.overall_error_rate == 50.0

        # Gate breakdown
        g = report.gate_errors
        assert g.total_decisions == 4
        assert g.total_stop == 2
        assert g.total_escalate == 2
        assert g.correct_stop == 1
        assert g.false_stop == 1
        assert g.correct_escalate == 1
        assert g.failed_escalate == 1
        assert g.false_stop_rate == 50.0
        assert g.failed_escalate_rate == 50.0

        # Check recorded errors
        assert len(report.error_records) == 2
        categories = {e.category for e in report.error_records}
        assert categories == {ErrorCategory.FALSE_STOP, ErrorCategory.FALSE_ESCALATE_FAILED}

    def test_pydantic_eval_result_records(self):
        eval_records = [
            EvalResult(
                task_id="t1",
                method="GateOrchestra",
                predicted_answer="Wrong",
                is_correct=False,
                tokens_spent=150,
                probe_tokens=150,
                mas_tokens=0,
                gate_decision=GateDecision(
                    task_id="t1",
                    decision="STOP",
                    confidence=0.92,
                ),
            ),
        ]
        report = analyze_errors_from_records(eval_records)
        assert report.total_errors == 1
        assert report.gate_errors.false_stop == 1


# ─────────────────────────────────────────────────────────────────────────────
# 4. Tests for File Analysis (Existing Trace Data)
# ─────────────────────────────────────────────────────────────────────────────


class TestAnalyzeErrorsFromFile:
    def test_analyze_week2_traces(self):
        trace_path = Path("logs/week2_baseline_results.jsonl")
        if not trace_path.exists():
            pytest.skip("logs/week2_baseline_results.jsonl not present")

        report = analyze_errors_from_file(trace_path)
        assert report.total_tasks_evaluated == 210
        assert report.total_errors == 53
        assert abs(report.overall_error_rate - 25.24) < 0.1

        # Check gate errors
        assert report.gate_errors.total_decisions == 150
        assert report.gate_errors.false_stop == 20
        assert report.gate_errors.failed_escalate == 10

        # Check per-method error counts
        breakdown = report.agent_breakdowns
        assert breakdown["Always-MAS"].error_count == 18
        assert breakdown["GateOrchestra"].error_count == 15
        assert breakdown["RuleBasedGate"].error_count == 8
        assert breakdown["RandomGate"].error_count == 7
        assert breakdown["CoT-SC-only"].error_count == 5

        # Check category percentages sum to 100
        pct_sum = sum(report.category_percentages.values())
        assert abs(pct_sum - 100.0) < 0.05

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            analyze_errors_from_file("logs/non_existent_traces_file.jsonl")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Tests for CLI Main Entrypoint
# ─────────────────────────────────────────────────────────────────────────────


class TestAnalyzeErrorsCLI:
    def test_cli_default_run(self, capsys):
        code = cli_main(["--input", "logs/week2_baseline_results.jsonl"])
        assert code == 0
        captured = capsys.readouterr()
        assert "GateOrchestra -- Error Analysis Report" in captured.out
        assert "Gate Routing Error Breakdown" in captured.out
        assert "False STOP (Under-routing)" in captured.out

    def test_cli_with_dataset_ground_truth(self, capsys):
        code = cli_main(
            [
                "--input",
                "logs/week2_baseline_results.jsonl",
                "--load-dataset-gt",
            ]
        )
        assert code == 0
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "Loaded 150 ground truth answers" in combined
        assert "Uruguay River" in combined

    def test_cli_json_format(self, capsys):
        code = cli_main(
            [
                "--input",
                "logs/week2_baseline_results.jsonl",
                "--format",
                "json",
            ]
        )
        assert code == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out.strip())
        assert parsed["total_errors"] == 53
        assert "gate_errors" in parsed

    def test_cli_export_file(self, tmp_path):
        out_file = tmp_path / "errors.md"
        code = cli_main(
            [
                "--input",
                "logs/week2_baseline_results.jsonl",
                "--output",
                str(out_file),
            ]
        )
        assert code == 0
        assert out_file.exists()
        content = out_file.read_text(encoding="utf-8")
        assert "Total Errors Identified:" in content
        assert "53" in content

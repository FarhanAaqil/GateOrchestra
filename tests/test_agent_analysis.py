"""tests/test_agent_analysis.py
==============================
Unit and integration tests for Agent Performance Analysis (Person 2).

Covers:
  - AgentPerformanceMetrics and AgentAnalysisReport formatting (Markdown & JSON)
  - analyze_eval_records on synthetic and real evaluation records
  - analyze_trace_file on existing evaluation data (logs/week2_baseline_results.jsonl)
  - benchmark_agents profiling across ProbeAgent, MAS Orchestrator, and Sub-Agents
  - scripts/analyze_agents.py CLI entrypoint
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.agent_analysis import (
    AgentAnalysisReport,
    AgentPerformanceMetrics,
    analyze_eval_records,
    analyze_trace_file,
    benchmark_agents,
    exact_match,
)
from scripts.analyze_agents import main as cli_main
from shared.schemas import EvalResult, Task
from shared.token_logger import TokenAccountant

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures & Helpers
# ─────────────────────────────────────────────────────────────────────────────


def dummy_caller(prompt: str, temperature: float, budget: int) -> tuple[str, int]:
    """Simple deterministic mock LLM caller for testing."""
    p_lower = prompt.lower()
    if "paris" in p_lower or "france" in p_lower:
        return "Final Answer: Paris", 40
    if "tokyo" in p_lower:
        return "Final Answer: Tokyo", 35
    if "12" in p_lower:
        return "Final Answer: 144", 25
    return "Final Answer: Target", 30


@pytest.fixture
def sample_tasks() -> list[Task]:
    return [
        Task(
            task_id="t1",
            question="What is the capital of France?",
            ground_truth="Paris",
            depth_score=1,
            parallel_score=1,
        ),
        Task(
            task_id="t2",
            question="Compare Tokyo vs Kyoto in population.",
            ground_truth="Tokyo",
            depth_score=1,
            parallel_score=3,
        ),
        Task(
            task_id="t3",
            question="Multi-hop reasoning about 12 squared.",
            ground_truth="144",
            depth_score=4,
            parallel_score=1,
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Tests for Data Models & Formatting
# ─────────────────────────────────────────────────────────────────────────────


class TestAgentPerformanceMetrics:
    def test_metrics_defaults(self):
        m = AgentPerformanceMetrics(name="TestAgent")
        assert m.name == "TestAgent"
        assert m.execution_count == 0
        assert m.accuracy == 0.0
        assert m.total_tokens == 0
        assert m.avg_latency_ms is None

    def test_metrics_to_dict(self):
        m = AgentPerformanceMetrics(
            name="ReActAgent",
            execution_count=10,
            correct_count=8,
            accuracy=80.0,
            total_tokens=1500,
            avg_tokens=150.0,
            min_tokens=100,
            max_tokens=200,
            avg_latency_ms=12.5,
            strategy_counts={"react": 10},
            strategy_frequencies={"react": 100.0},
        )
        d = m.to_dict()
        assert d["name"] == "ReActAgent"
        assert d["accuracy"] == 80.0
        assert d["strategy_frequencies"]["react"] == 100.0

    def test_report_formatting(self):
        m1 = AgentPerformanceMetrics(
            name="ProbeAgent",
            execution_count=5,
            correct_count=4,
            accuracy=80.0,
            total_tokens=500,
            avg_tokens=100.0,
            avg_latency_ms=15.0,
        )
        report = AgentAnalysisReport(
            metrics={"ProbeAgent": m1},
            source_info="test_suite",
            total_tasks_analyzed=5,
        )

        md = report.to_markdown()
        assert "GateOrchestra -- Agent Performance Analysis Report" in md
        assert "ProbeAgent" in md
        assert "80.0%" in md
        assert "15.0" in md

        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed["total_tasks_analyzed"] == 5
        assert "ProbeAgent" in parsed["metrics"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tests for Record Analysis
# ─────────────────────────────────────────────────────────────────────────────


class TestAnalyzeEvalRecords:
    def test_analyze_empty_records(self):
        report = analyze_eval_records([])
        assert report.total_tasks_analyzed == 0
        assert len(report.metrics) == 0

    def test_exact_match_helper(self):
        assert exact_match("Paris", "paris") is True
        assert exact_match("  Tokyo.  ", "Tokyo") is True
        assert exact_match("Wrong", "Right") is False
        assert exact_match("", "Right") is False

    def test_analyze_synthetic_records(self):
        records = [
            {
                "method": "ReActAgent",
                "task_id": "t1",
                "predicted_answer": "Paris",
                "ground_truth": "Paris",
                "is_correct": True,
                "tokens_spent": 120,
                "latency_ms": 10.0,
                "mas_strategy": "react",
            },
            {
                "method": "ReActAgent",
                "task_id": "t2",
                "predicted_answer": "Wrong",
                "ground_truth": "Tokyo",
                "is_correct": False,
                "tokens_spent": 180,
                "latency_ms": 20.0,
                "mas_strategy": "react",
            },
            {
                "method": "DebateAgent",
                "task_id": "t1",
                "predicted_answer": "Paris",
                "ground_truth": "Paris",
                "is_correct": True,
                "tokens_spent": 300,
                "latency_ms": 35.0,
                "mas_strategy": "debate",
            },
        ]

        report = analyze_eval_records(records, source_name="synthetic")
        assert report.total_tasks_analyzed == 3
        assert "ReActAgent" in report.metrics
        assert "DebateAgent" in report.metrics

        react_m = report.metrics["ReActAgent"]
        assert react_m.execution_count == 2
        assert react_m.correct_count == 1
        assert react_m.accuracy == 50.0
        assert react_m.total_tokens == 300
        assert react_m.avg_tokens == 150.0
        assert react_m.avg_latency_ms == 15.0
        assert react_m.strategy_frequencies == {"react": 100.0}

        debate_m = report.metrics["DebateAgent"]
        assert debate_m.execution_count == 1
        assert debate_m.accuracy == 100.0
        assert debate_m.total_tokens == 300

    def test_analyze_pydantic_eval_results(self):
        eval_res = [
            EvalResult(
                task_id="t1",
                method="GateOrchestra",
                predicted_answer="42",
                is_correct=True,
                tokens_spent=150,
                probe_tokens=150,
                mas_tokens=0,
                latency_ms=5.5,
            ),
            EvalResult(
                task_id="t2",
                method="GateOrchestra",
                predicted_answer="99",
                is_correct=False,
                tokens_spent=250,
                probe_tokens=250,
                mas_tokens=0,
                latency_ms=8.5,
            ),
        ]
        report = analyze_eval_records(eval_res)
        m = report.metrics["GateOrchestra"]
        assert m.execution_count == 2
        assert m.accuracy == 50.0
        assert m.avg_latency_ms == 7.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Tests for Trace File Loading (Existing Data)
# ─────────────────────────────────────────────────────────────────────────────


class TestAnalyzeTraceFile:
    def test_analyze_existing_week2_results(self):
        trace_path = Path("logs/week2_baseline_results.jsonl")
        if not trace_path.exists():
            pytest.skip("logs/week2_baseline_results.jsonl not present")

        report = analyze_trace_file(trace_path)
        assert report.total_tasks_analyzed >= 210

        expected_methods = {
            "Always-MAS",
            "CoT-SC-only",
            "GateOrchestra",
            "RandomGate",
            "RuleBasedGate",
        }
        assert set(report.metrics.keys()) == expected_methods

        # Check GateOrchestra stats from week 2
        go_m = report.metrics["GateOrchestra"]
        assert go_m.execution_count >= 90
        assert go_m.accuracy > 70.0
        assert go_m.avg_tokens < 300.0

        # Check Always-MAS stats
        mas_m = report.metrics["Always-MAS"]
        assert mas_m.execution_count >= 30
        assert mas_m.avg_tokens > 500.0

    def test_missing_trace_file_raises(self):
        with pytest.raises(FileNotFoundError):
            analyze_trace_file("logs/non_existent_file_xyz.jsonl")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Tests for Live / Benchmark Agent Profiling
# ─────────────────────────────────────────────────────────────────────────────


class TestBenchmarkAgents:
    def test_benchmark_agents_suite(self, sample_tasks):
        accountant = TokenAccountant()
        report = benchmark_agents(
            tasks=sample_tasks,
            accountant=accountant,
            llm_caller=dummy_caller,
            probe_budget=200,
            include_subagents=True,
        )

        assert report.total_tasks_analyzed == len(sample_tasks) * 6  # 6 agents

        expected_agents = {
            "ProbeAgent (CoT-SC)",
            "MASOrchestrator (Auto)",
            "MASOrchestrator (Bandit)",
            "ReActAgent",
            "DebateAgent",
            "ReflexionAgent",
        }
        assert set(report.metrics.keys()) == expected_agents

        for _name, m in report.metrics.items():
            assert m.execution_count == len(sample_tasks)
            assert m.total_tokens > 0
            assert m.avg_latency_ms is not None
            assert m.avg_latency_ms >= 0.0

        # Check MASOrchestrator (Auto) strategy allocation
        auto_m = report.metrics["MASOrchestrator (Auto)"]
        assert "react" in auto_m.strategy_counts or "debate" in auto_m.strategy_counts
        # Percentage sum check
        pct_sum = sum(auto_m.strategy_frequencies.values())
        assert abs(pct_sum - 100.0) < 0.05

        # Check that TokenAccountant logged calls
        assert len(accountant.get_total_by_method()) >= 4


# ─────────────────────────────────────────────────────────────────────────────
# 5. Tests for CLI Main
# ─────────────────────────────────────────────────────────────────────────────


class TestAnalyzeAgentsCLI:
    @pytest.fixture
    def sample_trace_file(self, tmp_path):
        trace_path = Path("logs/week2_baseline_results.jsonl")
        if trace_path.exists():
            return str(trace_path)
        synth_file = tmp_path / "sample_trace.jsonl"
        records = [
            {
                "task_id": "t1",
                "method": "GateOrchestra",
                "predicted_answer": "42",
                "ground_truth": "42",
                "is_correct": True,
                "tokens_spent": 150,
                "latency_ms": 10.0,
            },
            {
                "task_id": "t2",
                "method": "Always-MAS",
                "predicted_answer": "42",
                "ground_truth": "42",
                "is_correct": True,
                "tokens_spent": 650,
                "latency_ms": 30.0,
            },
        ]
        with synth_file.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return str(synth_file)

    def test_cli_trace_mode(self, capsys, sample_trace_file):
        code = cli_main(["--input", sample_trace_file])
        assert code == 0
        captured = capsys.readouterr()
        assert "GateOrchestra -- Agent Performance Analysis Report" in captured.out
        assert "GateOrchestra" in captured.out
        assert "Always-MAS" in captured.out

    def test_cli_benchmark_mode_json(self, capsys):
        code = cli_main(["--benchmark", "--split", "test", "--n", "2", "--format", "json"])
        assert code == 0
        captured = capsys.readouterr()
        assert "ProbeAgent (CoT-SC)" in captured.out
        assert "MASOrchestrator (Auto)" in captured.out

    def test_cli_export_file(self, tmp_path, sample_trace_file):
        out_file = tmp_path / "agent_report.md"
        code = cli_main(
            [
                "--input",
                sample_trace_file,
                "--output",
                str(out_file),
            ]
        )
        assert code == 0
        assert out_file.exists()
        content = out_file.read_text(encoding="utf-8")
        assert "GateOrchestra -- Agent Performance Analysis Report" in content

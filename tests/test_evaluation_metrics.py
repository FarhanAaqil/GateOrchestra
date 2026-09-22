"""
tests/test_evaluation_metrics.py
=================================
Unit tests for evaluation and Pareto metrics calculations (Person 2 / Person 4).
"""

from __future__ import annotations

from evaluation.metrics import (
    compute_evaluation_metrics,
    compute_mas_strategy_allocation,
)
from shared.schemas import EvalResult, GateDecision


def _make_eval(
    task_id: str,
    decision: str = "ESCALATE",
    mas_strategy: str | None = "react",
    is_correct: bool = True,
    probe_tokens: int = 100,
    mas_tokens: int = 200,
) -> EvalResult:
    total = probe_tokens + mas_tokens
    gate_decision = None
    if decision in ("STOP", "ESCALATE"):
        cap = 400 if decision == "ESCALATE" else None
        gate_decision = GateDecision(
            task_id=task_id, decision=decision, confidence=0.9, token_budget_cap=cap
        )
    return EvalResult(
        task_id=task_id,
        method="GateOrchestra",
        predicted_answer="42",
        is_correct=is_correct,
        tokens_spent=total,
        probe_tokens=probe_tokens,
        mas_tokens=mas_tokens,
        gate_decision=gate_decision,
        mas_strategy=mas_strategy if decision == "ESCALATE" else None,
    )


class TestMASStrategyAllocationMetrics:
    def test_empty_results_returns_zeroes(self):
        alloc = compute_mas_strategy_allocation([])
        assert alloc["total_escalated"] == 0
        assert alloc["react_pct"] == 0.0
        assert alloc["debate_pct"] == 0.0
        assert alloc["reflexion_pct"] == 0.0

    def test_allocation_percentage_calculation(self):
        results = [
            _make_eval("t1", decision="ESCALATE", mas_strategy="react"),
            _make_eval("t2", decision="ESCALATE", mas_strategy="react"),
            _make_eval("t3", decision="ESCALATE", mas_strategy="debate"),
            _make_eval("t4", decision="ESCALATE", mas_strategy="reflexion"),
            _make_eval("t5", decision="STOP", mas_strategy=None),
        ]
        alloc = compute_mas_strategy_allocation(results)
        assert alloc["total_escalated"] == 4
        assert alloc["react_count"] == 2
        assert alloc["debate_count"] == 1
        assert alloc["reflexion_count"] == 1

        assert alloc["react_pct"] == 50.0
        assert alloc["debate_pct"] == 25.0
        assert alloc["reflexion_pct"] == 25.0


class TestEvaluationMetrics:
    def test_compute_evaluation_metrics_without_baseline(self):
        results = [
            _make_eval("t1", decision="ESCALATE", mas_strategy="react", is_correct=True),
            _make_eval("t2", decision="STOP", mas_strategy=None, is_correct=False, mas_tokens=0),
        ]
        metrics = compute_evaluation_metrics(results)
        assert metrics["n_tasks"] == 2
        assert metrics["accuracy"] == 50.0
        assert metrics["stop_count"] == 1
        assert metrics["escalate_count"] == 1
        assert metrics["stop_rate"] == 0.5
        assert metrics["escalate_rate"] == 0.5
        assert metrics["strategy_allocation"]["react_pct"] == 100.0
        assert metrics["token_savings_pct"] is None
        assert metrics["accuracy_delta"] is None

    def test_compute_evaluation_metrics_with_baseline(self):
        results = [
            _make_eval(
                "t1",
                decision="ESCALATE",
                mas_strategy="react",
                is_correct=True,
                probe_tokens=100,
                mas_tokens=100,
            ),
            _make_eval(
                "t2",
                decision="STOP",
                mas_strategy=None,
                is_correct=True,
                probe_tokens=100,
                mas_tokens=0,
            ),
        ]
        # Baseline spends 400 tokens per task, 100% accuracy
        baseline = [
            _make_eval(
                "t1",
                decision="ESCALATE",
                mas_strategy="debate",
                is_correct=True,
                probe_tokens=100,
                mas_tokens=300,
            ),
            _make_eval(
                "t2",
                decision="ESCALATE",
                mas_strategy="reflexion",
                is_correct=True,
                probe_tokens=100,
                mas_tokens=300,
            ),
        ]
        metrics = compute_evaluation_metrics(results, baseline_results=baseline)
        assert metrics["accuracy"] == 100.0
        assert metrics["accuracy_delta"] == 0.0
        assert metrics["total_tokens"] == 300
        assert metrics["token_savings_pct"] == 62.5

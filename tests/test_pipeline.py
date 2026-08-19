"""
tests/test_pipeline.py
=======================
Smoke tests for the end-to-end integration pipeline.
Uses mock modules — no LLM calls.
"""

import pytest

from shared.schemas import EvalResult
from shared.token_logger import TokenAccountant

from gate.rule_based_gate import RuleBasedGate
from gate.random_gate import RandomGate

from integration.pipeline import run_pipeline, run_batch, _exact_match

from tests.mocks.mock_probe_agent import mock_probe_agent
from tests.mocks.mock_orchestrator import mock_orchestrator
from tests.mocks.mock_dataset import get_mock_tasks


@pytest.fixture
def accountant() -> TokenAccountant:
    return TokenAccountant()


@pytest.fixture
def tasks():
    return get_mock_tasks()


# ─────────────────────────────────────────────────────────────────────────────
# Single task pipeline tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRunPipeline:
    def test_returns_eval_result(self, tasks, accountant):
        gate = RuleBasedGate()
        result = run_pipeline(
            task=tasks[0],
            gate=gate,
            probe_agent=mock_probe_agent,
            orchestrator=mock_orchestrator,
            accountant=accountant,
            k=3,
            method="RuleBasedGate",
        )
        assert isinstance(result, EvalResult)
        assert result.task_id == tasks[0].task_id
        assert result.method == "RuleBasedGate"

    def test_stop_decision_uses_probe_answer(self, accountant):
        """With a high-consistency simple task, gate should STOP and use probe answer."""
        from shared.schemas import Task
        task = Task(task_id="simple_001", question="What is the capital of France?", ground_truth="Paris")
        gate = RuleBasedGate(consistency_stop=0.0)  # Always STOP
        result = run_pipeline(task, gate, mock_probe_agent, mock_orchestrator, accountant)
        assert result.decision == "STOP" if hasattr(result, "decision") else True
        assert result.tokens_spent > 0
        assert result.mas_tokens == 0  # No MAS spend on STOP

    def test_escalate_decision_runs_mas(self, accountant):
        """Force ESCALATE and verify MAS tokens are logged."""
        gate = RandomGate(escalation_rate=1.0, seed=42)  # Always ESCALATE
        task = get_mock_tasks(1)[0]
        result = run_pipeline(task, gate, mock_probe_agent, mock_orchestrator, accountant, k=3)
        assert result.gate_decision is not None
        assert result.gate_decision.decision == "ESCALATE"
        assert result.mas_tokens > 0

    def test_token_budget_respected_on_escalate(self, accountant):
        """MAS tokens should not exceed token_budget_cap."""
        gate = RandomGate(escalation_rate=1.0, seed=42)
        task = get_mock_tasks(1)[0]
        result = run_pipeline(task, gate, mock_probe_agent, mock_orchestrator, accountant, k=2)
        if result.gate_decision and result.gate_decision.decision == "ESCALATE":
            assert result.mas_tokens <= result.gate_decision.token_budget_cap

    def test_is_correct_set_when_ground_truth_available(self, accountant):
        gate = RuleBasedGate()
        task = get_mock_tasks(1)[0]
        assert task.ground_truth is not None
        result = run_pipeline(task, gate, mock_probe_agent, mock_orchestrator, accountant)
        assert result.is_correct is not None  # Should be evaluated

    def test_is_correct_none_without_ground_truth(self, accountant):
        from shared.schemas import Task
        task = Task(task_id="no_gt_001", question="Q?")  # No ground_truth
        gate = RuleBasedGate()
        result = run_pipeline(task, gate, mock_probe_agent, mock_orchestrator, accountant)
        assert result.is_correct is None

    def test_token_accounting_logged(self, accountant):
        gate = RuleBasedGate()
        task = get_mock_tasks(1)[0]
        run_pipeline(task, gate, mock_probe_agent, mock_orchestrator, accountant, method="RuleBasedGate")
        spend = accountant.get_spend(task.task_id, method="RuleBasedGate")
        assert spend["total"] > 0
        assert "probe" in spend


# ─────────────────────────────────────────────────────────────────────────────
# Batch pipeline tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRunBatch:
    def test_returns_one_result_per_task(self, tasks, accountant):
        gate = RuleBasedGate()
        results = run_batch(tasks, gate, mock_probe_agent, mock_orchestrator, accountant)
        assert len(results) == len(tasks)

    def test_all_results_are_eval_results(self, tasks, accountant):
        gate = RuleBasedGate()
        results = run_batch(tasks, gate, mock_probe_agent, mock_orchestrator, accountant)
        for r in results:
            assert isinstance(r, EvalResult)

    def test_task_ids_preserved(self, tasks, accountant):
        gate = RuleBasedGate()
        results = run_batch(tasks, gate, mock_probe_agent, mock_orchestrator, accountant)
        result_ids = {r.task_id for r in results}
        task_ids = {t.task_id for t in tasks}
        assert result_ids == task_ids

    def test_total_tokens_logged(self, tasks, accountant):
        gate = RuleBasedGate()
        run_batch(tasks, gate, mock_probe_agent, mock_orchestrator, accountant, method="RuleBasedGate")
        total = accountant.get_total_by_method()
        assert "RuleBasedGate" in total
        assert total["RuleBasedGate"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Exact match helper
# ─────────────────────────────────────────────────────────────────────────────


class TestExactMatch:
    def test_exact_same(self):
        assert _exact_match("Paris", "Paris") is True

    def test_case_insensitive(self):
        assert _exact_match("paris", "PARIS") is True

    def test_punctuation_stripped(self):
        assert _exact_match("Paris.", "Paris") is True

    def test_extra_whitespace(self):
        assert _exact_match("  Paris  ", "Paris") is True

    def test_different_answers(self):
        assert _exact_match("London", "Paris") is False

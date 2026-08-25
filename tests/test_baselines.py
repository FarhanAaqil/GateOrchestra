"""
tests/test_baselines.py
=======================
Unit tests for CoT-SC-only and Always-MAS baselines (Person 2).
"""

import pytest

from agents.baselines import (
    run_always_mas_baseline,
    run_always_mas_batch,
    run_cot_sc_baseline,
    run_cot_sc_batch,
)
from shared.schemas import EvalResult, ProbeResult, Task
from shared.token_logger import TokenAccountant


@pytest.fixture
def sample_task() -> Task:
    return Task(
        task_id="base_001",
        question="What is 10 + 20?",
        ground_truth="30",
        depth_score=1,
        parallel_score=1,
    )


@pytest.fixture
def sample_tasks() -> list[Task]:
    return [
        Task(task_id="base_batch_01", question="What is 1+1?", ground_truth="2"),
        Task(task_id="base_batch_02", question="What is 2*3?", ground_truth="6"),
    ]


class TestCotScBaseline:
    def test_run_cot_sc_baseline(self, sample_task):
        accountant = TokenAccountant()

        def mock_probe(t: Task) -> ProbeResult:
            return ProbeResult(
                task_id=t.task_id,
                answer="30",
                consistency_score=1.0,
                tokens_used=120,
                raw_outputs=["30", "30", "30", "30", "30"],
            )

        result = run_cot_sc_baseline(sample_task, probe_fn=mock_probe, accountant=accountant)

        assert isinstance(result, EvalResult)
        assert result.task_id == "base_001"
        assert result.method == "CoT-SC-only"
        assert result.predicted_answer == "30"
        assert result.is_correct is True
        assert result.tokens_spent == 120
        assert result.probe_tokens == 120
        assert result.mas_tokens is None
        assert accountant.get_spend("base_001")["probe"] == 120

    def test_run_cot_sc_batch(self, sample_tasks):
        def mock_probe(t: Task) -> ProbeResult:
            return ProbeResult(
                task_id=t.task_id,
                answer=t.ground_truth or "0",
                consistency_score=0.8,
                tokens_used=100,
                raw_outputs=["ans"] * 5,
            )

        results = run_cot_sc_batch(sample_tasks, probe_fn=mock_probe)
        assert len(results) == 2
        assert all(r.method == "CoT-SC-only" for r in results)
        assert all(r.is_correct is True for r in results)


class TestAlwaysMasBaseline:
    def test_run_always_mas_baseline(self, sample_task):
        accountant = TokenAccountant()

        def mock_mas(t: Task, budget: int) -> tuple[str, int]:
            return "30", 450

        result = run_always_mas_baseline(
            sample_task,
            orchestrator_fn=mock_mas,
            accountant=accountant,
            token_budget=1000,
        )

        assert isinstance(result, EvalResult)
        assert result.task_id == "base_001"
        assert result.method == "Always-MAS"
        assert result.predicted_answer == "30"
        assert result.is_correct is True
        assert result.tokens_spent == 450
        assert result.probe_tokens is None
        assert result.mas_tokens == 450
        assert accountant.get_spend("base_001")["mas"] == 450

    def test_run_always_mas_batch(self, sample_tasks):
        def mock_mas(t: Task, budget: int) -> tuple[str, int]:
            return t.ground_truth or "ans", 300

        results = run_always_mas_batch(sample_tasks, orchestrator_fn=mock_mas)
        assert len(results) == 2
        assert all(r.method == "Always-MAS" for r in results)
        assert all(r.mas_tokens == 300 for r in results)

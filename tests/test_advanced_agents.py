"""
tests/test_advanced_agents.py
=============================
Unit tests for advanced research components in Person 2:
  - Sequential Early-Exit SPRT in ProbeAgent
  - Semantic Soft Majority Voting
  - Contextual Multi-Armed Bandit (LinUCB) Sub-Agent Router
  - Bandit reward update and strategy adaptation
"""

import pytest
import numpy as np

from agents.probe_agent import ProbeAgent, extract_answer, normalize_answer
from agents.orchestrator import LinUCBRouter, MASOrchestrator
from shared.schemas import Task, ProbeResult


@pytest.fixture
def sample_task() -> Task:
    return Task(
        task_id="adv_task_01",
        question="What is the square root of 144?",
        ground_truth="12",
        depth_score=2,
        parallel_score=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Test Sequential Early-Exit in ProbeAgent
# ─────────────────────────────────────────────────────────────────────────────


class TestEarlyExitProbe:
    def test_early_exit_stops_at_3_samples_on_unanimity(self, sample_task):
        call_count = 0

        def unanimous_caller(prompt: str, temp: float, budget: int) -> tuple[str, int]:
            nonlocal call_count
            call_count += 1
            return "Final Answer: 12", 20

        # With early_exit=True, 5 max samples should stop at 3 samples
        agent = ProbeAgent(n_samples=5, early_exit=True, llm_caller=unanimous_caller)
        res = agent.run(sample_task)

        assert isinstance(res, ProbeResult)
        assert res.answer == "12"
        assert res.consistency_score == 1.0
        assert len(res.raw_outputs) == 3  # Only 3 samples generated instead of 5!
        assert call_count == 3
        assert res.tokens_used == 60

    def test_no_early_exit_when_split(self, sample_task):
        call_count = 0
        responses = ["Final Answer: 12", "Final Answer: 10", "Final Answer: 12", "Final Answer: 12", "Final Answer: 12"]

        def split_caller(prompt: str, temp: float, budget: int) -> tuple[str, int]:
            nonlocal call_count
            resp = responses[call_count % len(responses)]
            call_count += 1
            return resp, 20

        agent = ProbeAgent(n_samples=5, early_exit=True, llm_caller=split_caller)
        res = agent.run(sample_task)

        # Because sample 2 disagreed, early exit was not triggered at sample 3
        assert len(res.raw_outputs) == 5
        assert call_count == 5


# ─────────────────────────────────────────────────────────────────────────────
# 2. Test Semantic Soft Majority Voting
# ─────────────────────────────────────────────────────────────────────────────


class TestSemanticSoftVoting:
    def test_fuzzy_semantic_grouping(self):
        answers = [
            "Paris, France",
            "Paris",
            "paris",
            "London",
            "Rome",
        ]
        majority, score = ProbeAgent._majority_vote(answers)
        assert "paris" in majority.lower()
        # Paris variants group together (3/5 = 0.6)
        assert score == pytest.approx(0.6)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Test LinUCB Contextual Bandit Router
# ─────────────────────────────────────────────────────────────────────────────


class TestLinUCBRouter:
    def test_feature_extraction(self, sample_task):
        router = LinUCBRouter(arms=["react", "debate", "reflexion"])
        feat = router.extract_context_features(sample_task)
        assert isinstance(feat, np.ndarray)
        assert feat.shape == (6, 1)
        assert feat[0, 0] == 1.0  # intercept

    def test_arm_selection_and_update(self, sample_task):
        router = LinUCBRouter(arms=["react", "debate", "reflexion"], alpha=0.1)
        initial_arm = router.select_arm(sample_task)
        assert initial_arm in ["react", "debate", "reflexion"]

        # Provide high reward for 'react'
        for _ in range(5):
            router.update(sample_task, "react", reward=1.0)
            router.update(sample_task, "debate", reward=0.0)

        # 'react' should now have highest UCB score
        chosen_arm = router.select_arm(sample_task)
        assert chosen_arm == "react"

    def test_router_persistence(self, tmp_path, sample_task):
        save_file = tmp_path / "linucb_router.json"
        router = LinUCBRouter(alpha=0.3)
        router.update(sample_task, "debate", reward=1.0)
        router.save(save_file)

        loaded_router = LinUCBRouter(alpha=0.3)
        loaded_router.load(save_file)
        assert loaded_router.alpha == 0.3
        assert np.allclose(loaded_router.b["debate"], router.b["debate"])


# ─────────────────────────────────────────────────────────────────────────────
# 4. Test Orchestrator with Bandit Routing
# ─────────────────────────────────────────────────────────────────────────────


class TestOrchestratorBanditMode:
    def test_orchestrator_runs_bandit_mode(self, sample_task):
        def mock_caller(prompt: str, temp: float, budget: int) -> tuple[str, int]:
            return "Final Answer: 12", 25

        orch = MASOrchestrator(default_strategy="bandit", llm_caller=mock_caller)
        ans, tokens = orch.run(sample_task, token_budget=100)
        assert ans == "12"
        assert tokens > 0

        # Update bandit reward
        orch.update_bandit_reward(sample_task, chosen_strategy="reflexion", is_correct=True, tokens_spent=tokens, budget=100)

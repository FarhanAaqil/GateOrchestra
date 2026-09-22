"""
tests/test_advanced_agents.py
=============================
Unit tests for advanced research components in Person 2:
  - Sequential Early-Exit SPRT in ProbeAgent
  - Semantic Soft Majority Voting
  - Contextual Multi-Armed Bandit (LinUCB) Sub-Agent Router
  - Bandit reward update and strategy adaptation
"""

import numpy as np
import pytest

from agents.orchestrator import LinUCBRouter, MASOrchestrator
from agents.probe_agent import ProbeAgent
from shared.schemas import ProbeResult, Task


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
        responses = [
            "Final Answer: 12",
            "Final Answer: 10",
            "Final Answer: 12",
            "Final Answer: 12",
            "Final Answer: 12",
        ]

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

    def test_save_load_roundtrip(self, tmp_path, sample_task):
        """Save and load roundtrip restores exact weights and matrices."""
        state_file = tmp_path / "bandit_state.json"
        router = LinUCBRouter(alpha=0.25)
        router.update(sample_task, "react", reward=0.8)

        router.save(state_file)
        assert state_file.exists()

        loaded_router = LinUCBRouter()
        loaded_router.load(state_file)

        assert loaded_router.alpha == 0.25
        assert np.allclose(loaded_router.A["react"], router.A["react"])
        assert np.allclose(loaded_router.b["react"], router.b["react"])

    def test_persisted_state_changes_after_update(self, tmp_path, sample_task):
        """Persisted JSON state before update != state after update."""
        state_file = tmp_path / "bandit_state.json"
        router = LinUCBRouter(alpha=0.5)

        # Save initial state
        router.save(state_file)
        initial_json = state_file.read_text(encoding="utf-8")

        # Perform update with positive reward
        router.update(sample_task, "debate", reward=1.0)
        router.save(state_file)
        updated_json = state_file.read_text(encoding="utf-8")

        assert initial_json != updated_json, "Persisted file content must change after an update"

    def test_missing_state_file_handled_safely(self, tmp_path):
        """Loading a non-existent state file must log a warning and not raise an exception."""
        non_existent_file = tmp_path / "subfolder" / "missing_bandit.json"
        router = LinUCBRouter(alpha=0.5)

        # Should not raise FileNotFoundError
        router.load(non_existent_file)
        assert router.alpha == 0.5
        assert np.allclose(router.b["react"], np.zeros((6, 1)))

    def test_warmstart_from_tuples(self, sample_task):
        """Warm-starting from a list of (task, arm, reward) tuples updates matrices A and b."""
        router = LinUCBRouter(alpha=0.5)
        initial_b = float(router.b["react"].sum())

        count = router.warmstart_from_traces([(sample_task, "react", 0.95)])
        assert count == 1
        assert float(router.b["react"].sum()) != initial_b

    def test_warmstart_from_dict_records(self, sample_task):
        """Warm-starting from dict trace records calculates reward and updates state."""
        router = LinUCBRouter(alpha=0.5)
        initial_b = float(router.b["debate"].sum())

        record = {
            "task": sample_task,
            "mas_strategy": "debate",
            "is_correct": True,
            "tokens_spent": 100,
            "budget": 400,
        }
        count = router.warmstart_from_traces([record])
        assert count == 1
        assert float(router.b["debate"].sum()) != initial_b

    def test_save_is_atomic_no_tmp_file_left(self, tmp_path, sample_task):
        """After a successful save, the .tmp sidecar must NOT remain on disk."""
        state_file = tmp_path / "bandit_state.json"
        router = LinUCBRouter(alpha=0.4)
        router.update(sample_task, "react", reward=0.7)

        router.save(state_file)

        tmp_file = state_file.with_suffix(".tmp")
        assert state_file.exists(), "Final state file must exist after save"
        assert not tmp_file.exists(), ".tmp sidecar must be removed after atomic rename"

    def test_save_creates_parent_dirs(self, tmp_path, sample_task):
        """save() must create any missing parent directories automatically."""
        nested_path = tmp_path / "deep" / "nested" / "bandit.json"
        router = LinUCBRouter()
        router.save(nested_path)  # should not raise

        assert nested_path.exists()

    def test_save_atomic_preserves_old_state_on_read_error(self, tmp_path, sample_task):
        """If the old state was valid, after an atomic save the loaded router must match."""
        state_file = tmp_path / "atomic_test.json"
        router = LinUCBRouter(alpha=0.9)
        router.update(sample_task, "debate", reward=0.6)
        router.save(state_file)

        # Overwrite atomically with new state
        router.update(sample_task, "reflexion", reward=1.0)
        router.save(state_file)

        reloaded = LinUCBRouter()
        reloaded.load(state_file)
        assert reloaded.alpha == pytest.approx(0.9)
        assert np.allclose(reloaded.b["debate"], router.b["debate"])
        assert np.allclose(reloaded.b["reflexion"], router.b["reflexion"])

    def test_warmstart_from_json_file(self, tmp_path, sample_task):
        """Warm-starting from a JSON trace file correctly initializes router state."""
        import json

        trace_file = tmp_path / "warmstart_traces.json"
        traces_data = {
            "traces": [
                {
                    "task_id": sample_task.task_id,
                    "question": sample_task.question,
                    "ground_truth": sample_task.ground_truth,
                    "mas_strategy": "reflexion",
                    "is_correct": True,
                    "tokens_spent": 80,
                    "budget": 300,
                }
            ]
        }
        trace_file.write_text(json.dumps(traces_data), encoding="utf-8")

        router = LinUCBRouter(alpha=0.5)
        count = router.warmstart_from_traces(trace_file)
        assert count == 1
        assert float(router.b["reflexion"].sum()) > 0


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
        orch.update_bandit_reward(
            sample_task,
            chosen_strategy="reflexion",
            is_correct=True,
            tokens_spent=tokens,
            budget=100,
        )

    def test_orchestrator_save_load_methods(self, tmp_path, sample_task):
        """MASOrchestrator save_bandit_state and load_bandit_state delegate properly."""
        state_file = tmp_path / "orch_bandit.json"
        orch = MASOrchestrator(default_strategy="bandit")
        orch.update_bandit_reward(
            sample_task, "reflexion", is_correct=True, tokens_spent=50, budget=200
        )

        orch.save_bandit_state(state_file)
        assert state_file.exists()

        new_orch = MASOrchestrator(default_strategy="bandit", bandit_state_path=state_file)
        assert np.allclose(new_orch.bandit_router.b["reflexion"], orch.bandit_router.b["reflexion"])

    def test_orchestrator_warmstart_bandit(self, sample_task):
        """MASOrchestrator.warmstart_bandit delegates to bandit_router."""
        orch = MASOrchestrator(default_strategy="bandit")
        initial_b = float(orch.bandit_router.b["react"].sum())

        count = orch.warmstart_bandit([(sample_task, "react", 0.85)])
        assert count == 1
        assert float(orch.bandit_router.b["react"].sum()) != initial_b

"""
tests/test_week9_integration.py
================================
Week 9 Phase 5 - Final integration tests.

Covers end-to-end scenarios exercising multiple Week 9 deliverables together:

  1. run_batch + MASOrchestrator
       - strategy recorded per result
       - LinUCB b-vectors change after batch
  2. LinUCB warm-start -> run_batch
       - bandit is pre-seeded from traces before the batch runs
  3. Strategy allocation metrics computed from real batch output
  4. Atomic save / load round-trip after a batch run
       - no .tmp sidecar left on disk
       - reloaded router matches in-memory weights

No Groq/Ollama calls - all LLM calls use a lightweight mock.
"""

from __future__ import annotations

import json

import numpy as np

from agents.orchestrator import LinUCBRouter, MASOrchestrator  # noqa: F401
from evaluation.metrics import compute_evaluation_metrics, compute_mas_strategy_allocation
from gate.random_gate import RandomGate
from integration.pipeline import run_batch
from shared.schemas import EvalResult, Task
from shared.token_logger import TokenAccountant
from tests.mocks.mock_dataset import get_mock_tasks
from tests.mocks.mock_probe_agent import mock_probe_agent

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_orch(answer: str = "42", tokens: int = 30) -> MASOrchestrator:
    """Return a MASOrchestrator backed by a deterministic mock LLM."""

    def _caller(prompt: str, temp: float, budget: int) -> tuple[str, int]:
        return f"Final Answer: {answer}", min(tokens, budget)

    return MASOrchestrator(default_strategy="bandit", llm_caller=_caller)


def _make_tasks(n: int = 6) -> list[Task]:
    return get_mock_tasks(n)


def _run_batch(
    orch: MASOrchestrator,
    tasks: list[Task],
    gate: RandomGate,
    k: int = 3,
    method: str = "GateOrchestra",
) -> list[EvalResult]:
    return run_batch(
        tasks,
        gate,
        mock_probe_agent,
        orch.run,
        TokenAccountant(),
        k=k,
        method=method,
        mas_orchestrator=orch,
    )


# ---------------------------------------------------------------------------
# 1. run_batch + MASOrchestrator
# ---------------------------------------------------------------------------


class TestRunBatchWithMAS:
    """run_batch records mas_strategy per result and updates LinUCB on ESCALATE."""

    def test_all_results_have_strategy_on_force_escalate(self):
        """Every ESCALATE result must carry a valid mas_strategy."""
        orch = _make_orch()
        tasks = _make_tasks(4)
        gate = RandomGate(escalation_rate=1.0, seed=7)

        results = _run_batch(orch, tasks, gate)

        assert len(results) == 4
        for r in results:
            assert r.gate_decision is not None
            assert r.gate_decision.decision == "ESCALATE"
            assert r.mas_strategy in (
                "react",
                "debate",
                "reflexion",
            ), f"Expected valid strategy, got {r.mas_strategy!r}"

    def test_linucb_b_vectors_change_after_batch_escalate(self):
        """After a forced-ESCALATE batch the sum of b-vectors must differ from initial."""
        orch = _make_orch()
        tasks = _make_tasks(4)
        gate = RandomGate(escalation_rate=1.0, seed=3)
        initial_b_sum = sum(float(orch.bandit_router.b[a].sum()) for a in orch.bandit_router.arms)

        _run_batch(orch, tasks, gate)

        final_b_sum = sum(float(orch.bandit_router.b[a].sum()) for a in orch.bandit_router.arms)
        assert (
            final_b_sum != initial_b_sum
        ), "LinUCB b-vectors must change after ESCALATE batch; they did not."

    def test_stop_results_have_none_strategy_and_zero_mas_tokens(self):
        """Forced-STOP batch: every result must have mas_strategy=None and mas_tokens=0."""
        orch = _make_orch()
        tasks = _make_tasks(4)
        gate = RandomGate(escalation_rate=0.0, seed=0)

        results = _run_batch(orch, tasks, gate)

        for r in results:
            assert r.gate_decision is not None
            assert r.gate_decision.decision == "STOP"
            assert (
                r.mas_strategy is None
            ), f"STOP result must have mas_strategy=None, got {r.mas_strategy!r}"
            assert r.mas_tokens == 0, f"STOP result must have mas_tokens=0, got {r.mas_tokens}"

    def test_linucb_not_updated_on_stop_batch(self):
        """After a forced-STOP batch the bandit b-vectors must be unchanged."""
        orch = _make_orch()
        tasks = _make_tasks(4)
        gate = RandomGate(escalation_rate=0.0, seed=0)
        before = {arm: orch.bandit_router.b[arm].copy() for arm in orch.bandit_router.arms}

        _run_batch(orch, tasks, gate)

        for arm in orch.bandit_router.arms:
            assert np.allclose(
                before[arm], orch.bandit_router.b[arm]
            ), f"b-vector for arm={arm!r} changed on STOP batch - should not have."

    def test_mixed_gate_batch_token_accounting(self):
        """Mixed STOP/ESCALATE batch: logged tokens must be > 0 for the method label."""
        orch = _make_orch()
        tasks = _make_tasks(6)
        gate = RandomGate(escalation_rate=0.5, seed=99)
        accountant = TokenAccountant()

        run_batch(
            tasks,
            gate,
            mock_probe_agent,
            orch.run,
            accountant,
            k=3,
            method="GateOrchestra",
            mas_orchestrator=orch,
        )

        total = accountant.get_total_by_method()
        assert "GateOrchestra" in total
        assert total["GateOrchestra"] > 0


# ---------------------------------------------------------------------------
# 2. LinUCB warm-start -> run_batch
# ---------------------------------------------------------------------------


class TestWarmStartThenBatch:
    """Seeding the bandit from historical traces before running a batch."""

    def test_warmstart_from_tuple_traces_changes_b_before_batch(self):
        """b-vectors differ from cold-start after warm-starting from tuples."""
        sample_task = get_mock_tasks(1)[0]

        cold = MASOrchestrator(default_strategy="bandit")
        warm = MASOrchestrator(default_strategy="bandit")

        traces: list[tuple[Task, str, float]] = [
            (sample_task, "react", 0.9),
            (sample_task, "debate", 0.2),
            (sample_task, "react", 0.8),
        ]
        count = warm.warmstart_bandit(traces)
        assert count == 3
        assert not np.allclose(
            cold.bandit_router.b["react"], warm.bandit_router.b["react"]
        ), "Warm-started react b-vector must differ from cold-start."

    def test_warmstart_from_json_file(self, tmp_path):
        """Warm-starting from a JSON trace file correctly seeds the bandit."""
        sample_task = get_mock_tasks(1)[0]
        trace_file = tmp_path / "warmstart.json"
        traces_data = {
            "traces": [
                {
                    "task_id": sample_task.task_id,
                    "question": sample_task.question,
                    "ground_truth": sample_task.ground_truth,
                    "mas_strategy": "reflexion",
                    "is_correct": True,
                    "tokens_spent": 60,
                    "budget": 200,
                }
            ]
        }
        trace_file.write_text(json.dumps(traces_data), encoding="utf-8")

        orch = MASOrchestrator(default_strategy="bandit", warmstart_traces=trace_file)
        assert float(orch.bandit_router.b["reflexion"].sum()) > 0.0

    def test_warmstart_then_batch_selects_valid_strategies(self):
        """After warm-start the bandit still selects valid arms during a batch."""
        sample_task = get_mock_tasks(1)[0]
        traces: list[tuple[Task, str, float]] = [(sample_task, "react", 1.0)] * 5

        orch = _make_orch()
        orch.warmstart_bandit(traces)

        tasks = _make_tasks(3)
        gate = RandomGate(escalation_rate=1.0, seed=5)
        results = _run_batch(orch, tasks, gate)

        for r in results:
            assert r.mas_strategy in ("react", "debate", "reflexion")

    def test_warmstart_missing_file_is_safe(self, tmp_path):
        """Constructing with a missing warmstart file must not raise."""
        missing = tmp_path / "nonexistent_traces.json"
        orch = MASOrchestrator(default_strategy="bandit", warmstart_traces=missing)
        # Router must still be initialised with identity matrices
        assert np.allclose(orch.bandit_router.A["react"], np.eye(6))

    def test_warmstart_dict_records_reward_calculation(self):
        """Dict trace records with is_correct/tokens_spent compute reward correctly."""
        sample_task = get_mock_tasks(1)[0]
        orch = _make_orch()
        initial_b = float(orch.bandit_router.b["debate"].sum())

        record = {
            "task": sample_task,
            "mas_strategy": "debate",
            "is_correct": True,
            "tokens_spent": 100,
            "budget": 400,
        }
        count = orch.warmstart_bandit([record])
        assert count == 1
        assert float(orch.bandit_router.b["debate"].sum()) != initial_b


# ---------------------------------------------------------------------------
# 3. Strategy allocation metrics from batch output
# ---------------------------------------------------------------------------


class TestStrategyMetricsFromBatch:
    """compute_mas_strategy_allocation / compute_evaluation_metrics on real batch output."""

    def _escalate_batch(self, n: int = 6) -> list[EvalResult]:
        orch = _make_orch(answer="42")
        tasks = _make_tasks(n)
        gate = RandomGate(escalation_rate=1.0, seed=11)
        return _run_batch(orch, tasks, gate)

    def test_strategy_allocation_sums_to_100(self):
        results = self._escalate_batch(6)
        alloc = compute_mas_strategy_allocation(results)
        total_pct = alloc["react_pct"] + alloc["debate_pct"] + alloc["reflexion_pct"]
        assert (
            abs(total_pct - 100.0) < 0.05
        ), f"Strategy percentages must sum to ~100.0 (within 0.05), got {total_pct:.6f}"

    def test_strategy_allocation_counts_match_total_escalated(self):
        results = self._escalate_batch(6)
        alloc = compute_mas_strategy_allocation(results)
        count_total = alloc["react_count"] + alloc["debate_count"] + alloc["reflexion_count"]
        assert count_total == alloc["total_escalated"]
        assert alloc["total_escalated"] == 6

    def test_evaluation_metrics_includes_strategy_allocation(self):
        results = self._escalate_batch(4)
        metrics = compute_evaluation_metrics(results)
        assert "strategy_allocation" in metrics
        alloc = metrics["strategy_allocation"]
        assert "react_pct" in alloc
        assert "debate_pct" in alloc
        assert "reflexion_pct" in alloc
        assert alloc["total_escalated"] == 4

    def test_evaluation_metrics_escalate_rate_is_one_on_force_escalate(self):
        results = self._escalate_batch(5)
        metrics = compute_evaluation_metrics(results)
        assert metrics["escalate_rate"] == 1.0
        assert metrics["stop_rate"] == 0.0
        assert metrics["n_tasks"] == 5

    def test_evaluation_metrics_stop_only_batch(self):
        orch = _make_orch()
        tasks = _make_tasks(4)
        gate = RandomGate(escalation_rate=0.0, seed=0)
        results = _run_batch(orch, tasks, gate)
        metrics = compute_evaluation_metrics(results)
        assert metrics["stop_rate"] == 1.0
        assert metrics["escalate_rate"] == 0.0
        assert metrics["strategy_allocation"]["total_escalated"] == 0
        assert metrics["strategy_allocation"]["react_pct"] == 0.0

    def test_evaluation_metrics_token_savings_vs_baseline(self):
        """Gate-based (mixed) results should show token savings over always-escalate."""
        orch = _make_orch()
        tasks = _make_tasks(6)

        # Our method: mixed gate (~50% STOP = fewer MAS tokens)
        mixed_gate = RandomGate(escalation_rate=0.5, seed=42)
        results = _run_batch(orch, tasks, mixed_gate)

        # Baseline: always escalate
        baseline_orch = _make_orch()
        force_gate = RandomGate(escalation_rate=1.0, seed=42)
        baseline = _run_batch(baseline_orch, tasks, force_gate)

        metrics = compute_evaluation_metrics(results, baseline_results=baseline)
        assert metrics["token_savings_pct"] is not None
        assert isinstance(metrics["token_savings_pct"], float)


# ---------------------------------------------------------------------------
# 4. Atomic state save / load after a batch
# ---------------------------------------------------------------------------


class TestAtomicSaveAfterBatch:
    """State saved after a batch is atomic; no .tmp sidecar remains."""

    def test_no_tmp_sidecar_after_batch_save(self, tmp_path):
        """The .tmp file must not exist after a successful save."""
        orch = _make_orch()
        tasks = _make_tasks(3)
        gate = RandomGate(escalation_rate=1.0, seed=1)
        _run_batch(orch, tasks, gate)

        state_file = tmp_path / "bandit_state.json"
        orch.save_bandit_state(state_file)

        assert state_file.exists(), "State JSON must exist after save."
        assert not state_file.with_suffix(
            ".tmp"
        ).exists(), ".tmp sidecar must be removed by atomic rename."

    def test_save_load_roundtrip_after_batch(self, tmp_path):
        """Router weights are faithfully restored from disk after a batch run."""
        orch = _make_orch()
        tasks = _make_tasks(3)
        gate = RandomGate(escalation_rate=1.0, seed=2)
        _run_batch(orch, tasks, gate)

        state_file = tmp_path / "batch_state.json"
        orch.save_bandit_state(state_file)

        restored = MASOrchestrator(default_strategy="bandit", bandit_state_path=state_file)

        for arm in orch.bandit_router.arms:
            assert np.allclose(
                orch.bandit_router.A[arm], restored.bandit_router.A[arm]
            ), f"A matrix for arm={arm!r} did not survive save/load."
            assert np.allclose(
                orch.bandit_router.b[arm], restored.bandit_router.b[arm]
            ), f"b vector for arm={arm!r} did not survive save/load."

    def test_incremental_save_overwrites_atomically(self, tmp_path):
        """Two consecutive saves: second must atomically replace first."""
        orch = _make_orch()
        tasks = _make_tasks(2)
        gate = RandomGate(escalation_rate=1.0, seed=4)
        state_file = tmp_path / "incremental.json"

        _run_batch(orch, tasks, gate)
        orch.save_bandit_state(state_file)
        first_json = state_file.read_text(encoding="utf-8")

        _run_batch(orch, tasks, gate)
        orch.save_bandit_state(state_file)
        second_json = state_file.read_text(encoding="utf-8")

        assert first_json != second_json, "Second save must update file content."
        assert not state_file.with_suffix(
            ".tmp"
        ).exists(), ".tmp must not remain after second save."

    def test_save_to_nested_directory_creates_parents(self, tmp_path):
        """Atomic save must create missing parent directories automatically."""
        orch = _make_orch()
        nested = tmp_path / "week9" / "phase5" / "bandit.json"
        orch.save_bandit_state(nested)
        assert nested.exists(), "Nested state file must be created."
        assert not nested.with_suffix(".tmp").exists()

    def test_state_json_is_valid_and_complete_after_save(self, tmp_path):
        """The saved JSON must be parseable and contain all required keys."""
        orch = _make_orch()
        tasks = _make_tasks(2)
        gate = RandomGate(escalation_rate=1.0, seed=6)
        _run_batch(orch, tasks, gate)

        state_file = tmp_path / "valid_state.json"
        orch.save_bandit_state(state_file)

        raw = json.loads(state_file.read_text(encoding="utf-8"))
        for key in ("arms", "alpha", "d", "A", "b"):
            assert key in raw, f"Missing required key {key!r} in saved JSON."
        for arm in ("react", "debate", "reflexion"):
            assert arm in raw["A"], f"Missing arm {arm!r} in A matrix."
            assert arm in raw["b"], f"Missing arm {arm!r} in b vector."


# ---------------------------------------------------------------------------
# 5. Full week-9 composite scenario:
#    warm-start -> batch -> metrics -> atomic save -> reload
# ---------------------------------------------------------------------------


class TestWeek9FullScenario:
    """Single end-to-end test exercising all four Phase 1-4 deliverables."""

    def test_full_pipeline_scenario(self, tmp_path):
        """
        Step 1 - Warm-start bandit from historical traces.
        Step 2 - Run a forced-ESCALATE batch.
        Step 3 - Compute strategy allocation metrics.
        Step 4 - Atomically save the bandit state.
        Step 5 - Reload and verify round-trip fidelity + valid arm selection.
        """
        sample_task = get_mock_tasks(1)[0]

        # Step 1: warm-start
        orch = _make_orch(answer="42")
        traces: list[tuple[Task, str, float]] = [
            (sample_task, "react", 0.8),
            (sample_task, "debate", 0.5),
            (sample_task, "reflexion", 0.3),
        ]
        count = orch.warmstart_bandit(traces)
        assert count == 3
        warm_b_sum = sum(float(orch.bandit_router.b[a].sum()) for a in orch.bandit_router.arms)
        assert warm_b_sum != 0.0, "Warm-started bandit must have non-zero b-vectors."

        # Step 2: run batch
        tasks = _make_tasks(6)
        gate = RandomGate(escalation_rate=1.0, seed=13)
        results = _run_batch(orch, tasks, gate)

        assert len(results) == 6
        for r in results:
            assert r.mas_strategy in ("react", "debate", "reflexion")
            assert r.mas_tokens > 0

        # Step 3: strategy allocation metrics
        alloc = compute_mas_strategy_allocation(results)
        assert alloc["total_escalated"] == 6
        pct_sum = alloc["react_pct"] + alloc["debate_pct"] + alloc["reflexion_pct"]
        assert (
            abs(pct_sum - 100.0) < 0.05
        ), f"Percentages must sum to ~100 (within 0.05), got {pct_sum:.6f}"

        metrics = compute_evaluation_metrics(results)
        assert metrics["escalate_rate"] == 1.0
        assert metrics["strategy_allocation"]["total_escalated"] == 6
        assert metrics["n_tasks"] == 6

        # Step 4: atomic save
        state_file = tmp_path / "week9_bandit_state.json"
        orch.save_bandit_state(state_file)
        assert state_file.exists()
        assert not state_file.with_suffix(
            ".tmp"
        ).exists(), ".tmp sidecar must not exist after atomic save."

        # Step 5: reload and verify
        restored = MASOrchestrator(default_strategy="bandit", bandit_state_path=state_file)
        for arm in orch.bandit_router.arms:
            assert np.allclose(
                orch.bandit_router.b[arm], restored.bandit_router.b[arm]
            ), f"b-vector for arm={arm!r} must be identical after reload."

        chosen = restored.bandit_router.select_arm(sample_task)
        assert chosen in (
            "react",
            "debate",
            "reflexion",
        ), f"Reloaded bandit must select a valid arm, got {chosen!r}"

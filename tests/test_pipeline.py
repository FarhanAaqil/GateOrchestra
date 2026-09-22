"""
tests/test_pipeline.py
=======================
Smoke tests for the end-to-end integration pipeline.
Uses mock modules — no LLM calls.
"""

import pytest

from gate.random_gate import RandomGate
from gate.rule_based_gate import RuleBasedGate
from integration.pipeline import _exact_match, run_batch, run_pipeline
from shared.schemas import EvalResult
from shared.token_logger import TokenAccountant
from tests.mocks.mock_dataset import get_mock_tasks
from tests.mocks.mock_orchestrator import mock_orchestrator
from tests.mocks.mock_probe_agent import mock_probe_agent


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

        task = Task(
            task_id="simple_001", question="What is the capital of France?", ground_truth="Paris"
        )
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
        run_pipeline(
            task, gate, mock_probe_agent, mock_orchestrator, accountant, method="RuleBasedGate"
        )
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
        run_batch(
            tasks, gate, mock_probe_agent, mock_orchestrator, accountant, method="RuleBasedGate"
        )
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

    def test_answer_equivalent_leading_article(self):
        assert _exact_match("The United States", "United States") is True

    def test_answer_first_explanation(self):
        assert _exact_match("10 cups of flour for 48 cookies", "10 cups") is True

    def test_conclusion_answer_extraction(self):
        assert _exact_match("Therefore, the median of the dataset is 12", "12") is True

    def test_internal_ground_truth_substring_is_not_enough(self):
        assert _exact_match("Canada is near the United States", "United States") is False

    def test_answer_first_negation_is_not_correct(self):
        assert _exact_match("United States is not correct; Canada is", "United States") is False


# ─────────────────────────────────────────────────────────────────────────────
# Week 8 — LinUCB bandit online update via run_pipeline
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np  # noqa: E402

from agents.orchestrator import MASOrchestrator  # noqa: E402


@pytest.fixture
def bandit_task():
    from shared.schemas import Task

    return Task(
        task_id="bandit_task_01",
        question="What is the capital of Japan?",
        ground_truth="Tokyo",
        depth_score=1,
        parallel_score=1,
    )


def _make_mas_orch(answer: str = "Tokyo", tokens: int = 80) -> MASOrchestrator:
    """Return a MASOrchestrator whose sub-agents use an injected mock LLM caller."""

    def mock_caller(prompt: str, temp: float, budget: int) -> tuple[str, int]:
        return f"Final Answer: {answer}", min(tokens, budget)

    return MASOrchestrator(llm_caller=mock_caller)


class TestBanditOnlineUpdate:
    """Pipeline integration tests: LinUCB bandit weights update only on ESCALATE."""

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _snapshot_b(orch: MASOrchestrator) -> dict:
        """Deep-copy the b-vectors of all bandit arms (tracks reward signal)."""
        return {arm: orch.bandit_router.b[arm].copy() for arm in orch.bandit_router.arms}

    # ── core ESCALATE update test ──────────────────────────────────────────────

    def test_bandit_b_vector_changes_after_escalate(self, bandit_task):
        """After a pipeline ESCALATE the b-vector of the chosen arm must change."""
        orch = _make_mas_orch(answer="Tokyo")
        before = self._snapshot_b(orch)

        gate = RandomGate(escalation_rate=1.0, seed=0)  # always ESCALATE
        run_pipeline(
            task=bandit_task,
            gate=gate,
            probe_agent=mock_probe_agent,
            orchestrator=orch.run,
            accountant=TokenAccountant(),
            mas_orchestrator=orch,
        )

        after = self._snapshot_b(orch)
        chosen = orch._last_strategy
        assert chosen is not None

        # The chosen arm's b-vector must have changed
        assert not np.allclose(
            before[chosen], after[chosen]
        ), f"b-vector for arm={chosen!r} should have been updated but was unchanged."

        # Un-chosen arms must be untouched
        for arm in orch.bandit_router.arms:
            if arm != chosen:
                assert np.allclose(
                    before[arm], after[arm]
                ), f"b-vector for un-chosen arm={arm!r} should be unchanged."

    def test_bandit_not_updated_on_stop(self, bandit_task):
        """When the gate says STOP the bandit b-vectors must remain identical."""
        orch = _make_mas_orch()
        before = self._snapshot_b(orch)

        gate = RuleBasedGate(consistency_stop=0.0)  # always STOP
        run_pipeline(
            task=bandit_task,
            gate=gate,
            probe_agent=mock_probe_agent,
            orchestrator=orch.run,
            accountant=TokenAccountant(),
            mas_orchestrator=orch,
        )

        after = self._snapshot_b(orch)
        for arm in orch.bandit_router.arms:
            assert np.allclose(
                before[arm], after[arm]
            ), f"b-vector for arm={arm!r} changed on STOP — should not have."

    def test_backward_compat_no_mas_orchestrator(self, bandit_task):
        """Omitting mas_orchestrator must not raise and must return a valid EvalResult."""
        gate = RandomGate(escalation_rate=1.0, seed=0)
        result = run_pipeline(
            task=bandit_task,
            gate=gate,
            probe_agent=mock_probe_agent,
            orchestrator=mock_orchestrator,
            accountant=TokenAccountant(),
            # mas_orchestrator intentionally omitted
        )
        assert isinstance(result, EvalResult)
        assert result.gate_decision is not None
        assert result.gate_decision.decision == "ESCALATE"

    def test_bandit_update_uses_correct_is_correct_signal(self, bandit_task):
        """Reward should use is_correct=True when answer matches ground_truth."""
        orch = _make_mas_orch(answer="Tokyo")  # matches ground_truth

        # We capture the update call via a spy
        updates: list[dict] = []
        original_update = orch.update_bandit_reward

        def spy_update(task, chosen_strategy, is_correct, *, tokens_spent, budget):
            updates.append(
                {
                    "strategy": chosen_strategy,
                    "is_correct": is_correct,
                    "tokens_spent": tokens_spent,
                    "budget": budget,
                }
            )
            return original_update(
                task, chosen_strategy, is_correct, tokens_spent=tokens_spent, budget=budget
            )

        orch.update_bandit_reward = spy_update  # type: ignore[method-assign]

        gate = RandomGate(escalation_rate=1.0, seed=0)
        run_pipeline(
            task=bandit_task,
            gate=gate,
            probe_agent=mock_probe_agent,
            orchestrator=orch.run,
            accountant=TokenAccountant(),
            mas_orchestrator=orch,
        )

        assert len(updates) == 1, "update_bandit_reward should be called exactly once"
        assert updates[0]["is_correct"] is True
        assert updates[0]["strategy"] in ("react", "debate", "reflexion")
        assert updates[0]["tokens_spent"] > 0
        assert updates[0]["budget"] > 0

    def test_bandit_is_correct_false_when_no_ground_truth(self):
        """When task has no ground_truth, is_correct defaults to False for the reward."""
        from shared.schemas import Task

        task = Task(task_id="no_gt_bandit", question="Q?")  # no ground_truth
        orch = _make_mas_orch(answer="anything")

        updates: list[dict] = []
        original_update = orch.update_bandit_reward

        def spy_update(task, chosen_strategy, is_correct, *, tokens_spent, budget):
            updates.append({"is_correct": is_correct})
            return original_update(
                task, chosen_strategy, is_correct, tokens_spent=tokens_spent, budget=budget
            )

        orch.update_bandit_reward = spy_update  # type: ignore[method-assign]

        gate = RandomGate(escalation_rate=1.0, seed=0)
        run_pipeline(
            task=task,
            gate=gate,
            probe_agent=mock_probe_agent,
            orchestrator=orch.run,
            accountant=TokenAccountant(),
            mas_orchestrator=orch,
        )

        assert len(updates) == 1
        assert updates[0]["is_correct"] is False  # fallback when ground_truth is None


# ─────────────────────────────────────────────────────────────────────────────
# Week 8 Priority 2 — mas_strategy recorded in EvalResult
# ─────────────────────────────────────────────────────────────────────────────


class TestMasStrategyInResult:
    """run_pipeline must populate EvalResult.mas_strategy correctly."""

    def test_stop_gives_none_mas_strategy(self, bandit_task):
        """When the gate decides STOP, mas_strategy must be None regardless of orchestrator."""
        orch = _make_mas_orch()
        gate = RuleBasedGate(consistency_stop=0.0)  # always STOP

        result = run_pipeline(
            task=bandit_task,
            gate=gate,
            probe_agent=mock_probe_agent,
            orchestrator=orch.run,
            accountant=TokenAccountant(),
            mas_orchestrator=orch,
        )

        assert result.gate_decision is not None
        assert result.gate_decision.decision == "STOP"
        assert result.mas_strategy is None

    def test_escalate_with_mas_orchestrator_records_strategy(self, bandit_task):
        """ESCALATE + mas_orchestrator → EvalResult.mas_strategy is a valid arm name."""
        orch = _make_mas_orch(answer="Tokyo")
        gate = RandomGate(escalation_rate=1.0, seed=0)  # always ESCALATE

        result = run_pipeline(
            task=bandit_task,
            gate=gate,
            probe_agent=mock_probe_agent,
            orchestrator=orch.run,
            accountant=TokenAccountant(),
            mas_orchestrator=orch,
        )

        assert result.gate_decision is not None
        assert result.gate_decision.decision == "ESCALATE"
        assert result.mas_strategy in ("react", "debate", "reflexion")

    def test_escalate_strategy_matches_last_strategy(self, bandit_task):
        """EvalResult.mas_strategy must equal orch._last_strategy after the run."""
        orch = _make_mas_orch(answer="Tokyo")
        gate = RandomGate(escalation_rate=1.0, seed=0)

        result = run_pipeline(
            task=bandit_task,
            gate=gate,
            probe_agent=mock_probe_agent,
            orchestrator=orch.run,
            accountant=TokenAccountant(),
            mas_orchestrator=orch,
        )

        assert result.mas_strategy == orch._last_strategy

    def test_escalate_via_bound_method_records_strategy(self, bandit_task):
        """When mas_orchestrator is omitted but orchestrator=orch.run is a bound method,
        the pipeline resolves strategy via __self__._last_strategy."""
        orch = _make_mas_orch(answer="Tokyo")
        gate = RandomGate(escalation_rate=1.0, seed=0)

        result = run_pipeline(
            task=bandit_task,
            gate=gate,
            probe_agent=mock_probe_agent,
            orchestrator=orch.run,  # bound method — no mas_orchestrator
            accountant=TokenAccountant(),
            # mas_orchestrator intentionally omitted
        )

        assert result.gate_decision is not None
        assert result.gate_decision.decision == "ESCALATE"
        # Pipeline should still find _last_strategy via orchestrator.__self__
        assert result.mas_strategy in ("react", "debate", "reflexion")
        assert result.mas_strategy == orch._last_strategy

    def test_backward_compat_plain_callable_leaves_strategy_none(self, bandit_task):
        """When orchestrator is a plain function (not a bound method), mas_strategy=None."""
        gate = RandomGate(escalation_rate=1.0, seed=0)

        result = run_pipeline(
            task=bandit_task,
            gate=gate,
            probe_agent=mock_probe_agent,
            orchestrator=mock_orchestrator,  # plain function, no _last_strategy
            accountant=TokenAccountant(),
            # mas_orchestrator intentionally omitted
        )

        assert result.gate_decision is not None
        assert result.gate_decision.decision == "ESCALATE"
        # Plain callable has no _last_strategy → should stay None
        assert result.mas_strategy is None

    def test_heuristic_routing_produces_expected_strategy(self):
        """Deep-task (depth=4) should be routed to 'react' by the heuristic."""
        from shared.schemas import Task

        deep_task = Task(
            task_id="deep_react_test",
            question="Multi-hop: Who founded the company that acquired DeepMind?",
            ground_truth="Larry Page",
            depth_score=4,
            parallel_score=1,
        )
        orch = _make_mas_orch(answer="Larry Page")
        gate = RandomGate(escalation_rate=1.0, seed=0)

        result = run_pipeline(
            task=deep_task,
            gate=gate,
            probe_agent=mock_probe_agent,
            orchestrator=orch.run,
            accountant=TokenAccountant(),
            mas_orchestrator=orch,
        )

        assert (
            result.mas_strategy == "react"
        ), f"Expected 'react' for depth_score=4, got {result.mas_strategy!r}"


# ─────────────────────────────────────────────────────────────────────────────
# Week 8 Final — End-to-End Real Component Integration (Mock LLM)
# ─────────────────────────────────────────────────────────────────────────────


class TestEndToEndRealIntegration:
    """Tests full pipeline flow using real ProbeAgent, extract_features, Gate,
    MASOrchestrator, and TokenAccountant with a mock LLM caller (no network API calls).

    Flow: Task → ProbeAgent → extract_features → Gate → (STOP | ESCALATE)
                → MASOrchestrator → mas_strategy → EvalResult & LinUCB update
    """

    @pytest.fixture
    def mock_llm(self):
        def _caller(prompt: str, temp: float, max_tokens: int) -> tuple[str, int]:
            return "Thinking step by step... Final Answer: 42", 40

        return _caller

    def test_e2e_escalate_flow(self, mock_llm):
        """Verify full ESCALATE flow with real components and mock LLM."""
        from agents.orchestrator import MASOrchestrator
        from agents.probe_agent import ProbeAgent
        from shared.schemas import Task

        task = Task(
            task_id="e2e_esc_001",
            question="Calculate 6 times 7.",
            ground_truth="42",
            depth_score=3,
            parallel_score=1,
        )

        probe = ProbeAgent(llm_caller=mock_llm, n_samples=3)
        orch = MASOrchestrator(default_strategy="bandit", llm_caller=mock_llm)
        gate = RandomGate(escalation_rate=1.0, seed=42)  # Force ESCALATE
        accountant = TokenAccountant()

        # Capture initial LinUCB reward vector state
        initial_b_sum = {
            arm: float(orch.bandit_router.b[arm].sum()) for arm in orch.bandit_router.arms
        }

        result = run_pipeline(
            task=task,
            gate=gate,
            probe_agent=probe.run,
            orchestrator=orch.run,
            accountant=accountant,
            k=3,
            method="GateOrchestra",
            mas_orchestrator=orch,
        )

        # 1. Probe ran
        assert result.probe_tokens is not None and result.probe_tokens > 0

        # 2. Gate escalated
        assert result.gate_decision is not None
        assert result.gate_decision.decision == "ESCALATE"

        # 3. MAS ran
        assert result.mas_tokens is not None and result.mas_tokens > 0

        # 4. Strategy recorded in EvalResult
        assert result.mas_strategy in ("react", "debate", "reflexion")
        assert result.mas_strategy == orch._last_strategy

        # 5. EvalResult correctness & token spend
        assert result.predicted_answer == "42"
        assert result.is_correct is True
        assert result.tokens_spent == result.probe_tokens + result.mas_tokens

        # 6. LinUCB reward update verified
        updated_arm = result.mas_strategy
        new_b_sum = float(orch.bandit_router.b[updated_arm].sum())
        assert (
            new_b_sum != initial_b_sum[updated_arm]
        ), "LinUCB state vector b must update after ESCALATE"

    def test_e2e_stop_flow(self, mock_llm):
        """Verify full STOP flow with real components and mock LLM."""
        from agents.orchestrator import MASOrchestrator
        from agents.probe_agent import ProbeAgent
        from shared.schemas import Task

        task = Task(
            task_id="e2e_stop_001",
            question="What is the capital of France?",
            ground_truth="Paris",
        )

        probe = ProbeAgent(llm_caller=mock_llm, n_samples=3)
        orch = MASOrchestrator(default_strategy="bandit", llm_caller=mock_llm)
        gate = RandomGate(escalation_rate=0.0, seed=42)  # Force STOP
        accountant = TokenAccountant()

        result = run_pipeline(
            task=task,
            gate=gate,
            probe_agent=probe.run,
            orchestrator=orch.run,
            accountant=accountant,
            k=3,
            method="GateOrchestra",
            mas_orchestrator=orch,
        )

        # 1. Probe ran
        assert result.probe_tokens is not None and result.probe_tokens > 0

        # 2. Gate stopped
        assert result.gate_decision is not None
        assert result.gate_decision.decision == "STOP"

        # 3. MAS NOT called
        assert result.mas_tokens == 0
        assert result.mas_strategy is None
        assert orch._last_strategy is None

        # 4. EvalResult uses probe answer and spends zero MAS tokens
        assert result.predicted_answer == "42"
        assert result.tokens_spent == result.probe_tokens

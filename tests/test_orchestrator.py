"""
tests/test_orchestrator.py
==========================
Unit tests for MASOrchestrator and sub-agents (Person 2).
"""

import pytest

from agents.orchestrator import (
    DebateAgent,
    MASOrchestrator,
    ReActAgent,
    ReflexionAgent,
    orchestrator,
)
from shared.schemas import Task


@pytest.fixture
def react_task() -> Task:
    return Task(
        task_id="orch_deep_01",
        question="Which country's capital has the Eiffel Tower and what is its currency?",
        depth_score=4,
        parallel_score=1,
        context="The Eiffel Tower is located in Paris, the capital of France. The currency of France is the Euro.",
    )


@pytest.fixture
def debate_task() -> Task:
    return Task(
        task_id="orch_debate_01",
        question="Compare the populations of Tokyo and London.",
        depth_score=2,
        parallel_score=3,
    )


@pytest.fixture
def reflexion_task() -> Task:
    return Task(
        task_id="orch_refl_01",
        question="What is 15 * 8?",
        depth_score=1,
        parallel_score=1,
    )


class TestSubAgents:
    def test_react_agent_execution(self, react_task):
        def mock_caller(prompt: str, temp: float, budget: int) -> tuple[str, int]:
            return "Thought: Looking at context... Final Answer: Euro", 35

        agent = ReActAgent(llm_caller=mock_caller)
        ans, tokens = agent.run(react_task, token_budget=200)
        assert ans == "Euro"
        assert tokens > 0
        assert tokens <= 200

    def test_debate_agent_execution(self, debate_task):
        def mock_caller(prompt: str, temp: float, budget: int) -> tuple[str, int]:
            return "Proposer solution: Tokyo is larger. Final Answer: Tokyo", 40

        agent = DebateAgent(num_rounds=1, llm_caller=mock_caller)
        ans, tokens = agent.run(debate_task, token_budget=300)
        assert ans == "Tokyo"
        assert tokens > 0
        assert tokens <= 300

    def test_reflexion_agent_execution(self, reflexion_task):
        def mock_caller(prompt: str, temp: float, budget: int) -> tuple[str, int]:
            return "Draft answer 120. Final Answer: 120", 30

        agent = ReflexionAgent(llm_caller=mock_caller)
        ans, tokens = agent.run(reflexion_task, token_budget=150)
        assert ans == "120"
        assert tokens > 0
        assert tokens <= 150


class TestMASOrchestrator:
    def test_strategy_selection(self, react_task, debate_task, reflexion_task):
        orch = MASOrchestrator()
        assert orch.select_strategy(react_task) == "react"
        assert orch.select_strategy(debate_task) == "debate"
        assert orch.select_strategy(reflexion_task) == "reflexion"

    def test_orchestrator_execution(self, react_task):
        def mock_caller(prompt: str, temp: float, budget: int) -> tuple[str, int]:
            return "Final Answer: Paris", 45

        orch = MASOrchestrator(llm_caller=mock_caller)
        ans, tokens = orch.run(react_task, token_budget=100)
        assert ans == "Paris"
        assert tokens <= 100

    def test_functional_orchestrator_callable(self, reflexion_task):
        # Default functional orchestrator should be callable and return tuple[str, int]
        ans, tokens = orchestrator(reflexion_task, token_budget=100)
        assert isinstance(ans, str)
        assert isinstance(tokens, int)


# ─────────────────────────────────────────────────────────────────────────────
# Week 8 — _last_strategy caching
# ─────────────────────────────────────────────────────────────────────────────


class TestLastStrategyCache:
    """Verify that MASOrchestrator.run() caches the chosen arm in _last_strategy."""

    def test_last_strategy_is_none_before_first_run(self):
        """_last_strategy must be None immediately after construction."""
        orch = MASOrchestrator()
        assert orch._last_strategy is None

    def test_last_strategy_set_after_run(self, reflexion_task):
        """After run(), _last_strategy must be one of the three valid arm names."""
        def mock_caller(prompt: str, temp: float, budget: int) -> tuple[str, int]:
            return "Final Answer: 120", 30

        orch = MASOrchestrator(llm_caller=mock_caller)
        orch.run(reflexion_task, token_budget=150)

        assert orch._last_strategy is not None
        assert orch._last_strategy in ("react", "debate", "reflexion")

    def test_last_strategy_reflects_forced_strategy(self, react_task, debate_task, reflexion_task):
        """_last_strategy must match the strategy the heuristic router picks."""
        def mock_caller(prompt: str, temp: float, budget: int) -> tuple[str, int]:
            return "Final Answer: X", 20

        orch = MASOrchestrator(llm_caller=mock_caller)

        orch.run(react_task, token_budget=100)
        assert orch._last_strategy == "react"

        orch.run(debate_task, token_budget=100)
        assert orch._last_strategy == "debate"

        orch.run(reflexion_task, token_budget=100)
        assert orch._last_strategy == "reflexion"

    def test_last_strategy_updated_on_each_run(self, react_task, reflexion_task):
        """Running twice must overwrite _last_strategy with the new arm."""
        def mock_caller(prompt: str, temp: float, budget: int) -> tuple[str, int]:
            return "Final Answer: Y", 25

        orch = MASOrchestrator(llm_caller=mock_caller)

        orch.run(react_task, token_budget=100)
        first = orch._last_strategy

        orch.run(reflexion_task, token_budget=100)
        second = orch._last_strategy

        # Both must be valid; the second run must not preserve the first value
        assert first in ("react", "debate", "reflexion")
        assert second in ("react", "debate", "reflexion")
        # react_task routes to "react", reflexion_task routes to "reflexion"
        assert first != second


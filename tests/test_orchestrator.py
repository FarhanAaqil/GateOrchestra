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

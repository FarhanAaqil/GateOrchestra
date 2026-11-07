"""Tests for the ReAct tools, the tool-using ReAct loop and the learned MAS router."""

from __future__ import annotations

import pytest

from agents.orchestrator.bandit_router import LinUCBRouter
from agents.orchestrator.orchestrator import MASOrchestrator
from agents.orchestrator.sub_agents import ReActAgent
from agents.orchestrator.tools import ToolError, calculate, lookup
from shared.schemas import EvalResult, Task

CONTEXT = (
    "Mount Everest is in the Himalayas. It is 8849 metres tall. "
    "K2 is the second highest mountain. The Nile flows north."
)


def _task(**kwargs) -> Task:
    defaults = {"task_id": "t1", "question": "How tall is Everest?", "ground_truth": "8849"}
    return Task(**{**defaults, **kwargs})


class TestCalculator:
    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("6 * 2.5 + 1", "16"),
            ("20000 * 0.85**2", "14450"),
            ("sqrt(49)", "7"),
            ("1,200 / 4", "300"),
            ("15% * 200", "30"),
            ("2 ^ 10", "1024"),
            ("7 × 6", "42"),
            ("-3 + abs(-5)", "2"),
        ],
    )
    def test_evaluates_arithmetic(self, expression, expected):
        assert calculate(expression) == expected

    @pytest.mark.parametrize(
        "expression",
        ["__import__('os')", "open('x')", "2 ** 1000", "1 / 0", "x + 1", "", "sqrt(-1)"],
    )
    def test_rejects_unsafe_or_invalid_input(self, expression):
        with pytest.raises(ToolError):
            calculate(expression)


class TestLookup:
    def test_returns_matching_sentences_in_order(self):
        result = lookup("Everest height", CONTEXT)
        assert result.startswith("Mount Everest is in the Himalayas.")
        assert "Nile" not in result

    def test_reports_missing_context_and_misses(self):
        assert "No context" in lookup("Everest", None)
        assert "No sentence" in lookup("Amazon", CONTEXT)


def _scripted(responses: list[str]):
    """An LLM caller that replays canned responses and records prompts."""
    prompts: list[str] = []

    def caller(prompt: str, temperature: float, max_tokens: int) -> tuple[str, int]:
        prompts.append(prompt)
        return responses[min(len(prompts) - 1, len(responses) - 1)], 40

    return caller, prompts


class TestReActAgent:
    def test_uses_lookup_then_finishes(self):
        caller, prompts = _scripted(
            [
                "I should search the context.\nAction 1: lookup[Everest]\nObservation 1: made up",
                "The context gives the height.\nAction 2: finish[8849 metres]",
            ]
        )
        agent = ReActAgent(llm_caller=caller)
        answer, tokens = agent.run(_task(context=CONTEXT), 2000)

        assert answer == "8849 metres"
        assert tokens == 80
        assert agent.last_trace[1] == {
            "step": "lookup[Everest]",
            "output": lookup("Everest", CONTEXT),
            "tokens": 0,
        }
        # The invented observation is dropped; the real one is fed back.
        assert "made up" not in prompts[1]
        assert "Observation 1: Mount Everest" in prompts[1]

    def test_calculator_errors_are_observations(self):
        caller, prompts = _scripted(
            ["Action 1: calculate[1/0]", "Action 2: calculate[3*4]", "Action 3: finish[12]"]
        )
        answer, _ = ReActAgent(llm_caller=caller).run(_task(), 2000)
        assert answer == "12"
        assert "Observation 1: Error:" in prompts[1]
        assert "Observation 2: 12" in prompts[2]

    def test_placeholder_finish_is_rejected(self):
        caller, prompts = _scripted(["Action 1: finish[answer]", "Action 2: finish[Paris]"])
        answer, _ = ReActAgent(llm_caller=caller).run(_task(), 2000)
        assert answer == "Paris"
        assert "not a placeholder" in prompts[1]

    def test_synthesises_an_answer_when_steps_run_out(self):
        caller, prompts = _scripted(["Still thinking...", "Paris"])
        answer, _ = ReActAgent(llm_caller=caller, max_steps=2).run(_task(), 2000)
        assert answer == "Paris"
        assert prompts[-1].endswith("Final Answer:")


class TestLearnedRouter:
    def test_state_round_trips(self):
        router = LinUCBRouter()
        router.update(_task(), "debate", 1.0)
        restored = LinUCBRouter()
        restored.from_dict(router.to_dict())
        assert restored.to_dict() == router.to_dict()
        assert restored.select_arm(_task()) == router.select_arm(_task())

    def test_frozen_orchestrator_does_not_learn(self):
        orch = MASOrchestrator(default_strategy="bandit", llm_caller=_scripted(["x"])[0])
        before = orch.bandit_router.to_dict()
        orch.learning = False
        orch.update_bandit_reward(_task(), "debate", True, 100, 1000)
        assert orch.bandit_router.to_dict() == before
        orch.learning = True
        orch.update_bandit_reward(_task(), "debate", True, 100, 1000)
        assert orch.bandit_router.to_dict() != before

    def test_router_learns_the_better_arm(self):
        router = LinUCBRouter(alpha=0.1)
        task = _task()
        for _ in range(20):
            router.update(task, "react", 0.0)
            router.update(task, "debate", 0.0)
            router.update(task, "reflexion", 1.0)
        assert router.select_arm(task) == "reflexion"


def test_routing_summary_counts_arms():
    from scripts.run_final_evaluation import _routing_summary

    def result(strategy, correct):
        return EvalResult(
            task_id="t",
            method="Always-MAS",
            predicted_answer="a",
            is_correct=correct,
            tokens_spent=10,
            mas_tokens=10,
            mas_strategy=strategy,
        )

    summary = _routing_summary(
        [result("react", True), result("react", False), result("debate", True), result(None, True)]
    )
    assert summary == {
        "debate": {"n": 1, "correct": 1, "accuracy": 1.0},
        "react": {"n": 2, "correct": 1, "accuracy": 0.5},
    }


def test_finish_written_as_the_final_answer_is_unwrapped():
    caller, _ = _scripted(["Final Answer: finish[South China Sea]"])
    answer, _ = ReActAgent(llm_caller=caller).run(_task(), 2000)
    assert answer == "South China Sea"

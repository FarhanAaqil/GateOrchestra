"""The optional concise-answer step for the multi-agent system (GATE_MAS_CONCISE)."""

from __future__ import annotations

import pytest

from agents.orchestrator.orchestrator import MASOrchestrator
from agents.orchestrator.sub_agents import SubAgentError
from shared.schemas import Task

TASK = Task(task_id="t1", question="What is the capital of France?")


def orchestrator(concise: bool, short_reply: str | Exception = "Paris") -> MASOrchestrator:
    mas = MASOrchestrator(default_strategy="react", llm_caller=lambda prompt, t, b: ("unused", 1))
    mas.concise = concise
    mas.react_agent.run = lambda task, budget: ("The capital of France is the city of Paris.", 300)

    def call(prompt: str, budget: int, step: str = "call"):
        assert (
            "Draft answer: The capital of France is the city of Paris." in prompt
            and step == "concise"
        )
        if isinstance(short_reply, Exception):
            raise short_reply
        return short_reply, 12

    mas.react_agent._call = call
    return mas


def test_off_by_default_so_the_benchmark_reproduces(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GATE_MAS_CONCISE", raising=False)
    assert MASOrchestrator(default_strategy="react").concise is False
    monkeypatch.setenv("GATE_MAS_CONCISE", "1")
    assert MASOrchestrator(default_strategy="react").concise is True
    assert orchestrator(concise=False).run(TASK, 1000) == (
        "The capital of France is the city of Paris.",
        300,
    )


def test_the_answer_is_restated_and_its_tokens_are_counted() -> None:
    assert orchestrator(concise=True).run(TASK, 1000) == ("Paris", 312)
    assert (
        orchestrator(concise=True, short_reply="Paris\nIt is in France.").run(TASK, 1000)[0]
        == "Paris"
    )


def test_a_failed_restatement_keeps_the_full_answer() -> None:
    answer, tokens = orchestrator(concise=True, short_reply=SubAgentError("down")).run(TASK, 1000)
    assert answer == "The capital of France is the city of Paris." and tokens == 300
    assert orchestrator(concise=True, short_reply="  ").run(TASK, 1000)[0].startswith("The capital")


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("Paris", "Paris"),
        ("Paris\nIt is in France.", "Paris"),
        (
            "The Taj Mahal is in Agra, India.\nIts capital is New Delhi.\nYamuna River",
            "Yamuna River",
        ),
        ("**Answer:** 42 seconds", "42 seconds"),
        ("   \n  ", ""),
    ],
)
def test_the_answer_line_is_picked_from_short_replies(reply: str, expected: str) -> None:
    from agents.orchestrator.orchestrator import _answer_line

    assert _answer_line(reply) == expected

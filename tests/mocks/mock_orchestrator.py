"""Mock MAS orchestrator — returns deterministic (answer, tokens_used) tuples."""

from __future__ import annotations

import random

from shared.schemas import Task

_rng = random.Random(99)


def mock_orchestrator(task: Task, token_budget: int) -> tuple[str, int]:
    """Returns a plausible MAS answer — no LLM calls.

    Respects the token budget (never exceeds it).
    """
    answers = ["Paris", "42", "The French Revolution", "Marie Curie", "1789"]
    answer = answers[(hash(task.task_id) + 1) % len(answers)]
    tokens = min(_rng.randint(200, 600), token_budget)
    return answer, tokens

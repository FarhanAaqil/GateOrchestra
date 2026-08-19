"""Mock probe agent for testing — returns deterministic ProbeResult objects."""

from __future__ import annotations

import random

from shared.schemas import ProbeResult, Task

_rng = random.Random(42)


def mock_probe_agent(task: Task) -> ProbeResult:
    """Returns a plausible ProbeResult for any Task — no LLM calls."""
    n_samples = 5
    answers = ["Paris", "42", "The French Revolution", "Marie Curie", "1789"]
    majority = answers[hash(task.task_id) % len(answers)]
    agree = _rng.randint(3, 5)
    raw = [majority] * agree + ["Unknown"] * (n_samples - agree)
    _rng.shuffle(raw)

    return ProbeResult(
        task_id=task.task_id,
        answer=majority,
        consistency_score=agree / n_samples,
        tokens_used=_rng.randint(80, 200),
        raw_outputs=raw,
        model_name="mock-model",
    )

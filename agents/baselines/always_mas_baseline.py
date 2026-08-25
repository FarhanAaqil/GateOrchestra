"""
agents/baselines/always_mas_baseline.py
=======================================
Always-MAS Baseline Runner for GateOrchestra (Person 2).

This baseline sends EVERY task directly to the full Multi-Agent System (MAS)
Orchestrator without using the probe or the gate. It represents the high-cost,
high-compute ceiling against which GateOrchestra token savings are measured.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

from agents.orchestrator.orchestrator import orchestrator as default_orchestrator
from shared.config import K_DEFAULT, PROBE_TOKEN_BUDGET
from shared.schemas import EvalResult, Task
from shared.token_logger import TokenAccountant

logger = logging.getLogger(__name__)

OrchestratorFn = Callable[[Task, int], tuple[str, int]]


def _exact_match(predicted: str, ground_truth: str) -> bool:
    """Normalized exact match evaluation helper."""
    import re

    def normalize(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"[^\w\s]", "", s)
        return " ".join(s.split())

    return normalize(predicted) == normalize(ground_truth)


def run_always_mas_baseline(
    task: Task,
    orchestrator_fn: Optional[OrchestratorFn] = None,
    accountant: Optional[TokenAccountant] = None,
    token_budget: Optional[int] = None,
) -> EvalResult:
    """Run the Always-MAS baseline on a single task.

    Args:
        task: The task to evaluate.
        orchestrator_fn: Orchestrator callable (defaults to agents.orchestrator.orchestrator).
        accountant: Optional TokenAccountant instance for logging token spend.
        token_budget: Max token budget for MAS (defaults to K_DEFAULT * PROBE_TOKEN_BUDGET).

    Returns:
        EvalResult for the Always-MAS method.
    """
    fn = orchestrator_fn or default_orchestrator
    budget = token_budget if token_budget is not None else (K_DEFAULT * PROBE_TOKEN_BUDGET)
    start_time = time.perf_counter()

    answer, mas_tokens = fn(task, budget)

    if accountant is not None:
        accountant.log(
            task_id=task.task_id,
            method="Always-MAS",
            stage="mas",
            tokens=mas_tokens,
            path="ESCALATE",
        )

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    is_correct: Optional[bool] = None
    if task.ground_truth is not None:
        is_correct = _exact_match(answer, task.ground_truth)

    return EvalResult(
        task_id=task.task_id,
        method="Always-MAS",
        predicted_answer=answer,
        is_correct=is_correct,
        tokens_spent=mas_tokens,
        probe_tokens=None,
        mas_tokens=mas_tokens,
        gate_decision=None,
        latency_ms=latency_ms,
    )


def run_always_mas_batch(
    tasks: list[Task],
    orchestrator_fn: Optional[OrchestratorFn] = None,
    accountant: Optional[TokenAccountant] = None,
    token_budget: Optional[int] = None,
) -> list[EvalResult]:
    """Run the Always-MAS baseline across a list of tasks."""
    return [
        run_always_mas_baseline(
            task,
            orchestrator_fn=orchestrator_fn,
            accountant=accountant,
            token_budget=token_budget,
        )
        for task in tasks
    ]

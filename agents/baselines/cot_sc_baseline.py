"""
agents/baselines/cot_sc_baseline.py
===================================
CoT-SC-Only Baseline Runner for GateOrchestra (Person 2).

This baseline runs the cheap single-agent CoT-SC probe on all tasks without gating.
It represents the low-cost / lower-compute baseline for comparison against GateOrchestra
and Always-MAS.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from agents.probe_agent import probe_agent as default_probe_agent
from shared.schemas import EvalResult, ProbeResult, Task
from shared.token_logger import TokenAccountant

logger = logging.getLogger(__name__)

ProbeAgentFn = Callable[[Task], ProbeResult]


def _exact_match(predicted: str, ground_truth: str) -> bool:
    """Normalized exact match evaluation helper."""
    import re

    def normalize(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"[^\w\s]", "", s)
        return " ".join(s.split())

    return normalize(predicted) == normalize(ground_truth)


def run_cot_sc_baseline(
    task: Task,
    probe_fn: ProbeAgentFn | None = None,
    accountant: TokenAccountant | None = None,
) -> EvalResult:
    """Run the CoT-SC-only baseline on a single task.

    Args:
        task: The task to evaluate.
        probe_fn: Probe agent function (defaults to agents.probe_agent.probe_agent).
        accountant: Optional TokenAccountant instance for logging token spend.

    Returns:
        EvalResult for the CoT-SC-only method.
    """
    fn = probe_fn or default_probe_agent
    start_time = time.perf_counter()

    probe: ProbeResult = fn(task)

    if accountant is not None:
        accountant.log(
            task_id=task.task_id,
            method="CoT-SC-only",
            stage="probe",
            tokens=probe.tokens_used,
            path="N/A",
        )

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    is_correct: bool | None = None
    if task.ground_truth is not None:
        is_correct = _exact_match(probe.answer, task.ground_truth)

    return EvalResult(
        task_id=task.task_id,
        method="CoT-SC-only",
        predicted_answer=probe.answer,
        is_correct=is_correct,
        tokens_spent=probe.tokens_used,
        probe_tokens=probe.tokens_used,
        mas_tokens=None,
        gate_decision=None,
        latency_ms=latency_ms,
    )


def run_cot_sc_batch(
    tasks: list[Task],
    probe_fn: ProbeAgentFn | None = None,
    accountant: TokenAccountant | None = None,
) -> list[EvalResult]:
    """Run the CoT-SC baseline across a list of tasks."""
    return [run_cot_sc_baseline(task, probe_fn=probe_fn, accountant=accountant) for task in tasks]

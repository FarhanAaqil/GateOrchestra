"""
integration/pipeline.py
========================
The ONLY file that imports across all 4 modules.

Wires: Task → ProbeAgent → FeatureExtractor → Gate → (STOP | ESCALATE) → TokenLog → EvalResult

Design:
  - Dependencies (probe_agent, orchestrator) are injected — not hardcoded.
    This means swapping from mocks to real implementations in Week 10
    requires changing 0 lines in this file.
  - All token spend goes through TokenAccountant — no raw token counts escape.

Person 3 owns this file.

Week 4:  stub with mocked probe_agent and orchestrator (injectable)
Week 10: swap mock injections for real Person 2 modules
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from gate.classifier import GateClassifier
from gate.feature_extractor import extract_features
from shared.config import K_DEFAULT
from shared.schemas import EvalResult, GateDecision, ProbeResult, Task
from shared.token_logger import TokenAccountant

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Type aliases for injectable dependencies
# ─────────────────────────────────────────────────────────────────────────────

ProbeAgentFn = Callable[[Task], ProbeResult]
"""Signature that any probe agent implementation must satisfy."""

OrchestratorFn = Callable[[Task, int], tuple[str, int]]
"""Signature: (task, token_budget) → (answer, tokens_used)."""


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────────────────────────────────────


def run_pipeline(
    task: Task,
    gate: GateClassifier,
    probe_agent: ProbeAgentFn,
    orchestrator: OrchestratorFn,
    accountant: TokenAccountant,
    k: int = K_DEFAULT,
    method: str = "GateOrchestra",
) -> EvalResult:
    """Run the full GateOrchestra pipeline for a single task.

    Pipeline stages:
      1. Probe  — cheap CoT-SC pass via probe_agent
      2. Extract — extract GateFeatures from (task, probe)
      3. Gate   — route to STOP or ESCALATE
      4. Route  — if STOP: return probe answer; if ESCALATE: call orchestrator
      5. Log    — record all token spend to accountant
      6. Return — build and return EvalResult

    Args:
        task:         The task to evaluate.
        gate:         A trained GateClassifier (or RuleBasedGate/RandomGate).
        probe_agent:  Callable matching ProbeAgentFn signature.
        orchestrator: Callable matching OrchestratorFn signature.
        accountant:   TokenAccountant instance to log spend.
        k:            Token budget multiplier for MAS.
        method:       Label for logging (e.g. "GateOrchestra", "RuleBasedGate").

    Returns:
        EvalResult for this task.
    """
    logger.info(f"[Pipeline] task={task.task_id} method={method} k={k}")

    # ── Stage 1: Probe ────────────────────────────────────────────────────
    probe: ProbeResult = probe_agent(task)
    accountant.log(task.task_id, method, stage="probe", tokens=probe.tokens_used, path="N/A")
    logger.debug(f"  Probe: consistency={probe.consistency_score:.2f} tokens={probe.tokens_used}")

    # ── Stage 2: Feature extraction ───────────────────────────────────────
    features = extract_features(task, probe)

    # ── Stage 3: Gate decision ────────────────────────────────────────────
    decision: GateDecision = gate.predict(features, k=k, probe_tokens=probe.tokens_used)
    logger.debug(f"  Gate: decision={decision.decision} confidence={decision.confidence:.2f}")

    # ── Stage 4: Route ────────────────────────────────────────────────────
    if decision.decision == "STOP":
        answer = probe.answer
        mas_tokens = 0
        accountant.log(task.task_id, method, stage="mas", tokens=0, path="STOP")
        logger.debug(f"  STOP → using probe answer: {answer!r}")

    else:  # ESCALATE
        token_budget = decision.token_budget_cap or (k * probe.tokens_used)
        answer, mas_tokens = orchestrator(task, token_budget)
        accountant.log(task.task_id, method, stage="mas", tokens=mas_tokens, path="ESCALATE")
        logger.debug(f"  ESCALATE → MAS answer: {answer!r} tokens={mas_tokens}")

    # ── Stage 5: Build result ─────────────────────────────────────────────
    total_tokens = probe.tokens_used + mas_tokens
    is_correct: bool | None = None
    if task.ground_truth is not None:
        is_correct = _exact_match(answer, task.ground_truth)

    return EvalResult(
        task_id=task.task_id,
        method=method,  # type: ignore[arg-type]
        predicted_answer=answer,
        is_correct=is_correct,
        tokens_spent=total_tokens,
        probe_tokens=probe.tokens_used,
        mas_tokens=mas_tokens,
        gate_decision=decision,
    )


def run_batch(
    tasks: list[Task],
    gate: GateClassifier,
    probe_agent: ProbeAgentFn,
    orchestrator: OrchestratorFn,
    accountant: TokenAccountant,
    k: int = K_DEFAULT,
    method: str = "GateOrchestra",
) -> list[EvalResult]:
    """Run the pipeline on a batch of tasks.

    Returns:
        List of EvalResult, one per task (same order as input).
    """
    results = []
    for i, task in enumerate(tasks):
        logger.info(f"[Batch] {i+1}/{len(tasks)} task={task.task_id}")
        result = run_pipeline(task, gate, probe_agent, orchestrator, accountant, k, method)
        results.append(result)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _exact_match(predicted: str, ground_truth: str) -> bool:
    """Normalized exact match (lowercase, strip punctuation)."""
    import re
    def normalize(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"[^\w\s]", "", s)
        return " ".join(s.split())
    return normalize(predicted) == normalize(ground_truth)

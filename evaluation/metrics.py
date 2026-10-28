"""
evaluation/metrics.py
=====================
Evaluation and Pareto metrics computation module for GateOrchestra (Person 2 / Person 4).

Computes accuracy, token expenditure, gate decision ratios, and MAS strategy allocation
breakdowns (ReAct %, Debate %, Reflexion %) for evaluation runs and Pareto frontier plots.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from shared.grading import is_correct, normalize_text
from shared.schemas import EvalResult


def compute_mas_strategy_allocation(results: list[EvalResult]) -> dict[str, Any]:
    """Compute sub-agent strategy allocation counts and percentages for ESCALATE cases.

    Args:
        results: List of EvalResult records.

    Returns:
        Dict containing counts and percentages for ReAct, Debate, and Reflexion.
    """
    react_count = 0
    debate_count = 0
    reflexion_count = 0

    for r in results:
        strat = getattr(r, "mas_strategy", None)
        if strat == "react":
            react_count += 1
        elif strat == "debate":
            debate_count += 1
        elif strat == "reflexion":
            reflexion_count += 1

    total_escalated = react_count + debate_count + reflexion_count

    if total_escalated > 0:
        react_pct = round((react_count / total_escalated) * 100.0, 2)
        debate_pct = round((debate_count / total_escalated) * 100.0, 2)
        reflexion_pct = round((reflexion_count / total_escalated) * 100.0, 2)
    else:
        react_pct = 0.0
        debate_pct = 0.0
        reflexion_pct = 0.0

    return {
        "react_count": react_count,
        "debate_count": debate_count,
        "reflexion_count": reflexion_count,
        "total_escalated": total_escalated,
        "react_pct": react_pct,
        "debate_pct": debate_pct,
        "reflexion_pct": reflexion_pct,
    }


def compute_evaluation_metrics(
    results: list[EvalResult],
    baseline_results: list[EvalResult] | None = None,
) -> dict[str, Any]:
    """Compute comprehensive evaluation and Pareto metrics for a list of EvalResults.

    Args:
        results: Primary evaluation results (e.g. GateOrchestra method).
        baseline_results: Optional baseline results (e.g. Always-MAS) to compute
                          token savings % and accuracy delta.

    Returns:
        Dict containing accuracy, token metrics, gate decisions, strategy allocation,
        and relative baseline comparison metrics.
    """
    n = len(results)
    if n == 0:
        return {
            "n_tasks": 0,
            "accuracy": 0.0,
            "total_tokens": 0,
            "avg_tokens": 0.0,
            "stop_count": 0,
            "stop_rate": 0.0,
            "escalate_count": 0,
            "escalate_rate": 0.0,
            "strategy_allocation": compute_mas_strategy_allocation([]),
            "token_savings_pct": None,
            "accuracy_delta": None,
        }

    correct_count = sum(1 for r in results if r.is_correct is True)
    accuracy = round((correct_count / n) * 100.0, 2)

    total_tokens = sum(r.tokens_spent for r in results)
    avg_tokens = round(total_tokens / n, 2)

    stop_count = sum(1 for r in results if r.gate_decision and r.gate_decision.decision == "STOP")
    escalate_count = sum(
        1 for r in results if r.gate_decision and r.gate_decision.decision == "ESCALATE"
    )

    stop_rate = round(stop_count / n, 4)
    escalate_rate = round(escalate_count / n, 4)

    strategy_alloc = compute_mas_strategy_allocation(results)

    token_savings_pct: float | None = None
    accuracy_delta: float | None = None

    if baseline_results and len(baseline_results) > 0:
        base_n = len(baseline_results)
        base_correct = sum(1 for r in baseline_results if r.is_correct is True)
        base_acc = (base_correct / base_n) * 100.0
        accuracy_delta = round(accuracy - base_acc, 2)

        base_tokens = sum(r.tokens_spent for r in baseline_results)
        if base_tokens > 0:
            token_savings_pct = round(((base_tokens - total_tokens) / base_tokens) * 100.0, 2)

    return {
        "n_tasks": n,
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "avg_tokens": avg_tokens,
        "stop_count": stop_count,
        "stop_rate": stop_rate,
        "escalate_count": escalate_count,
        "escalate_rate": escalate_rate,
        "strategy_allocation": strategy_alloc,
        "token_savings_pct": token_savings_pct,
        "accuracy_delta": accuracy_delta,
    }


# -----------------------------------------------------------------------------
# Canonical Correctness Evaluation (Phase 5 – rescue plan)
# -----------------------------------------------------------------------------


def normalize_answer_for_eval(text: str) -> str:
    """Normalise an answer string the same way the canonical grader does."""
    return normalize_text(text)


def evaluate_answer(predicted: str, ground_truth: str, aliases: Iterable[str] = ()) -> bool:
    """Canonical correctness function used by all evaluation methods.

    Thin wrapper over :func:`shared.grading.is_correct`: numeric answers are
    compared by value at the gold answer's precision, text answers must equal
    the gold answer (or begin with it as whole words without a negation).
    Plain substring containment is never used, so "15" is not correct for
    "5" and "Unknown" is not correct for "no".
    """
    return is_correct(predicted, ground_truth, aliases)


def wilson_confidence_interval(
    n_correct: int, n_total: int, z: float = 1.96
) -> tuple[float, float]:
    """Compute Wilson 95% confidence interval for a proportion.

    Args:
        n_correct: Number of correct predictions.
        n_total: Total number of predictions.
        z: Z-score for the desired confidence level (default 1.96 for 95%).

    Returns:
        (lower, upper) bounds as fractions in [0, 1].
    """
    if n_total == 0:
        return 0.0, 0.0
    p = n_correct / n_total
    denominator = 1.0 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denominator
    margin = (z * (p * (1 - p) / n_total + z**2 / (4 * n_total**2)) ** 0.5) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def mcnemar_exact(
    correct_a: list[bool],
    correct_b: list[bool],
) -> dict[str, float | int]:
    """Exact (binomial) McNemar test for two methods graded on the same tasks.

    Args:
        correct_a: Per-task correctness of method A.
        correct_b: Per-task correctness of method B, in the same task order.

    Returns:
        Dict with ``a_only`` (A right, B wrong), ``b_only`` (B right, A wrong)
        and the two-sided ``p_value``. With no discordant pairs p_value is 1.0.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError("McNemar needs the same tasks for both methods")
    a_only = sum(1 for a, b in zip(correct_a, correct_b, strict=True) if a and not b)
    b_only = sum(1 for a, b in zip(correct_a, correct_b, strict=True) if b and not a)
    n = a_only + b_only
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p_value": 1.0}
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return {"a_only": a_only, "b_only": b_only, "p_value": min(1.0, 2 * tail)}

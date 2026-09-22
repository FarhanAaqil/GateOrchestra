"""
evaluation/metrics.py
=====================
Evaluation and Pareto metrics computation module for GateOrchestra (Person 2 / Person 4).

Computes accuracy, token expenditure, gate decision ratios, and MAS strategy allocation
breakdowns (ReAct %, Debate %, Reflexion %) for evaluation runs and Pareto frontier plots.
"""

from __future__ import annotations

from typing import Any

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

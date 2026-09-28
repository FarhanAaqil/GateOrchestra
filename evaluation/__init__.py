"""
evaluation/__init__.py
======================
Evaluation package for GateOrchestra (Person 2 / Person 4).
"""

from evaluation.metrics import (
    compute_evaluation_metrics,
    compute_mas_strategy_allocation,
    evaluate_answer,
    normalize_answer_for_eval,
    wilson_confidence_interval,
)

__all__ = [
    "compute_mas_strategy_allocation",
    "compute_evaluation_metrics",
    "evaluate_answer",
    "normalize_answer_for_eval",
    "wilson_confidence_interval",
]

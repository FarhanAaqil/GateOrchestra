"""
evaluation/__init__.py
======================
Evaluation package for GateOrchestra (Person 2 / Person 4).
"""

from evaluation.metrics import (
    compute_evaluation_metrics,
    compute_mas_strategy_allocation,
)

__all__ = [
    "compute_mas_strategy_allocation",
    "compute_evaluation_metrics",
]

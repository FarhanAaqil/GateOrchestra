"""
agents/baselines/__init__.py
============================
Baseline agents for GateOrchestra (Person 2).
Includes CoT-SC-only and Always-MAS baselines.
"""

from agents.baselines.always_mas_baseline import (
    run_always_mas_baseline,
    run_always_mas_batch,
)
from agents.baselines.cot_sc_baseline import (
    run_cot_sc_baseline,
    run_cot_sc_batch,
)

__all__ = [
    "run_cot_sc_baseline",
    "run_cot_sc_batch",
    "run_always_mas_baseline",
    "run_always_mas_batch",
]

"""shared package — public API."""

from shared.config import K_DEFAULT, K_VALUES, PROBE_TOKEN_BUDGET, TAU_ACC
from shared.data_loader import exact_match, load_all, load_split
from shared.schemas import (
    EvalResult,
    GateDecision,
    GateFeatures,
    ProbeResult,
    Task,
)
from shared.token_logger import TokenAccountant, get_global_accountant

__all__ = [
    # Schemas
    "Task",
    "ProbeResult",
    "GateFeatures",
    "GateDecision",
    "EvalResult",
    # Config
    "TAU_ACC",
    "K_DEFAULT",
    "K_VALUES",
    "PROBE_TOKEN_BUDGET",
    # Token logging
    "TokenAccountant",
    "get_global_accountant",
    # Data loading & evaluation
    "exact_match",
    "load_split",
    "load_all",
]

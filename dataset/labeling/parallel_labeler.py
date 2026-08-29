"""
dataset/labeling/parallel_labeler.py
======================================
Assigns parallel_score (1–4) to a Task based on extracted labeling features.

Parallel measures the number of independent reasoning threads that can be
solved concurrently:
  1 = sequential (single reasoning path, single-target query)
  2 = mild parallelism (pairwise entity comparison or 1–2 conjunction branches)
  3 = moderate parallelism (2–3 concurrent sub-questions, 3-way comparisons, or multi-item list)
  4 = high parallelism (3+ independent sub-tasks or complex multi-target operations)

Formula:
  weighted_score = (
      sub_q_norm       * w["sub_question_count"] +
      conj_norm        * w["conjunction_count"]  +
      list_norm        * w["list_count"]         +
      choice_norm      * w["choice_count"]       +
      branch_norm      * w["parallel_branches"]
  )

Thresholds:
  [0.00, 0.25) -> parallel 1
  [0.25, 0.55) -> parallel 2
  [0.55, 0.85) -> parallel 3
  [0.85, inf)  -> parallel 4

Person 1 owns this file.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Default configuration (mirror configs/dataset.yaml)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "sub_question_count": 0.30,
    "conjunction_count": 0.15,
    "list_count": 0.15,
    "choice_count": 0.20,
    "parallel_branches": 0.20,
}

# (upper_bound, label) — score < upper_bound -> label
DEFAULT_THRESHOLDS = [
    (0.20, 1),   # strictly sequential (single entity/target)
    (0.50, 2),   # mild: 2-way comparison, 1 sub-question
    (0.80, 3),   # moderate: 2 sub-questions, 3-way comparison, or lists
    (float("inf"), 4),  # high: 3+ sub-questions or large parallel sets
]

_CAPS = {
    "sub_question_count": 2.0,
    "conjunction_count": 3.0,
    "list_count": 3.0,
    "choice_count": 2.0,
    "parallel_branches": 2.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def assign_parallel(
    features: dict,
    weights: dict | None = None,
    thresholds: list[tuple[float, int]] | None = None,
) -> dict:
    """Compute parallel_score from labeling features.

    Args:
        features:   Output of dataset.labeling.feature_extractor.extract_labeling_features()
        weights:    Override default weight dict
        thresholds: Override default threshold list of (upper_bound, label) tuples

    Returns:
        dict with:
            parallel_score        : int   — final label 1–4
            parallel_raw_score    : float — weighted score before thresholding
            parallel_weights_used : dict  — which weights produced this score
            parallel_components   : dict  — normalized components
    """
    w = weights or DEFAULT_WEIGHTS
    t = thresholds or DEFAULT_THRESHOLDS

    sub_q = float(features.get("sub_question_count", 0))
    conj = float(features.get("conjunction_count", 0))
    lists = float(features.get("list_count", 0))
    choice = float(features.get("choice_count", 0))
    branches = float(features.get("parallel_branches", 1))

    # Normalize: map raw count -> [0, 1.5] scale relative to caps
    sub_q_norm = min(sub_q / _CAPS["sub_question_count"], 1.5)
    conj_norm = min(conj / _CAPS["conjunction_count"], 1.5)
    list_norm = min(lists / _CAPS["list_count"], 1.5)
    choice_norm = min(choice / _CAPS["choice_count"], 1.5) if choice > 0 else 0.0
    # Branch norm: 1 branch = 0.0, 2 branches = 1.0, 3+ branches = 1.5
    branch_norm = min(max(0.0, (branches - 1.0) / (_CAPS["parallel_branches"] - 1.0)), 1.5)

    raw_score = (
        sub_q_norm * w.get("sub_question_count", 0.30)
        + conj_norm * w.get("conjunction_count", 0.15)
        + list_norm * w.get("list_count", 0.15)
        + choice_norm * w.get("choice_count", 0.20)
        + branch_norm * w.get("parallel_branches", 0.20)
    )

    parallel_score = _threshold(raw_score, t)

    # Sanity clamp to [1, 4]
    parallel_score = max(1, min(4, int(parallel_score)))

    return {
        "parallel_score": parallel_score,
        "parallel_raw_score": round(raw_score, 4),
        "parallel_weights_used": dict(w),
        "parallel_components": {
            "sub_question_norm": round(sub_q_norm, 4),
            "conjunction_norm": round(conj_norm, 4),
            "list_norm": round(list_norm, 4),
            "choice_norm": round(choice_norm, 4),
            "branch_norm": round(branch_norm, 4),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _threshold(score: float, thresholds: list[tuple[float, int]]) -> int:
    for upper_bound, label in thresholds:
        if score < upper_bound:
            return label
    return thresholds[-1][1]


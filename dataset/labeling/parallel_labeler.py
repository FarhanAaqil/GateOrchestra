"""
dataset/labeling/parallel_labeler.py
======================================
Assigns parallel_score (1–4) to a Task based on extracted labeling features.

Parallel measures the number of independent reasoning threads that can be
solved concurrently — the more sub-questions or list items, the higher
the parallel score.

  1 = sequential (one reasoning path, no branching)
  2 = mild parallelism (1–2 independent sub-questions or facts)
  3 = moderate parallelism (2–3 parallel threads)
  4 = high parallelism (3+ independent threads or explicit list)

Formula:
  weighted_score = (
      sub_question_count * 0.50 +
      conjunction_count  * 0.30 +
      list_count         * 0.20
  )
  where each is capped to avoid extreme outliers.

Thresholds:
  [0.0, 0.5) → parallel 1
  [0.5, 1.5) → parallel 2
  [1.5, 2.5) → parallel 3
  [2.5, ∞  ) → parallel 4

All intermediate features are returned.

Person 1 owns this file.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Default configuration (mirror configs/dataset.yaml)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "sub_question_count": 0.50,
    "conjunction_count": 0.30,
    "list_count": 0.20,
}

# (upper_bound, label) — score < upper_bound → label
# Tuned: even 1-2 conjunctions should score parallel 2
DEFAULT_THRESHOLDS = [
    (0.10, 1),   # truly sequential, no conjunctions/lists
    (0.35, 2),   # mild: 1-2 conjunctions or 1 sub-question
    (0.65, 3),   # moderate: multiple sub-questions or list
    (float("inf"), 4),  # high: explicit multi-part queries
]

# Normalization caps — values above these are clipped before weighting
_CAPS = {
    "sub_question_count": 4.0,  # rarely more than 4 sub-questions
    "conjunction_count": 5.0,
    "list_count": 6.0,
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
    """
    w = weights or DEFAULT_WEIGHTS
    t = thresholds or DEFAULT_THRESHOLDS

    sub_q = float(features.get("sub_question_count", 0))
    conj  = float(features.get("conjunction_count", 0))
    lists = float(features.get("list_count", 0))

    # Normalize: map raw count → [0, 1] scale relative to caps
    sub_q_norm = min(sub_q / _CAPS["sub_question_count"], 1.5)
    conj_norm  = min(conj  / _CAPS["conjunction_count"],  1.5)
    list_norm  = min(lists / _CAPS["list_count"],          1.5)

    raw_score = (
        sub_q_norm * w["sub_question_count"]
        + conj_norm  * w["conjunction_count"]
        + list_norm  * w["list_count"]
    )

    parallel_score = _threshold(raw_score, t)

    return {
        "parallel_score": parallel_score,
        "parallel_raw_score": round(raw_score, 4),
        "parallel_weights_used": dict(w),
        "parallel_components": {
            "sub_question_norm": round(sub_q_norm, 4),
            "conjunction_norm": round(conj_norm, 4),
            "list_norm": round(list_norm, 4),
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

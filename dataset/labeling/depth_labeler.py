"""
dataset/labeling/depth_labeler.py
===================================
Assigns depth_score (1–5) to a Task based on extracted labeling features.

Depth measures reasoning chain complexity:
  1 = shallow, single-hop, direct lookup
  2 = 2-hop, moderate complexity
  3 = 3-hop or complex 2-hop
  4 = multi-hop with significant entity linking
  5 = deep multi-hop, requires several reasoning steps

Formula (weighted score → threshold → label):
  weighted_score = (
      hop_count        * 0.40 +
      clause_count     * 0.30 +
      entity_count     * 0.20 +
      word_count_norm  * 0.10
  )
  where word_count_norm = question_word_count / 15 (capped at 1.0)

Thresholds (tunable in configs/dataset.yaml):
  [0.0, 1.0) → depth 1
  [1.0, 2.0) → depth 2
  [2.0, 3.2) → depth 3
  [3.2, 4.2) → depth 4
  [4.2, ∞  ) → depth 5

All intermediate features are returned for transparency and debugging.

Person 1 owns this file.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Default thresholds (mirror configs/dataset.yaml)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "hop_count": 0.40,
    "clause_count": 0.30,
    "entity_count": 0.20,
    "word_count_norm": 0.10,
}

# Upper bounds for each depth bin (score < bound → label)
# Tuned to produce a natural spread across 1-hop arithmetic (depth 1-2)
# through 3-hop compositional (depth 4-5)
DEFAULT_THRESHOLDS = [
    (0.45, 1),  # 1-hop, no clauses, short question
    (0.85, 2),  # 1-2 hop, minimal clauses
    (1.30, 3),  # 2 hop with clauses or 3 hop simple
    (1.80, 4),  # 3 hop or complex 2-hop with many entities
    (float("inf"), 5),  # deep multi-hop
]

WORD_COUNT_NORM_DENOMINATOR = 15.0  # questions longer than this normalized to 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def assign_depth(
    features: dict,
    weights: dict | None = None,
    thresholds: list[tuple[float, int]] | None = None,
) -> dict:
    """Compute depth_score from labeling features.

    Args:
        features:   Output of dataset.labeling.feature_extractor.extract_labeling_features()
        weights:    Override default weight dict (keys: hop_count, clause_count,
                    entity_count, word_count_norm)
        thresholds: Override default threshold list of (upper_bound, label) tuples

    Returns:
        dict with:
            depth_score        : int  — final label 1–5
            depth_raw_score    : float — weighted score before thresholding
            depth_weights_used : dict — which weights produced this score
    """
    w = weights or DEFAULT_WEIGHTS
    t = thresholds or DEFAULT_THRESHOLDS

    hop_count = float(features.get("hop_count", 1))
    clause_count = float(features.get("clause_count", 0))
    entity_count = float(features.get("entity_count", 0))
    word_count = float(features.get("question_word_count", 0))

    # Normalize entity and clause counts to reasonable scale
    entity_norm = min(entity_count / 3.0, 2.0)  # 3 entities → 1.0, capped at 2.0
    clause_norm = min(clause_count / 2.0, 2.0)  # 2 clauses → 1.0, capped at 2.0
    word_norm = min(word_count / WORD_COUNT_NORM_DENOMINATOR, 1.0)

    raw_score = (
        hop_count * w["hop_count"]
        + clause_norm * w["clause_count"]
        + entity_norm * w["entity_count"]
        + word_norm * w["word_count_norm"]
    )

    depth_score = _threshold(raw_score, t)

    return {
        "depth_score": depth_score,
        "depth_raw_score": round(raw_score, 4),
        "depth_weights_used": dict(w),
        "depth_components": {
            "hop_count": hop_count,
            "clause_norm": round(clause_norm, 4),
            "entity_norm": round(entity_norm, 4),
            "word_norm": round(word_norm, 4),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _threshold(score: float, thresholds: list[tuple[float, int]]) -> int:
    """Map a continuous score to a discrete label using ordered thresholds."""
    for upper_bound, label in thresholds:
        if score < upper_bound:
            return label
    return thresholds[-1][1]  # fallback: highest label

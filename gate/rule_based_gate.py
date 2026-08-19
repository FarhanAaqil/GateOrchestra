"""
gate/rule_based_gate.py
=======================
Hand-crafted rule-based gate — a strong deterministic baseline.

Needed by Person 4 (evaluation) from Week 7 for baseline comparisons.
All thresholds are pulled from shared/config.py — no magic numbers here.

Rules (evaluated in priority order):
  1. If consistency_score >= THRESHOLD_HIGH AND entity_count < ENTITY_STOP  → STOP
  2. If estimated_depth >= DEPTH_ESCALATE OR estimated_parallel >= PARALLEL_ESCALATE → ESCALATE
  3. Default → STOP (conservative: when in doubt, trust the probe)

Person 3 owns this file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from shared.config import (
    RULE_CONSISTENCY_STOP_THRESHOLD,
    RULE_DEPTH_ESCALATE_THRESHOLD,
    RULE_ENTITY_STOP_THRESHOLD,
    RULE_PARALLEL_ESCALATE_THRESHOLD,
)
from shared.schemas import GateDecision, GateFeatures

logger = logging.getLogger(__name__)


@dataclass
class RuleTrace:
    """Debug trace of which rule fired (and why)."""

    rule_id: int
    rule_name: str
    decision: str
    reason: str


class RuleBasedGate:
    """
    Deterministic rule-based gate.  No training required.

    Usage::

        gate = RuleBasedGate()
        decision = gate.predict(features, k=3, probe_tokens=120)

    To customize thresholds::

        gate = RuleBasedGate(
            consistency_stop=0.9,
            entity_stop=2,
        )
    """

    def __init__(
        self,
        consistency_stop: float = RULE_CONSISTENCY_STOP_THRESHOLD,
        entity_stop: int = RULE_ENTITY_STOP_THRESHOLD,
        depth_escalate: float = RULE_DEPTH_ESCALATE_THRESHOLD,
        parallel_escalate: float = RULE_PARALLEL_ESCALATE_THRESHOLD,
    ) -> None:
        self.consistency_stop = consistency_stop
        self.entity_stop = entity_stop
        self.depth_escalate = depth_escalate
        self.parallel_escalate = parallel_escalate

    def predict(
        self,
        features: GateFeatures,
        k: int,
        probe_tokens: int,
    ) -> GateDecision:
        """Apply rule set and return a GateDecision.

        Args:
            features:     GateFeatures for the task.
            k:            Token budget multiplier.
            probe_tokens: Probe token count (used for token_budget_cap).
        """
        trace = self._apply_rules(features)
        logger.debug(
            f"[RuleBasedGate] task={features.task_id} "
            f"rule={trace.rule_name} decision={trace.decision} reason={trace.reason}"
        )

        return GateDecision(
            task_id=features.task_id,
            decision=trace.decision,  # type: ignore[arg-type]
            confidence=1.0,  # Rule-based: always 100% confident (deterministic)
            token_budget_cap=k * probe_tokens if trace.decision == "ESCALATE" else None,
            gate_type="rule_based",
        )

    def explain(self, features: GateFeatures) -> RuleTrace:
        """Return the rule that would fire for these features (for debugging)."""
        return self._apply_rules(features)

    def _apply_rules(self, features: GateFeatures) -> RuleTrace:
        """Evaluate rules in priority order and return the first match."""

        # ── Rule 1: High-confidence, simple task → STOP ────────────────────
        if (
            features.consistency_score >= self.consistency_stop
            and features.entity_count < self.entity_stop
        ):
            return RuleTrace(
                rule_id=1,
                rule_name="high_consistency_simple_task",
                decision="STOP",
                reason=(
                    f"consistency={features.consistency_score:.2f} >= {self.consistency_stop} "
                    f"AND entity_count={features.entity_count} < {self.entity_stop}"
                ),
            )

        # ── Rule 2a: Deep multi-hop task → ESCALATE ────────────────────────
        if (
            features.estimated_depth is not None
            and features.estimated_depth >= self.depth_escalate
        ):
            return RuleTrace(
                rule_id=2,
                rule_name="deep_task_escalate",
                decision="ESCALATE",
                reason=(
                    f"estimated_depth={features.estimated_depth:.2f} >= {self.depth_escalate}"
                ),
            )

        # ── Rule 2b: Highly parallel task → ESCALATE ───────────────────────
        if (
            features.estimated_parallel is not None
            and features.estimated_parallel >= self.parallel_escalate
        ):
            return RuleTrace(
                rule_id=3,
                rule_name="parallel_task_escalate",
                decision="ESCALATE",
                reason=(
                    f"estimated_parallel={features.estimated_parallel:.2f} >= {self.parallel_escalate}"
                ),
            )

        # ── Default: trust the probe ────────────────────────────────────────
        return RuleTrace(
            rule_id=0,
            rule_name="default_stop",
            decision="STOP",
            reason="No escalation rule fired; defaulting to STOP",
        )

    def rule_summary(self) -> dict:
        """Return a summary of the configured thresholds (for logging/reporting)."""
        return {
            "consistency_stop_threshold": self.consistency_stop,
            "entity_stop_threshold": self.entity_stop,
            "depth_escalate_threshold": self.depth_escalate,
            "parallel_escalate_threshold": self.parallel_escalate,
        }

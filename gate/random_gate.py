"""
gate/random_gate.py
===================
Random gate baseline — coin-flip routing at a fixed escalation rate.

Purpose: Isolates whether gate decisions are informative vs. just
         matching the average escalation frequency.

If GateOrchestra doesn't significantly outperform RandomGate at the same
escalation rate, the learned features aren't helping.

Person 3 owns this file.
"""

from __future__ import annotations

import logging
import random

from shared.schemas import GateDecision, GateFeatures

logger = logging.getLogger(__name__)


class RandomGate:
    """
    Coin-flip gate at a configurable escalation rate.

    The gate is seeded for reproducibility — same seed = same decisions.

    Usage::

        gate = RandomGate(escalation_rate=0.33, seed=42)
        decision = gate.predict(features, k=3, probe_tokens=120)

    Matching GateOrchestra's escalation rate::

        # After running GateOrchestra on val set:
        rate = accountant.get_escalation_rate("GateOrchestra")
        matched_gate = RandomGate(escalation_rate=rate)
    """

    def __init__(self, escalation_rate: float = 0.5, seed: int = 42) -> None:
        if not 0.0 <= escalation_rate <= 1.0:
            raise ValueError(f"escalation_rate must be in [0, 1], got {escalation_rate}")
        self.escalation_rate = escalation_rate
        self.seed = seed
        self._rng = random.Random(seed)
        self._call_count: int = 0

    def predict(
        self,
        features: GateFeatures,
        k: int,
        probe_tokens: int,
    ) -> GateDecision:
        """Randomly route the task based on escalation_rate.

        Args:
            features:     GateFeatures (task_id used for logging only).
            k:            Token budget multiplier.
            probe_tokens: Probe token count.
        """
        self._call_count += 1
        escalate = self._rng.random() < self.escalation_rate
        decision = "ESCALATE" if escalate else "STOP"

        logger.debug(
            f"[RandomGate] task={features.task_id} "
            f"roll={self._call_count} decision={decision}"
        )

        return GateDecision(
            task_id=features.task_id,
            decision=decision,  # type: ignore[arg-type]
            confidence=0.5,  # Always 50% — by definition uncertain
            token_budget_cap=k * probe_tokens if decision == "ESCALATE" else None,
            gate_type="random",
        )

    def reset(self, seed: int | None = None) -> None:
        """Reset the RNG (optionally with a new seed)."""
        self._rng = random.Random(seed if seed is not None else self.seed)
        self._call_count = 0

    def observed_escalation_rate(self, n: int = 10_000) -> float:
        """Simulate n calls and return the empirical escalation rate.

        Useful for verifying the gate matches the target rate.
        """
        gate = RandomGate(self.escalation_rate, self.seed)
        dummy_features = GateFeatures(
            task_id="test",
            consistency_score=0.5,
            probe_tokens=100,
            question_word_count=10,
            entity_count=2,
            clause_count=1,
            has_context=False,
        )
        escalations = sum(
            1 for _ in range(n) if gate.predict(dummy_features, k=3, probe_tokens=100).decision == "ESCALATE"
        )
        return escalations / n

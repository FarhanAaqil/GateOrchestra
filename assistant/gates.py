"""
assistant/gates.py
==================
The gates the assistant can route with, and their decisions.

  learned        the gate calibrated by the final benchmark (configs/models/)
  rule_based     consistency/entity/depth thresholds
  always_stop    never escalate (cheapest; one model call per answer)
  always_escalate escalate everything (most thorough, most expensive)

Every gate's decision is cheap to compute, so all of them are evaluated on
every message; only the selected gate's decision is acted on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from gate.classifier import GateClassifier, load_benchmark_gate
from gate.rule_based_gate import RuleBasedGate
from shared.config import K_DEFAULT, MODELS_DIR
from shared.schemas import GateFeatures

logger = logging.getLogger(__name__)

GATES = ("learned", "rule_based", "always_stop", "always_escalate")


@dataclass
class GateVerdict:
    gate: str
    decision: str  # STOP | ESCALATE
    confidence: float | None = None


class GateSet:
    def __init__(self, learned: GateClassifier | None = None) -> None:
        if learned is None:
            found = load_benchmark_gate(MODELS_DIR)
            learned = found[0] if found else None
        self.learned = learned
        self.rule = RuleBasedGate()

    def available(self) -> list[str]:
        return [g for g in GATES if g != "learned" or self.learned is not None]

    def decide_all(self, features: GateFeatures) -> dict[str, GateVerdict]:
        verdicts: dict[str, GateVerdict] = {
            "always_stop": GateVerdict("always_stop", "STOP", 1.0),
            "always_escalate": GateVerdict("always_escalate", "ESCALATE", 1.0),
        }
        rule = self.rule.predict(features, k=K_DEFAULT, probe_tokens=features.probe_tokens)
        verdicts["rule_based"] = GateVerdict("rule_based", rule.decision, None)
        if self.learned is not None:
            try:
                d = self.learned.predict(features, k=K_DEFAULT, probe_tokens=features.probe_tokens)
                verdicts["learned"] = GateVerdict("learned", d.decision, float(d.confidence))
            except Exception as err:  # a broken model must not break the chat
                logger.warning(f"learned gate failed: {err}")
        return verdicts

    def default_gate(self) -> str:
        return "learned" if self.learned is not None else "rule_based"

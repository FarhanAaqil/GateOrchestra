"""
assistant/selector.py
=====================
Automatic gate selection: use the cheapest gate whose accuracy is within a
margin of the best.

Cost. Every chat turn records the verdict of every gate, not only the one in
charge, and how many tokens the turn spent. From recent turns we know the
average cost of a STOP turn and of an ESCALATE turn, and how often each gate
says ESCALATE, so every gate gets an expected cost per message, including the
gates that were not used.

Accuracy. A thumbs up or down on an answer counts for every gate whose verdict
matched the route that produced it: those gates would have produced the same
answer. Accuracy is the mean of Beta(1 + up, 1 + down), so a gate with no
ratings starts at 0.5 with wide uncertainty.

Choice. Gates whose accuracy is within ``margin`` of the best are eligible;
the cheapest eligible gate wins. With probability ``explore`` another
available gate is used instead, so every gate keeps receiving feedback.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from assistant.store import Store

WINDOW = 500  # recent turns used for cost estimates
DEFAULT_COST = {"STOP": 1200.0, "ESCALATE": 3500.0}  # before any turns are logged


@dataclass
class GateScore:
    gate: str
    escalate_rate: float
    expected_tokens: float
    up: int
    down: int
    accuracy: float
    eligible: bool = False
    chosen: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class Selection:
    gate: str
    reason: str
    scores: list[GateScore] = field(default_factory=list)
    explored: bool = False


def decision_costs(turns: list[dict[str, Any]]) -> dict[str, float]:
    """Average tokens of STOP and ESCALATE turns (defaults until both are seen)."""
    costs = dict(DEFAULT_COST)
    for decision in ("STOP", "ESCALATE"):
        spent = [
            t["total_tokens"]
            for t in turns
            if t.get("decision") == decision and t.get("total_tokens")
        ]
        if spent:
            costs[decision] = sum(spent) / len(spent)
    return costs


def escalate_rates(turns: list[dict[str, Any]], gates: list[str]) -> dict[str, float]:
    """How often each gate said ESCALATE over turns where it gave a verdict."""
    rates = {}
    for gate in gates:
        votes = [t["verdicts"][gate] for t in turns if gate in (t.get("verdicts") or {})]
        if votes:
            rates[gate] = sum(v == "ESCALATE" for v in votes) / len(votes)
        else:
            rates[gate] = {"always_stop": 0.0, "always_escalate": 1.0}.get(gate, 0.5)
    return rates


def feedback_counts(rated: list[dict[str, Any]], gates: list[str]) -> dict[str, dict[str, int]]:
    """Ratings credited to every gate whose verdict matched the route taken."""
    counts = {g: {"up": 0, "down": 0} for g in gates}
    for item in rated:
        decision = item.get("decision")
        for gate, verdict in (item.get("verdicts") or {}).items():
            if gate in counts and verdict == decision:
                counts[gate]["up" if item["rating"] > 0 else "down"] += 1
    return counts


class GateSelector:
    def __init__(
        self,
        store: Store,
        margin: float = 0.02,
        explore: float = 0.1,
        rng: random.Random | None = None,
    ) -> None:
        self.store = store
        self.margin = margin
        self.explore = explore
        self.rng = rng or random.Random()

    def scores(self, gates: list[str]) -> list[GateScore]:
        turns = [t for t in self.store.events("chat", limit=WINDOW) if t.get("gate") != "triage"]
        costs = decision_costs(turns)
        rates = escalate_rates(turns, gates)
        counts = feedback_counts(self.store.rated_turns(), gates)
        out = []
        for gate in gates:
            up, down = counts[gate]["up"], counts[gate]["down"]
            rate = rates[gate]
            out.append(
                GateScore(
                    gate=gate,
                    escalate_rate=rate,
                    expected_tokens=rate * costs["ESCALATE"] + (1 - rate) * costs["STOP"],
                    up=up,
                    down=down,
                    accuracy=(1 + up) / (2 + up + down),
                )
            )
        return out

    def choose(self, gates: list[str]) -> Selection:
        scores = self.scores(gates)
        if not scores:
            raise ValueError("no gates available")
        best = max(s.accuracy for s in scores)
        for s in scores:
            s.eligible = s.accuracy >= best - self.margin
        winner = min((s for s in scores if s.eligible), key=lambda s: (s.expected_tokens, s.gate))
        reason = (
            f"cheapest gate within {self.margin:.0%} of the best accuracy "
            f"(≈{winner.expected_tokens:,.0f} tokens, {winner.accuracy:.0%} rated useful)"
        )
        explored = False
        others = [s for s in scores if s.gate != winner.gate]
        if others and self.rng.random() < self.explore:
            winner = self.rng.choice(others)
            reason = f"exploring {winner.gate} to keep its feedback current"
            explored = True
        winner.chosen = True
        return Selection(gate=winner.gate, reason=reason, scores=scores, explored=explored)

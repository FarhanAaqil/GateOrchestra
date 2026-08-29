"""
shared/token_logger.py
======================
Thread-safe Token Accountant.

RULE: Every LLM call in the codebase MUST log tokens here.
      No raw API calls without wrapping them in TokenAccountant.log().

Output format agreed with Person 4 (evaluation/) on Day 5.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict


@dataclass
class _SpendRecord:
    """Internal record for a single logged LLM call."""

    task_id: str
    method: str
    stage: str  # "probe", "mas", "gate_feature" etc.
    tokens: int
    path: str  # routing path taken: "STOP" | "ESCALATE" | "N/A"


class _RecordDict(TypedDict):
    """Typed dict matching the JSON output format for a single spend record."""

    task_id: str
    method: str
    stage: str
    tokens: int
    path: str


@dataclass
class TokenAccountant:
    """
    Tracks token spending across all pipeline stages.

    Usage::

        accountant = TokenAccountant()
        accountant.log("task_001", method="GateOrchestra", stage="probe", tokens=120, path="N/A")
        accountant.log("task_001", method="GateOrchestra", stage="mas",   tokens=480, path="ESCALATE")

        print(accountant.get_spend("task_001"))
        # {'probe': 120, 'mas': 480, 'total': 600}

        accountant.save_to_json("logs/token_log.json")
    """

    _records: list[_SpendRecord] = field(default_factory=list, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def log(
        self,
        task_id: str,
        method: str,
        stage: str,
        tokens: int,
        path: str = "N/A",
    ) -> None:
        """Record a token spend event.

        Args:
            task_id: Matches Task.task_id.
            method:  Pipeline method name (e.g. "GateOrchestra", "CoT-SC-only").
            stage:   Pipeline stage (e.g. "probe", "mas").
            tokens:  Number of tokens consumed.
            path:    Gate routing path taken ("STOP", "ESCALATE", or "N/A").
        """
        if tokens < 0:
            raise ValueError(f"tokens must be ≥ 0, got {tokens}")
        record = _SpendRecord(task_id=task_id, method=method, stage=stage, tokens=tokens, path=path)
        with self._lock:
            self._records.append(record)

    def get_spend(self, task_id: str, method: str | None = None) -> dict[str, int]:
        """Return aggregated token spend for a task (and optionally a specific method).

        Returns:
            Dict with per-stage token counts plus 'total'.
            Example: {'probe': 120, 'mas': 480, 'total': 600}
        """
        with self._lock:
            records = [
                r
                for r in self._records
                if r.task_id == task_id and (method is None or r.method == method)
            ]

        by_stage: dict[str, int] = defaultdict(int)
        for r in records:
            by_stage[r.stage] += r.tokens

        by_stage["total"] = sum(by_stage.values())
        return dict(by_stage)

    def get_total_by_method(self) -> dict[str, int]:
        """Return total tokens spent per method across all tasks."""
        with self._lock:
            records = list(self._records)

        totals: dict[str, int] = defaultdict(int)
        for r in records:
            totals[r.method] += r.tokens
        return dict(totals)

    def get_escalation_rate(self, method: str = "GateOrchestra") -> float:
        """Fraction of tasks that were ESCALATED by the gate."""
        with self._lock:
            relevant = [r for r in self._records if r.method == method]

        if not relevant:
            return 0.0

        task_paths: dict[str, str] = {}
        for r in relevant:
            if r.path in ("STOP", "ESCALATE"):
                task_paths[r.task_id] = r.path  # last path wins (should be same)

        if not task_paths:
            return 0.0

        n_escalated = sum(1 for p in task_paths.values() if p == "ESCALATE")
        return n_escalated / len(task_paths)

    def save_to_json(self, path: str | Path) -> None:
        """Persist all records to a JSON file.

        Output format (agreed with Person 4)::

            {
              "records": [
                {
                  "task_id": "hotpot_001",
                  "method": "GateOrchestra",
                  "stage": "probe",
                  "tokens": 120,
                  "path": "STOP"
                },
                ...
              ],
              "summary": {
                "GateOrchestra": {"total": 600, "escalation_rate": 0.33},
                ...
              }
            }
        """
        with self._lock:
            records_data: list[dict[str, Any]] = [
                {
                    "task_id": r.task_id,
                    "method": r.method,
                    "stage": r.stage,
                    "tokens": r.tokens,
                    "path": r.path,
                }
                for r in self._records
            ]

        methods: set[str] = {str(r["method"]) for r in records_data}
        summary: dict[str, Any] = {}
        for m in methods:
            total = sum(int(r["tokens"]) for r in records_data if r["method"] == m)
            summary[m] = {
                "total_tokens": total,
                "escalation_rate": self.get_escalation_rate(m),
            }

        output = {"records": records_data, "summary": summary}
        Path(path).write_text(json.dumps(output, indent=2), encoding="utf-8")

    @classmethod
    def load_from_json(cls, path: str | Path) -> TokenAccountant:
        """Reload a TokenAccountant from a saved JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        accountant = cls()
        for rec in data["records"]:
            accountant.log(
                task_id=rec["task_id"],
                method=rec["method"],
                stage=rec["stage"],
                tokens=rec["tokens"],
                path=rec["path"],
            )
        return accountant

    def reset(self) -> None:
        """Clear all records (use between experiment runs)."""
        with self._lock:
            self._records.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def __repr__(self) -> str:
        return f"TokenAccountant(records={len(self)}, methods={list(self.get_total_by_method())})"


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton (optional convenience import)
# ─────────────────────────────────────────────────────────────────────────────

_global_accountant: TokenAccountant | None = None


def get_global_accountant() -> TokenAccountant:
    """Get (or create) the module-level singleton TokenAccountant."""
    global _global_accountant
    if _global_accountant is None:
        _global_accountant = TokenAccountant()
    return _global_accountant


def reset_global_accountant() -> None:
    """Reset the global accountant (call between experiment runs)."""
    global _global_accountant
    _global_accountant = None

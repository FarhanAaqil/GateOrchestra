"""
evaluation/results_detail.py
============================
Derived views of master_results.json for the dashboard: per-task rows joined
with dataset labels, Depth x Parallel cells, escalation precision/recall
against an oracle, and the validation k-sweep recorded during calibration.

Everything here is computed from recorded results; nothing is estimated.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

GATED_METHODS = ("RandomGate", "RuleBasedGate", "GateOrchestra")


def _per_task_rows(master: dict[str, Any], labels: dict[str, dict]) -> list[dict[str, Any]]:
    rows = []
    for run in master.get("per_seed", []):
        for method, records in (run.get("per_task") or {}).items():
            for record in records:
                label = labels.get(record["task_id"], {})
                rows.append(
                    {
                        "seed": run.get("seed"),
                        "method": method,
                        "task_id": record["task_id"],
                        "is_correct": record.get("is_correct"),
                        "tokens_spent": record.get("tokens_spent"),
                        "decision": record.get("decision"),
                        "depth": label.get("depth_score"),
                        "parallel": label.get("parallel_score"),
                        "source": label.get("source_dataset"),
                    }
                )
    return rows


def _axis_cells(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["depth"], row["parallel"])].append(row)
    cells = []
    for (method, depth, parallel), members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        n = len(members)
        escalated = sum(1 for m in members if m["decision"] == "ESCALATE")
        cells.append(
            {
                "method": method,
                "depth": depth,
                "parallel": parallel,
                "n": n,
                "accuracy": sum(1 for m in members if m["is_correct"] is True) / n,
                "escalation_rate": escalated / n,
                "avg_tokens": sum(m["tokens_spent"] or 0 for m in members) / n,
            }
        )
    return cells


def _escalation_quality(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Precision/recall of ESCALATE against the oracle "MAS right, CoT-SC wrong"."""
    by_key = {(r["seed"], r["method"], r["task_id"]): r for r in rows}
    out = []
    for method in GATED_METHODS:
        tp = fp = fn = 0
        for (seed, m, task_id), row in by_key.items():
            if m != method:
                continue
            cot = by_key.get((seed, "CoT-SC-only", task_id))
            mas = by_key.get((seed, "Always-MAS", task_id))
            if cot is None or mas is None:
                continue
            needed = mas["is_correct"] is True and cot["is_correct"] is not True
            escalated = row["decision"] == "ESCALATE"
            tp += escalated and needed
            fp += escalated and not needed
            fn += needed and not escalated
        if tp + fp + fn == 0 and not any(r["method"] == method for r in rows):
            continue
        out.append(
            {
                "method": method,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
            }
        )
    return out


def _k_sweep(master: dict[str, Any]) -> list[dict[str, Any]]:
    """Validation metrics per k for the classifier and tau each seed selected."""
    points = []
    for run in master.get("per_seed", []):
        gate = run.get("gate") or {}
        for candidate in run.get("calibration") or []:
            if candidate.get("classifier") != gate.get("classifier"):
                continue
            if candidate.get("tau_acc") != gate.get("tau_acc"):
                continue
            points.append(
                {
                    "seed": run.get("seed"),
                    "k": candidate.get("k"),
                    "accuracy": candidate.get("accuracy"),
                    "avg_tokens": candidate.get("avg_tokens"),
                    "token_savings_pct": candidate.get("token_savings_pct"),
                    "escalation_rate": candidate.get("escalation_rate"),
                    "selected": candidate.get("k") == run.get("k"),
                }
            )
    return sorted(points, key=lambda p: (p["k"] or 0, p["seed"] or 0))


def build_results_detail(master: dict[str, Any], labels: dict[str, dict]) -> dict[str, Any]:
    """Assemble every derived view from one master_results payload."""
    rows = _per_task_rows(master, labels)
    return {
        "per_task": rows,
        "axis_cells": _axis_cells(rows),
        "escalation_quality": _escalation_quality(rows),
        "k_sweep": _k_sweep(master),
        "gates": [
            {"seed": run.get("seed"), **(run.get("gate") or {})}
            for run in master.get("per_seed", [])
        ],
    }

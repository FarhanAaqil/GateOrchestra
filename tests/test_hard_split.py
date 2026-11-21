"""The harder-task experiment: its task file and analysis (no model calls)."""

from __future__ import annotations

import pytest

from scripts import hard_split
from shared.schemas import Task


def test_the_hard_tasks_are_valid_and_separate_from_the_benchmark() -> None:
    tasks = hard_split.load_tasks()
    assert len(tasks) == 48 and len({t["task_id"] for t in tasks}) == 48
    for spec in tasks:
        task = Task(**{k: v for k, v in spec.items() if k in Task.model_fields})
        assert task.ground_truth and task.depth_score and task.depth_score >= 3
        assert spec["task_id"].startswith("hard_")
    from shared.data_loader import load_split

    benchmark_ids = {t.task_id for s in ("train", "val", "test") for t in load_split(s)}
    assert not benchmark_ids & {t["task_id"] for t in tasks}


def row(
    tid: str, cot: bool, mas: bool, concise: bool, gate: str = "STOP", consistency: float = 1.0
) -> dict:
    return {
        "task_id": tid,
        "source": "hard_arith",
        "consistency": consistency,
        "cot_correct": cot,
        "cot_tokens": 1000,
        "gate_decision": gate,
        "mas_correct": mas,
        "mas_tokens": 2000,
        "mas_concise_correct": concise,
        "mas_concise_tokens": 2050,
    }


def test_analysis_scores_every_method_from_the_same_runs() -> None:
    rows = [
        row("a", cot=False, mas=True, concise=True, gate="ESCALATE", consistency=1 / 3),
        row("b", cot=True, mas=False, concise=True),
        row("c", cot=True, mas=True, concise=True),
        row("d", cot=True, mas=True, concise=True),
        {"task_id": "e", "source": "hard_arith", "error": "RateLimit"},
    ]
    s = hard_split.analyze(rows)
    m = s["methods"]
    assert s["n_scored"] == 4
    assert m["CoT-SC only"]["accuracy"] == 75.0 and m["Always-MAS"]["accuracy"] == 75.0
    assert m["Always-MAS + concise answer"]["accuracy"] == 100.0
    assert m["GateOrchestra (learned gate)"]["accuracy"] == 100.0
    assert m["GateOrchestra (learned gate)"]["avg_tokens"] == pytest.approx((3000 + 1000 * 3) / 4)
    assert m["Oracle"]["escalation_rate"] == 25.0
    assert (
        s["by_source"]["hard_arith"]["mas_helps"] == 1
        and s["by_source"]["hard_arith"]["mas_hurts"] == 1
    )
    report = hard_split.write_report(s)
    assert "| Always-MAS + concise answer | 100.0% |" in report

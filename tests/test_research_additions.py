"""The threshold-baseline analysis and the second-review tooling."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from scripts import second_review, threshold_baseline

ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / "results" / "real" / "final" / "master_results.json"


def row(
    split: str, consistency: float, cot: bool, mas: bool, gate: str | None = None, **kw
) -> dict:
    base = {
        "seed": 42,
        "split": split,
        "task_id": kw.pop("task_id", "t"),
        "repeat": 0,
        "consistency": consistency,
        "cot_correct": cot,
        "cot_tokens": 100,
        "mas_correct": mas,
        "mas_tokens": 300,
    }
    if gate is not None:
        base.update(
            gate_decision=gate,
            gate_correct=mas if gate == "ESCALATE" else cot,
            gate_tokens=400 if gate == "ESCALATE" else 100,
        )
    return {**base, **kw}


def test_escalation_pays_for_the_probe_and_the_mas() -> None:
    r = row("test", 1 / 3, False, True)
    assert threshold_baseline.outcome(r, escalate=False) == (False, 100)
    assert threshold_baseline.outcome(r, escalate=True) == (True, 400)


def test_the_threshold_is_chosen_on_validation_by_the_gates_rule() -> None:
    # Escalating on disagreement fixes many validation answers: t=1 wins clearly.
    val = [row("val", 2 / 3, False, True) for _ in range(10)] + [
        row("val", 1.0, True, True) for _ in range(10)
    ]
    chosen, scores = threshold_baseline.choose_threshold(val)
    assert chosen == "t=1" and scores["t=1"]["accuracy"] == 100.0
    # When escalating barely helps, the cheapest option within epsilon wins.
    val = [row("val", 2 / 3, True, True) for _ in range(40)]
    assert threshold_baseline.choose_threshold(val)[0] == "never"


def test_exact_mcnemar() -> None:
    assert threshold_baseline.mcnemar_exact(0, 0) == 1.0
    assert threshold_baseline.mcnemar_exact(2, 0) == 0.5
    assert threshold_baseline.mcnemar_exact(10, 0) == pytest.approx(0.002, abs=1e-3)


def test_analysis_on_a_small_table() -> None:
    rows = [row("val", 1.0, True, True, task_id=f"v{i}") for i in range(8)]
    rows += [
        row("test", 1.0, True, False, "STOP", task_id="a"),
        row("test", 1 / 3, False, True, "ESCALATE", task_id="b"),
        row("test", 2 / 3, True, True, "STOP", task_id="c"),
    ]
    summary = threshold_baseline.analyze(rows)
    test = summary["test"]
    assert summary["chosen_threshold"] == "never"
    assert test["CoT-SC-only (never escalate)"]["accuracy"] == pytest.approx(66.7)
    assert test["GateOrchestra (live, as recorded)"]["accuracy"] == 100.0
    assert test["Oracle (escalate only when it helps)"]["escalation_rate"] == pytest.approx(33.3)
    report = threshold_baseline.write_report(summary)
    assert "one-line rule" in report and "| GateOrchestra (live, as recorded) |" in report


def test_the_committed_table_reproduces_the_recorded_benchmark() -> None:
    """The extracted per-task table must give exactly the numbers master_results.json reports."""
    rows = json.loads(threshold_baseline.OUTCOMES.read_text(encoding="utf-8"))
    test = [r for r in rows if r["split"] == "test"]
    master = json.loads(MASTER.read_text(encoding="utf-8"))
    recorded = {m["method"]: m for m in master["aggregated"]}
    assert len(test) == 192
    gate = recorded["GateOrchestra"]
    live_acc = 100 * sum(r["gate_correct"] for r in test) / len(test)
    assert live_acc == pytest.approx(gate["accuracy_mean"], abs=0.6)
    cot = recorded["CoT-SC-only"]
    assert 100 * sum(r["cot_correct"] for r in test) / len(test) == pytest.approx(
        cot["accuracy_mean"], abs=0.6
    )
    summary = json.loads(threshold_baseline.SUMMARY.read_text(encoding="utf-8"))
    assert summary == threshold_baseline.analyze(rows)  # the committed summary is up to date


def test_review_sheet_lists_every_assistant_reviewed_task(tmp_path: Path) -> None:
    out = tmp_path / "sheet.csv"
    n = second_review.write_sheet(out)
    expected = second_review.tasks_to_review()
    assert n == len(expected) == 64
    with out.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["task_id"] for r in rows] == sorted(expected)
    assert all(r["question"] and r["gold_answer"] and r["verdict"] == "" for r in rows)


def test_importing_a_review_checks_it_and_records_it(tmp_path: Path) -> None:
    review = tmp_path / "REVIEW.json"
    shutil.copy(second_review.REVIEW, review)
    sheet = tmp_path / "sheet.csv"
    second_review.write_sheet(sheet, review)
    expected = second_review.tasks_to_review(review)

    with pytest.raises(ValueError, match="not finished"):
        second_review.read_verdicts(sheet, expected)

    rows = list(csv.DictReader(sheet.open(encoding="utf-8")))
    for r in rows:
        r["verdict"] = "ok"
    rows[0]["verdict"] = "wrong"
    with sheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=second_review.COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="correct_answer is needed"):
        second_review.read_verdicts(sheet, expected)

    rows[0]["correct_answer"] = "$27.00"
    rows[0]["note"] = "change is miscounted"
    with sheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=second_review.COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    verdicts = second_review.read_verdicts(sheet, expected)
    with pytest.raises(ValueError, match="by a person"):
        second_review.record_review("Claude", verdicts, review)
    summary = second_review.record_review("A Teammate", verdicts, review, today="2026-11-21")
    assert summary["counts"] == {"ambiguous": 0, "ok": 63, "wrong": 1}
    assert summary["flagged"][0]["correct_answer"] == "$27.00"
    stored = json.loads(review.read_text(encoding="utf-8"))
    assert stored["second_review"]["reviewed_by"] == "A Teammate"
    assert stored["assistant_reviewed"]["task_ids"] == expected  # nothing else changed


def test_the_api_serves_the_threshold_comparison(service) -> None:
    from fastapi.testclient import TestClient

    from api.main import app

    data = TestClient(app).get("/results/threshold").json()
    assert data["available"] is True and data["chosen_threshold"] in ("never", "t=2/3", "t=1")
    assert "GateOrchestra (live, as recorded)" in data["test"]

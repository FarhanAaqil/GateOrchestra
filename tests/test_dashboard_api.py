"""
tests/test_dashboard_api.py
===========================
Additive API surface used by the dashboard: probe details and sub-agent
traces on /run, the optional threshold, /gate/importance and
/results/detail. Existing response fields are covered elsewhere.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

import api.main as api_main
from agents.orchestrator import MASOrchestrator
from evaluation.results_detail import build_results_detail
from gate.classifier import GBTGate
from shared.schemas import GateFeatures, Task


@pytest.fixture
def client() -> TestClient:
    return TestClient(api_main.app)


def _features(task_id: str, consistency: float, depth: float) -> GateFeatures:
    return GateFeatures(
        task_id=task_id,
        consistency_score=consistency,
        probe_tokens=300,
        question_word_count=12,
        entity_count=2,
        clause_count=1,
        has_context=False,
        estimated_depth=depth,
        estimated_parallel=1.0,
    )


def test_sub_agent_trace_records_every_step() -> None:
    def caller(prompt: str, temp: float, budget: int) -> tuple[str, int]:
        return "Some reasoning. Final Answer: 42", 50

    orch = MASOrchestrator(default_strategy="reflexion", llm_caller=caller)
    orch.run(Task(task_id="t", question="What is 6 x 7?"), token_budget=2000)
    assert [step["step"] for step in orch._last_trace] == ["Draft", "Reflect", "Refine"]
    assert all(step["tokens"] == 50 for step in orch._last_trace)


def test_run_returns_probe_details_and_empty_trace_when_stopping(client: TestClient) -> None:
    payload = {
        "task_id": "dash-1",
        "question": "What is the capital of France?",
        "method": "CoT-SC",
        "force_simulation": True,
    }
    data = client.post("/run", json=payload).json()
    assert 0.0 <= data["probe"]["consistency_score"] <= 1.0
    assert len(data["probe"]["samples"]) >= 1
    assert data["trace"] == []


def test_threshold_is_validated(client: TestClient) -> None:
    payload = {
        "task_id": "dash-2",
        "question": "Q?",
        "method": "RuleBasedGate",
        "force_simulation": True,
        "threshold": 1.5,
    }
    assert client.post("/run", json=payload).status_code == 422


def test_gate_importance_reports_gbt_importances(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    gate = GBTGate(n_estimators=10)
    gate.train(
        [_features(f"t{i}", 1.0 if i % 2 else 0.3, float(i % 4)) for i in range(12)],
        ["STOP" if i % 2 else "ESCALATE" for i in range(12)],
    )
    path = tmp_path / "best_gate_seed_7.pkl"
    gate.save(path)
    monkeypatch.setattr(api_main, "_CACHED_LEARNED_GATE", None)
    monkeypatch.setattr(api_main, "_CACHED_GATE_INFO", None)
    monkeypatch.setattr(api_main, "_CACHED_GATE_KEY", None)
    monkeypatch.setattr(api_main, "_gate_model_candidates", lambda: [path])

    data = client.get("/gate/importance").json()
    assert data["available"] is True
    assert data["gate"]["seed"] == 7
    assert len(data["importances"]) == 8
    assert data["importances"][0]["feature"] == "consistency_score"
    assert sum(row["importance"] for row in data["importances"]) == pytest.approx(1.0)


def test_gate_importance_without_gate(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(api_main, "_CACHED_LEARNED_GATE", None)
    monkeypatch.setattr(api_main, "_CACHED_GATE_KEY", None)
    monkeypatch.setattr(api_main, "_gate_model_candidates", lambda: [tmp_path / "none.pkl"])
    assert client.get("/gate/importance").json()["available"] is False


def test_results_detail_without_results(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(api_main, "MASTER_RESULTS_PATH", tmp_path / "missing.json")
    data = client.get("/results/detail").json()
    assert data["status"] == "no_results"
    assert data["per_task"] == []


def _master() -> dict:
    per_task = {
        "CoT-SC-only": [
            {"task_id": "a", "is_correct": False, "tokens_spent": 100, "decision": None},
            {"task_id": "b", "is_correct": True, "tokens_spent": 100, "decision": None},
        ],
        "Always-MAS": [
            {"task_id": "a", "is_correct": True, "tokens_spent": 400, "decision": None},
            {"task_id": "b", "is_correct": True, "tokens_spent": 400, "decision": None},
        ],
        "GateOrchestra": [
            {"task_id": "a", "is_correct": True, "tokens_spent": 500, "decision": "ESCALATE"},
            {"task_id": "b", "is_correct": True, "tokens_spent": 100, "decision": "ESCALATE"},
        ],
    }
    calibration = [
        {
            "classifier": "gbt",
            "tau_acc": 0.05,
            "k": k,
            "accuracy": 0.5 + k / 100,
            "avg_tokens": 100.0 * k,
            "token_savings_pct": 10.0,
            "escalation_rate": 0.2,
        }
        for k in (2, 3, 5)
    ] + [{"classifier": "logreg", "tau_acc": 0.05, "k": 3, "accuracy": 0.1}]
    return {
        "per_seed": [
            {
                "seed": 42,
                "k": 3,
                "gate": {"classifier": "gbt", "tau_acc": 0.05},
                "calibration": calibration,
                "per_task": per_task,
            }
        ]
    }


def test_build_results_detail() -> None:
    labels = {
        "a": {"depth_score": 4, "parallel_score": 1, "source_dataset": "musique_style"},
        "b": {"depth_score": 2, "parallel_score": 1, "source_dataset": "template_arithmetic"},
    }
    detail = build_results_detail(_master(), labels)

    assert len(detail["per_task"]) == 6
    gate_row = next(r for r in detail["per_task"] if r["method"] == "GateOrchestra")
    assert (gate_row["depth"], gate_row["parallel"]) == (4, 1)

    quality = {q["method"]: q for q in detail["escalation_quality"]}["GateOrchestra"]
    # Task a needed MAS (escalated: TP); task b did not (escalated: FP).
    assert (quality["true_positive"], quality["false_positive"], quality["false_negative"]) == (
        1,
        1,
        0,
    )
    assert quality["precision"] == 0.5
    assert quality["recall"] == 1.0

    assert [p["k"] for p in detail["k_sweep"]] == [2, 3, 5]
    assert [p["selected"] for p in detail["k_sweep"]] == [False, True, False]

    cell = next(
        c for c in detail["axis_cells"] if c["method"] == "GateOrchestra" and c["depth"] == 4
    )
    assert cell["escalation_rate"] == 1.0


def test_results_detail_endpoint_reads_master(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    path = tmp_path / "master_results.json"
    path.write_text(json.dumps(_master()), encoding="utf-8")
    monkeypatch.setattr(api_main, "MASTER_RESULTS_PATH", path)
    data = client.get("/results/detail").json()
    assert data["status"] == "unverified"  # synthetic file lacks seeds, hashes, etc.
    assert len(data["per_task"]) == 6

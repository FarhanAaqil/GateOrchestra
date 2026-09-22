"""tests/test_run_experiments.py
==============================
Unit and integration tests for the unified final experiments runner
(Phase 1 of GateOrchestra).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gate.classifier import GBTGate
from gate.feature_extractor import extract_features
from scripts.run_experiments import (
    AblatedGateWrapper,
    compute_gate_classification_metrics,
    make_deterministic_simulated_mas,
    run_experiments,
    run_feature_ablations,
)
from shared.config import PROBE_TOKEN_BUDGET
from shared.data_loader import load_split
from shared.schemas import EvalResult, GateDecision, ProbeResult, Task


@pytest.fixture
def sample_tasks() -> list[Task]:
    """Load a small slice of valid tasks for testing."""
    tasks = load_split("test")
    return tasks[:4]


@pytest.fixture
def trained_gbt_gate() -> GBTGate:
    """Train a quick GBT gate on train split."""
    train_tasks = load_split("train")[:20]
    labeled_features = []
    labels = []
    for t in train_tasks:
        probe = ProbeResult(
            task_id=t.task_id,
            answer=t.ground_truth or "A",
            confidence_score=0.9,
            consistency_score=0.95,
            tokens_used=PROBE_TOKEN_BUDGET,
            n_samples=3,
        )
        feat = extract_features(t, probe)
        labeled_features.append(feat)
        labels.append("STOP" if (t.depth_score or 1) <= 2 else "ESCALATE")

    gate = GBTGate()
    gate.train(labeled_features, labels)
    return gate


def test_ablated_gate_wrapper(trained_gbt_gate: GBTGate, sample_tasks: list[Task]) -> None:
    """Verify AblatedGateWrapper masks the specified feature correctly."""
    wrapper = AblatedGateWrapper(trained_gbt_gate, mask_indices=[0], name_suffix="no_consistency")
    assert wrapper._is_trained is True

    # Test train pass-through
    wrapper.train([], [])

    task = sample_tasks[0]
    probe = ProbeResult(
        task_id=task.task_id,
        answer="A",
        confidence_score=0.8,
        consistency_score=0.9,
        tokens_used=100,
        n_samples=3,
    )
    features = extract_features(task, probe)
    decision = wrapper.predict(features, k=3, probe_tokens=100)
    assert decision.decision in ("STOP", "ESCALATE")
    if decision.decision == "ESCALATE":
        assert decision.token_budget_cap is not None
    else:
        assert decision.token_budget_cap is None


def test_deterministic_simulated_mas(sample_tasks: list[Task]) -> None:
    """Verify simulated MAS returns deterministic answers and valid token costs."""
    mas_fn1 = make_deterministic_simulated_mas(seed=42)
    mas_fn2 = make_deterministic_simulated_mas(seed=42)
    task = sample_tasks[0]
    ans1, tokens1 = mas_fn1(task, 1500)
    ans2, tokens2 = mas_fn2(task, 1500)

    assert ans1 == ans2
    assert tokens1 == tokens2
    assert tokens1 > 0
    assert isinstance(ans1, str)


def test_feature_ablations(trained_gbt_gate: GBTGate, sample_tasks: list[Task]) -> None:
    """Verify leave-one-feature-out ablation pipeline executes and measures deltas."""
    mas_fn = make_deterministic_simulated_mas(seed=42)

    def probe_fn(task: Task) -> ProbeResult:
        return ProbeResult(
            task_id=task.task_id,
            answer=task.ground_truth or "ans",
            confidence_score=0.9,
            consistency_score=0.95,
            tokens_used=200,
            n_samples=3,
        )

    # Empty baseline results list is allowed
    ablations = run_feature_ablations(
        tasks=sample_tasks,
        base_gate=trained_gbt_gate,
        probe_fn=probe_fn,
        mas_fn=mas_fn,
        baseline_results=[],
        k=3,
    )
    assert len(ablations) == 8
    for ab in ablations:
        assert "ablated_feature" in ab
        assert "accuracy" in ab
        assert "avg_tokens" in ab


def test_gate_classification_metrics() -> None:
    """Verify compute_gate_classification_metrics precision, recall, and F1."""
    eval_results = [
        EvalResult(
            task_id="t1",
            method="GateOrchestra",
            predicted_answer="ans1",
            is_correct=True,
            tokens_spent=100,
            latency_ms=10.0,
            gate_decision=GateDecision(
                task_id="t1", decision="ESCALATE", confidence=0.9, token_budget_cap=300
            ),
        ),
        EvalResult(
            task_id="t2",
            method="GateOrchestra",
            predicted_answer="ans2",
            is_correct=True,
            tokens_spent=50,
            latency_ms=5.0,
            gate_decision=GateDecision(task_id="t2", decision="STOP", confidence=0.8),
        ),
    ]
    optimal_labels = {"t1": "ESCALATE", "t2": "STOP"}
    metrics = compute_gate_classification_metrics(eval_results, optimal_labels)
    assert metrics["gate_accuracy"] == 100.0
    assert metrics["gate_f1"] == 100.0
    assert metrics["tp"] == 1
    assert metrics["tn"] == 1


def test_run_experiments_end_to_end(tmp_path: Path) -> None:
    """Run an end-to-end slice of run_experiments with small n and export files."""
    json_path = tmp_path / "test_results.json"
    md_path = tmp_path / "test_summary.md"
    plot_path = tmp_path / "test_plot.png"

    summary = run_experiments(
        split="test",
        seed=42,
        k_values=[2, 3],
        mode="mock",
        n=4,
        output_json=json_path,
        output_md=md_path,
        output_plot=plot_path,
    )

    assert json_path.exists()
    assert md_path.exists()
    assert plot_path.exists()

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert "metadata" in data
    assert "baseline_comparison" in data
    assert "pareto_sweep" in data
    assert "feature_ablations" in data
    assert "agent_profiling" in data
    assert "error_diagnostics" in data
    assert len(data["baseline_comparison"]) == 5
    assert len(summary["baseline_comparison"]) == 5

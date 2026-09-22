"""
tests/test_week6_ablations.py
==============================
Unit and integration tests for Week 6 Gate Analysis & Ablation Suite.

Tests:
  - Feature masking and AblatedGateWrapper
  - Gate failure taxonomy quadrant calculations
  - Feature ablation execution and deltas
  - N-sweep sample size scaling
  - Pareto frontier metrics calculation
  - LinUCB bandit strategy routing breakdown
  - Markdown report generation
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from agents.baselines.simulated_probe import SimulatedProbe
from gate.classifier import make_classifier
from gate.rule_based_gate import RuleBasedGate
from scripts.week6_ablations import (
    AblatedGateWrapper,
    FailureTaxonomyResult,
    _make_simulated_mas,
    generate_markdown_report,
    mask_features,
    run_bandit_breakdown,
    run_feature_ablations,
    run_gate_taxonomy,
    run_n_sweep,
    run_pareto_analysis,
)
from shared.schemas import GateDecision, GateFeatures, Task


@pytest.fixture
def sample_tasks() -> list[Task]:
    """Create a minimal set of diverse tasks for testing."""
    return [
        Task(
            task_id="task_001",
            question="What is 2 + 2?",
            ground_truth="4",
            source_dataset="template_arithmetic",
            depth_score=1,
            parallel_score=1,
        ),
        Task(
            task_id="task_002",
            question="Who directed the movie Inception and when was he born?",
            ground_truth="Christopher Nolan (1970)",
            source_dataset="hotpotqa_style",
            depth_score=3,
            parallel_score=2,
            context="Christopher Nolan was born in 1970. He directed Inception in 2010.",
        ),
        Task(
            task_id="task_003",
            question="Compare the populations of Paris and London and identify which is larger.",
            ground_truth="London",
            source_dataset="template_comparison",
            depth_score=2,
            parallel_score=2,
        ),
        Task(
            task_id="task_004",
            question="What is the capital of France?",
            ground_truth="Paris",
            source_dataset="synthetic_factoid",
            depth_score=1,
            parallel_score=1,
        ),
    ]


@pytest.fixture
def sample_gate_features() -> GateFeatures:
    return GateFeatures(
        task_id="task_test",
        consistency_score=0.85,
        probe_tokens=140,
        question_word_count=12,
        entity_count=3,
        clause_count=2,
        has_context=True,
        estimated_depth=2,
        estimated_parallel=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feature Masking & Wrapper Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_mask_features_single(sample_gate_features: GateFeatures) -> None:
    masked = mask_features(sample_gate_features, {"consistency_score"})
    assert masked.consistency_score == 0.0
    assert masked.probe_tokens == 140
    assert masked.question_word_count == 12
    assert masked.has_context is True


def test_mask_features_multiple(sample_gate_features: GateFeatures) -> None:
    masked = mask_features(
        sample_gate_features,
        {"probe_tokens", "has_context", "estimated_depth"},
    )
    assert masked.consistency_score == 0.85
    assert masked.probe_tokens == 0
    assert masked.has_context is False
    assert masked.estimated_depth == 0
    assert masked.estimated_parallel == 1


def test_ablated_gate_wrapper(sample_gate_features: GateFeatures) -> None:
    base_gate = make_classifier("logreg")
    # Fit dummy gate
    base_gate.train(
        [sample_gate_features, sample_gate_features],
        ["STOP", "ESCALATE"],
    )

    wrapper = AblatedGateWrapper(base_gate, {"consistency_score"})
    decision = wrapper.predict(sample_gate_features, k=3, probe_tokens=100)
    assert isinstance(decision, GateDecision)
    assert decision.decision in ("STOP", "ESCALATE")
    assert "ablated:consistency_score" in wrapper.name


# ─────────────────────────────────────────────────────────────────────────────
# Taxonomy Analysis Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_run_gate_taxonomy(sample_tasks: list[Task]) -> None:
    probe = SimulatedProbe(seed=42)
    mas_fn = _make_simulated_mas(seed=42)
    gate = RuleBasedGate()

    taxonomy = run_gate_taxonomy(
        tasks=sample_tasks,
        gate=gate,
        probe=probe,
        mas_fn=mas_fn,
        k=3,
    )

    assert isinstance(taxonomy, FailureTaxonomyResult)
    assert taxonomy.total_tasks == len(sample_tasks)
    total_quadrants = (
        taxonomy.true_stops
        + taxonomy.false_stops
        + taxonomy.true_escalates
        + taxonomy.false_escalates
    )
    assert total_quadrants == len(sample_tasks)
    assert 0.0 <= taxonomy.precision <= 1.0
    assert 0.0 <= taxonomy.recall <= 1.0
    assert 0.0 <= taxonomy.f1 <= 1.0
    assert len(taxonomy.detailed_cases) == len(sample_tasks)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Ablation Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_run_feature_ablations(sample_tasks: list[Task]) -> None:
    metrics = run_feature_ablations(
        train_tasks=sample_tasks,
        test_tasks=sample_tasks,
        seeds=[42],
        k=3,
    )

    assert len(metrics) > 0
    baseline = metrics[0]
    assert "Baseline" in baseline.name
    assert baseline.delta_accuracy == 0.0
    assert baseline.delta_savings == 0.0

    for m in metrics:
        assert 0.0 <= m.accuracy <= 100.0
        assert m.avg_tokens >= 0.0
        assert 0.0 <= m.stop_rate_pct <= 100.0


# ─────────────────────────────────────────────────────────────────────────────
# N-Sweep Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_run_n_sweep(sample_tasks: list[Task]) -> None:
    n_results = run_n_sweep(
        train_tasks=sample_tasks,
        test_tasks=sample_tasks,
        seeds=[42],
        n_values=[3, 5],
        k=3,
    )

    assert len(n_results) == 2
    for res in n_results:
        assert res.n_samples in (3, 5)
        assert 0.0 <= res.probe_accuracy <= 100.0
        assert 0.0 <= res.pipeline_accuracy <= 100.0
        assert res.probe_avg_tokens > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Pareto Analysis Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_run_pareto_analysis(sample_tasks: list[Task]) -> None:
    points = run_pareto_analysis(
        train_tasks=sample_tasks,
        test_tasks=sample_tasks,
        seeds=[42],
        k_values=[2, 3],
    )

    assert len(points) == 10  # 5 methods x 2 k-values
    methods = {p.method for p in points}
    assert "GateOrchestra" in methods
    assert "Always-MAS" in methods
    assert "CoT-SC-only" in methods

    for p in points:
        assert p.k in (2, 3)
        assert 0.0 <= p.accuracy <= 100.0
        assert p.avg_tokens >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Bandit Strategy Breakdown Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_run_bandit_breakdown(sample_tasks: list[Task]) -> None:
    bandit = run_bandit_breakdown(sample_tasks, seed=42)

    assert bandit.total_tasks == len(sample_tasks)
    arm_sum = sum(bandit.arm_counts.values())
    assert arm_sum == len(sample_tasks)

    for arm in ("react", "debate", "reflexion"):
        assert arm in bandit.arm_counts
        assert 0.0 <= bandit.arm_proportions[arm] <= 100.0
        assert 0.0 <= bandit.mean_rewards[arm] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Markdown Report Generator Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_generate_markdown_report(sample_tasks: list[Task]) -> None:
    probe = SimulatedProbe(seed=42)
    mas_fn = _make_simulated_mas(seed=42)
    gate = RuleBasedGate()

    taxonomy = run_gate_taxonomy(sample_tasks, gate, probe, mas_fn, k=3)
    bandit = run_bandit_breakdown(sample_tasks, seed=42)

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = Path(tmpdir) / "test_report.md"
        generate_markdown_report(
            taxonomy=taxonomy,
            ablations=None,
            n_sweep=None,
            pareto=None,
            bandit=bandit,
            output_path=report_path,
        )

        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "Gate Failure Taxonomy" in content
        assert "LinUCB Bandit Strategy Routing Breakdown" in content

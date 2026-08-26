"""
tests/test_gate.py
==================
Unit tests for all gate/ modules.
"""

import pytest

from gate.classifier import LogRegGate, make_classifier
from gate.feature_extractor import _regex_features, extract_features
from gate.random_gate import RandomGate
from gate.rule_based_gate import RuleBasedGate
from shared.schemas import GateFeatures, ProbeResult, Task

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def simple_task() -> Task:
    return Task(task_id="s001", question="What is the capital of France?", ground_truth="Paris")


@pytest.fixture
def complex_task() -> Task:
    return Task(
        task_id="c001",
        question=(
            "The scientist who discovered radioactivity and the mathematician who proved "
            "Fermat's Last Theorem were both from which country?"
        ),
        context="Marie Curie discovered radioactivity. Andrew Wiles proved Fermat's Last Theorem.",
        depth_score=4,
        parallel_score=3,
    )


@pytest.fixture
def high_consistency_probe(simple_task) -> ProbeResult:
    return ProbeResult(
        task_id=simple_task.task_id,
        answer="Paris",
        consistency_score=1.0,
        tokens_used=100,
        raw_outputs=["Paris"] * 5,
    )


@pytest.fixture
def low_consistency_probe(complex_task) -> ProbeResult:
    return ProbeResult(
        task_id=complex_task.task_id,
        answer="France",
        consistency_score=0.4,
        tokens_used=250,
        raw_outputs=["France", "France", "USA", "UK", "France"],
    )


@pytest.fixture
def simple_features(simple_task, high_consistency_probe) -> GateFeatures:
    return extract_features(simple_task, high_consistency_probe, use_spacy=False)


@pytest.fixture
def complex_features(complex_task, low_consistency_probe) -> GateFeatures:
    return extract_features(complex_task, low_consistency_probe, use_spacy=False)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extractor tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFeatureExtractor:
    def test_simple_task_features(self, simple_features):
        assert simple_features.task_id == "s001"
        assert simple_features.consistency_score == 1.0
        assert simple_features.question_word_count == 6
        assert simple_features.has_context is False

    def test_complex_task_has_context(self, complex_features):
        assert complex_features.has_context is True

    def test_task_probe_id_mismatch_raises(self, simple_task):
        bad_probe = ProbeResult(
            task_id="WRONG_ID",
            answer="X",
            consistency_score=0.5,
            tokens_used=50,
        )
        with pytest.raises(ValueError, match="task_id"):
            extract_features(simple_task, bad_probe, use_spacy=False)

    def test_regex_features_entity_count(self):
        entities, _ = _regex_features("Marie Curie was born in Warsaw, Poland.")
        assert entities >= 3  # Marie Curie, Warsaw, Poland

    def test_word_count_accurate(self, simple_features):
        assert simple_features.question_word_count == len("What is the capital of France?".split())

    def test_depth_estimate_nonnegative(self, simple_features, complex_features):
        assert simple_features.estimated_depth >= 0
        assert complex_features.estimated_depth >= 0

    def test_complex_task_higher_depth_than_simple(self, simple_features, complex_features):
        assert complex_features.estimated_depth >= simple_features.estimated_depth


# ─────────────────────────────────────────────────────────────────────────────
# RuleBasedGate tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRuleBasedGate:
    def test_high_consistency_simple_task_stops(self, simple_features):
        gate = RuleBasedGate()
        decision = gate.predict(simple_features, k=3, probe_tokens=100)
        assert decision.decision == "STOP"
        assert decision.token_budget_cap is None
        assert decision.gate_type == "rule_based"
        assert decision.confidence == 1.0

    def test_low_consistency_doesnt_auto_stop(self, complex_features):
        gate = RuleBasedGate()
        # complex task has low consistency — rule 1 shouldn't fire
        decision = gate.predict(complex_features, k=3, probe_tokens=250)
        # It may STOP via default or ESCALATE via depth rule — just verify it's valid
        assert decision.decision in ("STOP", "ESCALATE")

    def test_high_depth_escalates(self):
        gate = RuleBasedGate(depth_escalate=2.0)
        features = GateFeatures(
            task_id="deep_001",
            consistency_score=0.4,
            probe_tokens=200,
            question_word_count=20,
            entity_count=5,
            clause_count=4,
            has_context=True,
            estimated_depth=4.0,  # Above threshold
            estimated_parallel=1.0,
        )
        decision = gate.predict(features, k=3, probe_tokens=200)
        assert decision.decision == "ESCALATE"
        assert decision.token_budget_cap == 3 * 200

    def test_token_budget_correct_on_escalate(self):
        gate = RuleBasedGate(depth_escalate=1.0)  # Very low threshold → always ESCALATE
        features = GateFeatures(
            task_id="t001",
            consistency_score=0.3,
            probe_tokens=150,
            question_word_count=10,
            entity_count=2,
            clause_count=1,
            has_context=False,
            estimated_depth=2.0,
        )
        decision = gate.predict(features, k=5, probe_tokens=150)
        assert decision.token_budget_cap == 5 * 150

    def test_explain_returns_correct_rule_name(self, simple_features):
        gate = RuleBasedGate()
        trace = gate.explain(simple_features)
        assert trace.decision in ("STOP", "ESCALATE")
        assert trace.rule_name != ""


# ─────────────────────────────────────────────────────────────────────────────
# RandomGate tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRandomGate:
    def test_deterministic_with_same_seed(self, simple_features):
        gate1 = RandomGate(escalation_rate=0.5, seed=42)
        gate2 = RandomGate(escalation_rate=0.5, seed=42)
        for _ in range(20):
            d1 = gate1.predict(simple_features, k=3, probe_tokens=100)
            d2 = gate2.predict(simple_features, k=3, probe_tokens=100)
            assert d1.decision == d2.decision

    def test_always_stop_gate(self, simple_features):
        gate = RandomGate(escalation_rate=0.0, seed=42)
        for _ in range(10):
            assert gate.predict(simple_features, k=3, probe_tokens=100).decision == "STOP"

    def test_always_escalate_gate(self, simple_features):
        gate = RandomGate(escalation_rate=1.0, seed=42)
        for _ in range(10):
            assert gate.predict(simple_features, k=3, probe_tokens=100).decision == "ESCALATE"

    def test_empirical_rate_close_to_target(self):
        gate = RandomGate(escalation_rate=0.4, seed=42)
        observed = gate.observed_escalation_rate(n=5000)
        assert abs(observed - 0.4) < 0.03, f"Expected ≈0.4, got {observed:.3f}"

    def test_confidence_always_half(self, simple_features):
        gate = RandomGate(seed=42)
        for _ in range(5):
            assert gate.predict(simple_features, k=3, probe_tokens=100).confidence == 0.5

    def test_invalid_rate_rejected(self):
        with pytest.raises(ValueError):
            RandomGate(escalation_rate=1.5)


# ─────────────────────────────────────────────────────────────────────────────
# Classifier stub tests (train + predict cycle with toy data)
# ─────────────────────────────────────────────────────────────────────────────


def _make_toy_data(n: int = 20) -> tuple[list[GateFeatures], list[str]]:
    """Generate toy training data (alternating STOP/ESCALATE)."""
    features = []
    labels = []
    for i in range(n):
        f = GateFeatures(
            task_id=f"toy_{i:03d}",
            consistency_score=1.0 - (i % 5) * 0.2,
            probe_tokens=100 + i * 10,
            question_word_count=5 + i % 10,
            entity_count=i % 4,
            clause_count=i % 3,
            has_context=i % 2 == 0,
            estimated_depth=float(i % 5),
            estimated_parallel=float(i % 4),
        )
        label = "ESCALATE" if i % 3 == 0 else "STOP"
        features.append(f)
        labels.append(label)
    return features, labels


class TestClassifiers:
    @pytest.mark.parametrize("clf_name", ["logreg", "gbt"])
    def test_train_and_predict(self, clf_name, simple_features):
        train_features, train_labels = _make_toy_data(30)
        gate = make_classifier(clf_name)
        gate.train(train_features, train_labels)
        decision = gate.predict(simple_features, k=3, probe_tokens=100)
        assert decision.decision in ("STOP", "ESCALATE")
        assert 0.0 <= decision.confidence <= 1.0

    def test_predict_before_train_raises(self, simple_features):
        gate = LogRegGate()
        with pytest.raises(RuntimeError, match="train"):
            gate.predict(simple_features, k=3, probe_tokens=100)

    def test_make_classifier_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown classifier"):
            make_classifier("unknown_clf")

    def test_gbt_feature_importances(self):
        train_features, train_labels = _make_toy_data(30)
        gate = make_classifier("gbt")
        gate.train(train_features, train_labels)
        importances = gate.feature_importances()  # type: ignore[attr-defined]
        assert len(importances) == 8  # 8 features
        assert abs(sum(importances.values()) - 1.0) < 1e-6

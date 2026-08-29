"""
tests/test_coverage_boost.py
=============================
Additional tests to cover previously untested code paths in:
  - shared/token_logger.py  (get_total_by_method, get_escalation_rate, save/load JSON, reset, globals)
  - gate/train_gate.py      (apply_label_rule, evaluate_classifier, train_gate, load_eval_results_from_jsonl)
  - gate/classifier.py      (save/load, GBTGate paths)

These tests do NOT change any application behaviour; they only exercise
existing logic that was previously unreachable from the test suite.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from gate.classifier import GateClassifier, GBTGate, make_classifier
from gate.train_gate import (
    apply_label_rule,
    evaluate_classifier,
    load_eval_results_from_jsonl,
    train_gate,
)
from shared.schemas import EvalResult, GateFeatures
from shared.token_logger import (
    TokenAccountant,
    get_global_accountant,
    reset_global_accountant,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_features() -> GateFeatures:
    return GateFeatures(
        task_id="t001",
        consistency_score=0.8,
        probe_tokens=120,
        question_word_count=6,
        entity_count=1,
        clause_count=0,
        has_context=False,
        estimated_depth=1.0,
        estimated_parallel=0.5,
    )


@pytest.fixture
def features_list() -> list[GateFeatures]:
    """A small list of GateFeatures for training."""
    items = []
    for i in range(20):
        items.append(
            GateFeatures(
                task_id=f"t{i:03d}",
                consistency_score=0.9 if i % 2 == 0 else 0.4,
                probe_tokens=100 + i * 10,
                question_word_count=5 + i,
                entity_count=i % 3,
                clause_count=i % 2,
                has_context=bool(i % 3),
                estimated_depth=float(1 + i % 5),
                estimated_parallel=float(i % 4),
            )
        )
    return items


@pytest.fixture
def train_labels() -> list[str]:
    return ["STOP" if i % 2 == 0 else "ESCALATE" for i in range(20)]


# ─────────────────────────────────────────────────────────────────────────────
# TokenAccountant — extended coverage
# ─────────────────────────────────────────────────────────────────────────────


class TestTokenAccountantExtended:
    def _make_accountant(self) -> TokenAccountant:
        acc = TokenAccountant()
        acc.log("t001", method="GateOrchestra", stage="probe", tokens=100, path="STOP")
        acc.log("t001", method="GateOrchestra", stage="mas", tokens=400, path="ESCALATE")
        acc.log("t002", method="CoT-SC", stage="probe", tokens=150, path="STOP")
        acc.log("t003", method="GateOrchestra", stage="probe", tokens=80, path="ESCALATE")
        return acc

    def test_get_total_by_method(self) -> None:
        acc = self._make_accountant()
        totals = acc.get_total_by_method()
        assert totals["GateOrchestra"] == 580
        assert totals["CoT-SC"] == 150

    def test_get_escalation_rate_no_records(self) -> None:
        acc = TokenAccountant()
        assert acc.get_escalation_rate() == 0.0

    def test_get_escalation_rate_no_stop_or_escalate_paths(self) -> None:
        acc = TokenAccountant()
        acc.log("t001", method="GateOrchestra", stage="probe", tokens=100, path="N/A")
        assert acc.get_escalation_rate() == 0.0

    def test_get_escalation_rate_mixed(self) -> None:
        acc = TokenAccountant()
        # t001 → ESCALATE, t002 → STOP  => rate = 0.5
        acc.log("t001", method="GateOrchestra", stage="probe", tokens=100, path="ESCALATE")
        acc.log("t002", method="GateOrchestra", stage="probe", tokens=100, path="STOP")
        rate = acc.get_escalation_rate("GateOrchestra")
        assert rate == pytest.approx(0.5)

    def test_get_escalation_rate_all_escalate(self) -> None:
        acc = TokenAccountant()
        acc.log("t001", method="GateOrchestra", stage="probe", tokens=100, path="ESCALATE")
        acc.log("t002", method="GateOrchestra", stage="probe", tokens=100, path="ESCALATE")
        assert acc.get_escalation_rate("GateOrchestra") == pytest.approx(1.0)

    def test_reset_clears_records(self) -> None:
        acc = self._make_accountant()
        assert len(acc) > 0
        acc.reset()
        assert len(acc) == 0

    def test_repr_contains_info(self) -> None:
        acc = self._make_accountant()
        r = repr(acc)
        assert "TokenAccountant" in r
        assert "records=" in r

    def test_save_and_load_json_roundtrip(self) -> None:
        acc = self._make_accountant()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = Path(f.name)
        try:
            acc.save_to_json(tmp_path)
            data = json.loads(tmp_path.read_text(encoding="utf-8"))
            assert "records" in data
            assert "summary" in data
            assert len(data["records"]) == 4
            # Reload
            acc2 = TokenAccountant.load_from_json(tmp_path)
            assert len(acc2) == 4
            assert acc2.get_total_by_method()["GateOrchestra"] == 580
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_save_to_json_summary_structure(self) -> None:
        acc = TokenAccountant()
        acc.log("t001", method="GateOrchestra", stage="probe", tokens=200, path="ESCALATE")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = Path(f.name)
        try:
            acc.save_to_json(tmp_path)
            data = json.loads(tmp_path.read_text(encoding="utf-8"))
            summary = data["summary"]
            assert "GateOrchestra" in summary
            assert "total_tokens" in summary["GateOrchestra"]
            assert "escalation_rate" in summary["GateOrchestra"]
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_global_accountant_singleton(self) -> None:
        reset_global_accountant()
        a1 = get_global_accountant()
        a2 = get_global_accountant()
        assert a1 is a2

    def test_reset_global_accountant(self) -> None:
        a1 = get_global_accountant()
        reset_global_accountant()
        a2 = get_global_accountant()
        assert a1 is not a2


# ─────────────────────────────────────────────────────────────────────────────
# gate/train_gate.py
# ─────────────────────────────────────────────────────────────────────────────


def _make_eval_result(task_id: str, method: str, is_correct: bool | None) -> EvalResult:
    # Use valid Literal values accepted by EvalResult.method
    valid_methods = ["CoT-SC-only", "Always-MAS", "GateOrchestra", "RuleBasedGate", "RandomGate"]
    safe_method = method if method in valid_methods else "CoT-SC-only"
    return EvalResult(
        task_id=task_id,
        method=safe_method,  # type: ignore[arg-type]
        predicted_answer="A",
        is_correct=is_correct,
        tokens_spent=100,
        probe_tokens=50,
    )


class TestApplyLabelRule:
    def test_basic_labeling(self) -> None:
        cot = {
            "t001": _make_eval_result("t001", "CoT-SC", is_correct=False),
            "t002": _make_eval_result("t002", "CoT-SC", is_correct=True),
        }
        mas = {
            "t001": _make_eval_result("t001", "AlwaysMAS", is_correct=True),
            "t002": _make_eval_result("t002", "AlwaysMAS", is_correct=True),
        }
        labels = apply_label_rule(cot, mas)
        assert labels["t001"] == "ESCALATE"
        assert labels["t002"] == "STOP"

    def test_both_wrong_is_stop(self) -> None:
        cot = {"t001": _make_eval_result("t001", "CoT-SC", is_correct=False)}
        mas = {"t001": _make_eval_result("t001", "AlwaysMAS", is_correct=False)}
        labels = apply_label_rule(cot, mas)
        assert labels["t001"] == "STOP"

    def test_no_common_tasks_raises(self) -> None:
        cot = {"t001": _make_eval_result("t001", "CoT-SC", is_correct=True)}
        mas = {"t999": _make_eval_result("t999", "AlwaysMAS", is_correct=True)}
        with pytest.raises(ValueError, match="No common task_ids"):
            apply_label_rule(cot, mas)

    def test_none_is_correct_skipped(self) -> None:
        cot = {
            "t001": _make_eval_result("t001", "CoT-SC", is_correct=None),
            "t002": _make_eval_result("t002", "CoT-SC", is_correct=True),
        }
        mas = {
            "t001": _make_eval_result("t001", "AlwaysMAS", is_correct=None),
            "t002": _make_eval_result("t002", "AlwaysMAS", is_correct=False),
        }
        labels = apply_label_rule(cot, mas)
        assert "t001" not in labels
        assert labels["t002"] == "STOP"


class TestEvaluateClassifier:
    def test_perfect_predictions(self, features_list: list[GateFeatures]) -> None:
        labels = ["STOP" if i % 2 == 0 else "ESCALATE" for i in range(20)]
        gate = make_classifier("logreg")
        gate.train(features_list, labels)
        metrics = evaluate_classifier(gate, features_list, labels, k=3)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "escalation_rate" in metrics
        assert metrics["n"] == 20

    def test_all_stop_labels(self, features_list: list[GateFeatures]) -> None:
        # Logistic regression requires >=2 classes; we test a gate pre-trained
        # on mixed labels, then measure against all-STOP val set
        mixed_labels = ["STOP" if i % 2 == 0 else "ESCALATE" for i in range(20)]
        gate = make_classifier("logreg")
        gate.train(features_list, mixed_labels)
        all_stop_labels = ["STOP"] * 20
        metrics = evaluate_classifier(gate, features_list, all_stop_labels, k=3)
        # With no ESCALATE ground-truth, recall for ESCALATE == 0
        assert metrics["recall"] == pytest.approx(0.0)

    def test_empty_labels(self, features_list: list[GateFeatures]) -> None:
        labels = ["STOP" if i % 2 == 0 else "ESCALATE" for i in range(20)]
        gate = make_classifier("logreg")
        gate.train(features_list, labels)
        metrics = evaluate_classifier(gate, [], [], k=3)
        assert metrics["accuracy"] == 0.0
        assert metrics["escalation_rate"] == 0.0


class TestTrainGate:
    def test_train_gate_returns_best_gate_and_metrics(
        self, features_list: list[GateFeatures], train_labels: list[str], tmp_path: Path
    ) -> None:
        val_features = features_list[:5]
        val_labels = train_labels[:5]
        save_path = tmp_path / "best_gate.pkl"
        best_gate, best_metrics = train_gate(
            features_list,
            train_labels,
            val_features,
            val_labels,
            classifier_names=["logreg"],
            k_values=[3],
            save_path=save_path,
        )
        assert isinstance(best_gate, GateClassifier)
        assert "classifier" in best_metrics
        assert save_path.exists()

    def test_train_gate_gbt_only(
        self, features_list: list[GateFeatures], train_labels: list[str], tmp_path: Path
    ) -> None:
        """train_gate with GBT-only classifier selects the best config."""
        val_features = features_list[:5]
        val_labels = train_labels[:5]
        save_path = tmp_path / "best_gate_gbt.pkl"
        best_gate, best_metrics = train_gate(
            features_list,
            train_labels,
            val_features,
            val_labels,
            classifier_names=["gbt"],
            k_values=[2, 5],
            save_path=save_path,
        )
        assert isinstance(best_gate, GateClassifier)
        assert best_metrics["classifier"] == "gbt"
        assert save_path.exists()


class TestLoadEvalResultsFromJsonl:
    def test_roundtrip(self, tmp_path: Path) -> None:
        result = _make_eval_result("t001", "CoT-SC", is_correct=True)
        jsonl_path = tmp_path / "results.jsonl"
        jsonl_path.write_text(result.model_dump_json() + "\n", encoding="utf-8")
        loaded = load_eval_results_from_jsonl(jsonl_path)
        assert "t001" in loaded
        assert loaded["t001"].is_correct is True

    def test_empty_lines_skipped(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "results.jsonl"
        jsonl_path.write_text("\n\n", encoding="utf-8")
        loaded = load_eval_results_from_jsonl(jsonl_path)
        assert loaded == {}


# ─────────────────────────────────────────────────────────────────────────────
# gate/classifier.py — save / load
# ─────────────────────────────────────────────────────────────────────────────


class TestClassifierSaveLoad:
    def test_logreg_save_load_roundtrip(
        self, features_list: list[GateFeatures], train_labels: list[str], tmp_path: Path
    ) -> None:
        gate = make_classifier("logreg")
        gate.train(features_list, train_labels)
        save_path = tmp_path / "logreg.pkl"
        gate.save(save_path)
        assert save_path.exists()

        loaded = GateClassifier.load(save_path)
        decision = loaded.predict(features_list[0], k=3, probe_tokens=100)
        assert decision.decision in ("STOP", "ESCALATE")

    def test_gbt_save_load_roundtrip(
        self, features_list: list[GateFeatures], train_labels: list[str], tmp_path: Path
    ) -> None:
        gate = make_classifier("gbt")
        gate.train(features_list, train_labels)
        save_path = tmp_path / "gbt.pkl"
        gate.save(save_path)
        loaded = GateClassifier.load(save_path)
        decision = loaded.predict(features_list[0], k=3, probe_tokens=100)
        assert decision.decision in ("STOP", "ESCALATE")

    def test_gbt_feature_importances_after_train(
        self, features_list: list[GateFeatures], train_labels: list[str]
    ) -> None:
        gate = GBTGate()
        gate.train(features_list, train_labels)
        importances = gate.feature_importances()
        assert isinstance(importances, dict)
        assert len(importances) > 0
        assert all(isinstance(v, float) for v in importances.values())

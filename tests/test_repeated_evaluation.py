"""Offline tests for repeated train/validation evaluation support."""

from pathlib import Path

import pytest

from gate.train_gate import (
    apply_repeated_label_rule,
    calibrate_gate,
    evaluate_validation_policy,
    select_calibration_candidate,
)
from scripts.run_final_evaluation import (
    _load_checkpoint,
    _new_checkpoint,
    _result_key,
    _run_repeated_baseline,
    _save_checkpoint,
)
from shared.schemas import EvalResult, GateDecision, GateFeatures, Task
from shared.token_logger import TokenAccountant


def _result(task_id: str, method: str, is_correct: bool) -> EvalResult:
    return EvalResult(
        task_id=task_id,
        method=method,  # type: ignore[arg-type]
        predicted_answer="answer",
        is_correct=is_correct,
        tokens_spent=10,
        probe_tokens=10 if method == "CoT-SC-only" else None,
        mas_tokens=10 if method == "Always-MAS" else None,
    )


def test_repeat_key_includes_repeat_index() -> None:
    assert _result_key("train", "CoT-SC-only", "task-1", 2) == ("train:CoT-SC-only:task-1:2")


def test_repeated_label_rule_uses_accuracy_gap_and_strict_threshold() -> None:
    cot = {"task-1": [_result("task-1", "CoT-SC-only", True)] * 3}
    mas = {
        "task-1": [
            _result("task-1", "Always-MAS", True),
            _result("task-1", "Always-MAS", False),
            _result("task-1", "Always-MAS", False),
        ]
    }

    assert apply_repeated_label_rule(cot, mas, n_repeats=3, tau_acc=-0.7)["task-1"] == "ESCALATE"
    assert apply_repeated_label_rule(cot, mas, n_repeats=3, tau_acc=-2 / 3)["task-1"] == "STOP"


def test_repeated_label_rule_rejects_missing_repeats() -> None:
    cot = {"task-1": [_result("task-1", "CoT-SC-only", False)]}
    mas = {"task-1": [_result("task-1", "Always-MAS", True)] * 2}

    with pytest.raises(ValueError, match="Missing repeats"):
        apply_repeated_label_rule(cot, mas, n_repeats=2)


def test_repeated_baseline_resumes_completed_repeats(tmp_path: Path) -> None:
    task = Task(task_id="task-1", question="Question", ground_truth="answer")
    metadata = {"checkpoint_version": 3, "test": "repeat-resume"}
    state = _new_checkpoint(metadata)
    checkpoint_path = tmp_path / "checkpoint.json"
    calls: list[int] = []

    def runner(current_task: Task, repeat_index: int, _: TokenAccountant) -> EvalResult:
        calls.append(repeat_index)
        return _result(current_task.task_id, "CoT-SC-only", True)

    first = _run_repeated_baseline(
        tasks=[task],
        phase="train",
        method="CoT-SC-only",
        n_repeats=3,
        state=state,
        checkpoint_path=checkpoint_path,
        accountant_factory=TokenAccountant,
        runner=runner,
    )
    assert len(first[task.task_id]) == 3
    assert calls == [0, 1, 2]

    resumed_state = _load_checkpoint(checkpoint_path, metadata)
    calls.clear()
    second = _run_repeated_baseline(
        tasks=[task],
        phase="train",
        method="CoT-SC-only",
        n_repeats=3,
        state=resumed_state,
        checkpoint_path=checkpoint_path,
        accountant_factory=TokenAccountant,
        runner=runner,
    )
    assert len(second[task.task_id]) == 3
    assert calls == []


def test_checkpoint_round_trip_preserves_repeat_records(tmp_path: Path) -> None:
    metadata = {"checkpoint_version": 3, "test": "round-trip"}
    state = _new_checkpoint(metadata)
    state["results"][_result_key("val", "Always-MAS", "task-1", 1)] = {
        "repeat_index": 1,
        "result": _result("task-1", "Always-MAS", True).model_dump(mode="json"),
    }
    checkpoint_path = tmp_path / "checkpoint.json"
    _save_checkpoint(state, checkpoint_path)

    loaded = _load_checkpoint(checkpoint_path, metadata)
    assert _result_key("val", "Always-MAS", "task-1", 1) in loaded["results"]


class _TaskGate:
    def __init__(self, escalate_task_id: str) -> None:
        self.escalate_task_id = escalate_task_id

    def predict(self, features: GateFeatures, k: int, probe_tokens: int) -> GateDecision:
        decision = "ESCALATE" if features.task_id == self.escalate_task_id else "STOP"
        return GateDecision(
            task_id=features.task_id,
            decision=decision,
            confidence=1.0,
            token_budget_cap=k * probe_tokens if decision == "ESCALATE" else None,
        )


def test_validation_policy_calculates_accuracy_and_savings() -> None:
    features = [
        GateFeatures(
            task_id="task-1",
            consistency_score=1.0,
            probe_tokens=10,
            question_word_count=1,
            entity_count=0,
            clause_count=0,
            has_context=False,
        ),
        GateFeatures(
            task_id="task-2",
            consistency_score=1.0,
            probe_tokens=20,
            question_word_count=1,
            entity_count=0,
            clause_count=0,
            has_context=False,
        ),
    ]
    cot = {
        "task-1": _result("task-1", "CoT-SC-only", False),
        "task-2": _result("task-2", "CoT-SC-only", True),
    }
    mas = {
        "task-1": _result("task-1", "Always-MAS", True),
        "task-2": _result("task-2", "Always-MAS", False),
    }
    metrics = evaluate_validation_policy(_TaskGate("task-1"), features, cot, {2: mas}, k=2)
    assert metrics["accuracy"] == 1.0
    assert metrics["avg_tokens"] == 15.0
    assert metrics["escalation_rate"] == 0.5
    assert metrics["token_savings_pct"] == pytest.approx(-50.0)


def test_calibration_candidate_generation_and_selection(tmp_path: Path) -> None:
    features = [
        GateFeatures(
            task_id="task-1",
            consistency_score=1.0,
            probe_tokens=10,
            question_word_count=1,
            entity_count=0,
            clause_count=0,
            has_context=False,
        ),
        GateFeatures(
            task_id="task-2",
            consistency_score=1.0,
            probe_tokens=10,
            question_word_count=1,
            entity_count=0,
            clause_count=0,
            has_context=False,
        ),
    ]
    train_cot = {
        task_id: [_result(task_id, "CoT-SC-only", correct) for _ in range(3)]
        for task_id, correct in {"task-1": False, "task-2": True}.items()
    }
    train_mas = {
        task_id: [_result(task_id, "Always-MAS", not correct) for _ in range(3)]
        for task_id, correct in {"task-1": False, "task-2": True}.items()
    }
    val_cot = {
        task_id: [_result(task_id, "CoT-SC-only", correct) for _ in range(3)]
        for task_id, correct in {"task-1": False, "task-2": True}.items()
    }
    val_mas_repeated = {
        task_id: [_result(task_id, "Always-MAS", not correct) for _ in range(3)]
        for task_id, correct in {"task-1": False, "task-2": True}.items()
    }
    val_cot_one = {task_id: runs[0] for task_id, runs in val_cot.items()}
    mas_by_k = {
        2: {
            task_id: _result(task_id, "Always-MAS", not result.is_correct)
            for task_id, result in val_cot_one.items()
        },
    }
    gate, selected, candidates = calibrate_gate(
        train_features=features,
        train_repeated_cot=train_cot,
        train_repeated_mas=train_mas,
        val_features=features,
        val_repeated_cot=val_cot,
        val_repeated_mas=val_mas_repeated,
        val_cot_results=val_cot_one,
        val_mas_results_by_k=mas_by_k,
        n_repeats=3,
        tau_values=[0.05],
        k_values=[2],
        classifier_names=["logreg"],
        save_path=tmp_path / "gate.pkl",
    )
    assert gate is not None
    assert selected["tau_acc"] == 0.05
    assert selected["k"] == 2
    assert len(candidates) == 1


def test_calibration_selection_uses_epsilon_and_deterministic_ties() -> None:
    candidates = [
        {
            "classifier": "gbt",
            "tau_acc": 0.08,
            "k": 5,
            "accuracy": 1.0,
            "token_savings_pct": 10.0,
            "avg_tokens": 90.0,
        },
        {
            "classifier": "logreg",
            "tau_acc": 0.03,
            "k": 2,
            "accuracy": 0.995,
            "token_savings_pct": 20.0,
            "avg_tokens": 80.0,
        },
    ]
    selected = select_calibration_candidate(candidates, epsilon=0.01)
    assert selected["classifier"] == "logreg"

    tied = [
        {
            "classifier": "gbt",
            "tau_acc": 0.08,
            "k": 5,
            "accuracy": 1.0,
            "token_savings_pct": 20.0,
            "avg_tokens": 80.0,
        },
        {
            "classifier": "logreg",
            "tau_acc": 0.03,
            "k": 2,
            "accuracy": 1.0,
            "token_savings_pct": 20.0,
            "avg_tokens": 80.0,
        },
    ]
    assert select_calibration_candidate(tied, epsilon=0.01)["k"] == 2

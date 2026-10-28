"""
tests/test_final_results.py
===========================
Result aggregation, the integrity verifier, paired statistics, and honest
gate calibration when the training labels have a single class.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.metrics import mcnemar_exact
from gate.classifier import ConstantGate
from gate.train_gate import calibrate_gate
from scripts.final_benchmark import build_master_results
from scripts.verify_final_results import verify_master_results
from shared.schemas import EvalResult, GateFeatures

METHODS = ["CoT-SC-only", "Always-MAS", "RandomGate", "RuleBasedGate", "GateOrchestra"]
DATASET_HASH = "a" * 64


def _seed_run(seed: int, task_ids: list[str], correct: dict[str, list[bool]]) -> dict:
    results = []
    per_task = {}
    for method in METHODS:
        flags = correct[method]
        n_correct = sum(flags)
        results.append(
            {
                "method": method,
                "n": len(flags),
                "n_correct": n_correct,
                "accuracy": n_correct / len(flags),
                "accuracy_ci95": [0.0, 1.0],
                "avg_tokens": 100.0,
                "token_savings_pct": 0.0,
                "escalate_rate": 0.0,
                "avg_latency_ms": 10.0,
            }
        )
        per_task[method] = [
            {"task_id": t, "is_correct": flag, "tokens_spent": 100, "decision": None}
            for t, flag in zip(task_ids, flags, strict=True)
        ]
    return {
        "seed": seed,
        "n": None,
        "dry_run": False,
        "execution_mode": "real",
        "dataset_sha256": DATASET_HASH,
        "test_task_ids": task_ids,
        "gate": {"classifier": "gbt", "degenerate": False, "note": None},
        "results": results,
        "per_task": per_task,
    }


@pytest.fixture
def master_path(tmp_path: Path) -> Path:
    task_ids = ["t1", "t2", "t3", "t4"]
    correct = {method: [True, False, True, False] for method in METHODS}
    correct["Always-MAS"] = [False, False, True, False]
    runs = [_seed_run(seed, task_ids, correct) for seed in (42, 123, 999)]
    path = tmp_path / "master_results.json"
    path.write_text(json.dumps(build_master_results(runs, None)), encoding="utf-8")
    return path


def _verify(path: Path, **kwargs):
    return verify_master_results(
        path, expected_dataset_hash=DATASET_HASH, expected_test_size=4, **kwargs
    )


def test_master_results_have_real_seeds_and_pooled_stats(master_path: Path) -> None:
    data = json.loads(master_path.read_text(encoding="utf-8"))
    assert data["seeds"] == [42, 123, 999]
    by_method = {row["method"]: row for row in data["aggregated"]}
    assert by_method["CoT-SC-only"]["seeds"] == [42, 123, 999]
    assert by_method["CoT-SC-only"]["accuracy_mean"] == 50.0
    test = next(
        t
        for t in data["pooled_paired_tests"]
        if t["method"] == "CoT-SC-only" and t["reference"] == "Always-MAS"
    )
    assert (test["n_pairs"], test["method_only_correct"], test["reference_only_correct"]) == (
        12,
        3,
        0,
    )


def test_verifier_passes_a_complete_consistent_run(master_path: Path) -> None:
    errors, partial = _verify(master_path)
    assert errors == []
    assert partial == []


def test_verifier_rejects_dataset_hash_mismatch(master_path: Path) -> None:
    errors, _ = verify_master_results(
        master_path, expected_dataset_hash="b" * 64, expected_test_size=4
    )
    assert any("dataset hash" in e for e in errors)


def test_verifier_rejects_tampered_aggregate(master_path: Path) -> None:
    data = json.loads(master_path.read_text(encoding="utf-8"))
    data["aggregated"][0]["accuracy_mean"] = 93.8
    master_path.write_text(json.dumps(data), encoding="utf-8")
    errors, _ = _verify(master_path)
    assert any("does not match per-seed" in e for e in errors)


def test_verifier_rejects_mock_runs(master_path: Path) -> None:
    data = json.loads(master_path.read_text(encoding="utf-8"))
    data["per_seed"][0]["gate"]["classifier"] = "gbt_pretrained_on_simulated_data"
    data["per_seed"][1]["execution_mode"] = "mock"
    master_path.write_text(json.dumps(data), encoding="utf-8")
    errors, _ = _verify(master_path)
    assert any("simulated/mock" in e for e in errors)
    assert any("not a real run" in e for e in errors)


def test_verifier_flags_smoke_runs_as_partial(master_path: Path) -> None:
    data = json.loads(master_path.read_text(encoding="utf-8"))
    data["seeds"] = [42]
    data["per_seed"] = data["per_seed"][:1]
    for row in data["aggregated"]:
        row["seeds"] = [42]
    master_path.write_text(json.dumps(data), encoding="utf-8")
    errors, partial = _verify(master_path)
    assert any("Missing required seeds" in e for e in errors)
    errors, partial = _verify(master_path, allow_partial=True)
    assert errors == []
    assert partial


def test_mcnemar_exact() -> None:
    assert mcnemar_exact([True, True], [True, True]) == {"a_only": 0, "b_only": 0, "p_value": 1.0}
    outcome = mcnemar_exact([True] * 6 + [False] * 4, [False] * 6 + [False] * 4)
    assert outcome["a_only"] == 6
    assert outcome["p_value"] == pytest.approx(2 / 64)


def _features(task_id: str) -> GateFeatures:
    return GateFeatures(
        task_id=task_id,
        consistency_score=1.0,
        probe_tokens=100,
        question_word_count=10,
        entity_count=1,
        clause_count=0,
        has_context=False,
    )


def _result(task_id: str, method: str, correct: bool, tokens: int) -> EvalResult:
    return EvalResult(
        task_id=task_id,
        method=method,
        predicted_answer="x",
        is_correct=correct,
        tokens_spent=tokens,
    )


def test_single_class_labels_produce_an_honest_constant_gate(tmp_path: Path) -> None:
    ids = ["a", "b", "c"]
    # CoT-SC always right, so no task is ever labelled ESCALATE.
    cot = {t: [_result(t, "CoT-SC-only", True, 100)] for t in ids}
    mas = {t: [_result(t, "Always-MAS", False, 300)] for t in ids}
    gate, selected, candidates = calibrate_gate(
        train_features=[_features(t) for t in ids],
        train_repeated_cot=cot,
        train_repeated_mas=mas,
        val_features=[_features(t) for t in ids],
        val_repeated_cot=cot,
        val_repeated_mas=mas,
        val_cot_results={t: runs[0] for t, runs in cot.items()},
        val_mas_results_by_k={3: {t: runs[0] for t, runs in mas.items()}},
        n_repeats=1,
        tau_values=[0.05],
        k_values=[3],
        save_path=tmp_path / "gate.pkl",
    )
    assert isinstance(gate, ConstantGate)
    assert selected["degenerate"] is True
    assert selected["escalation_rate"] == 0.0
    # Metrics are measured on validation data, not invented.
    assert selected["accuracy"] == 1.0
    assert selected["token_savings_pct"] == pytest.approx(100 * (1 - 100 / 300))
    assert all(candidate["classifier"] == "constant" for candidate in candidates)

"""
gate/train_gate.py
==================
Gate classifier training pipeline.

Steps:
  1. Load (GateFeatures, EvalResult) pairs from Person 2's baseline logs
  2. Apply the label rule (TAU_ACC) → binary STOP/ESCALATE labels
  3. Train logreg → GBT → (optionally MLP) classifiers
  4. Sweep τ_acc ∈ {0.03, 0.05, 0.08} and k ∈ {2, 3, 5} on val split
  5. Select best config via token-matched accuracy (Person 4's metric)
  6. Save best model to configs/models/best_gate.pkl

Person 3 owns this file.
Depends on: Person 2's baseline run logs (available ~Week 6).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from gate.classifier import GateClassifier, make_classifier
from shared.config import (
    BEST_MODEL_PATH,
    CLASSIFIER_NAMES,
    K_VALUES,
    TAU_ACC,
)
from shared.schemas import EvalResult, GateFeatures

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Label rule
# ─────────────────────────────────────────────────────────────────────────────


def apply_label_rule(
    cot_sc_results: dict[str, EvalResult],
    mas_results: dict[str, EvalResult],
) -> dict[str, str]:
    """Derive per-task STOP/ESCALATE labels from CoT-SC and Always-MAS results.

    Label rule (τ_acc = binary):
      ESCALATE  iff  MAS got it right  AND  CoT-SC got it wrong
      STOP      otherwise (CoT-SC correct, both wrong, or both right)

    Args:
        cot_sc_results: task_id → EvalResult from CoT-SC-only method.
        mas_results:    task_id → EvalResult from Always-MAS method.

    Returns:
        task_id → "STOP" | "ESCALATE"
    """
    labels: dict[str, str] = {}

    all_task_ids = set(cot_sc_results) & set(mas_results)
    if not all_task_ids:
        raise ValueError("No common task_ids found between CoT-SC and MAS results.")

    for task_id in all_task_ids:
        cot = cot_sc_results[task_id]
        mas = mas_results[task_id]

        if cot.is_correct is None or mas.is_correct is None:
            logger.warning(f"Skipping {task_id}: is_correct is None (no ground truth?)")
            continue

        # Core label rule: only escalate when MAS wins and CoT-SC loses
        if mas.is_correct and not cot.is_correct:
            labels[task_id] = "ESCALATE"
        else:
            labels[task_id] = "STOP"

    n_escalate = sum(1 for v in labels.values() if v == "ESCALATE")
    logger.info(
        f"Label rule applied: {len(labels)} tasks → "
        f"{n_escalate} ESCALATE ({100*n_escalate/len(labels):.1f}%), "
        f"{len(labels)-n_escalate} STOP"
    )
    return labels


def apply_repeated_label_rule(
    cot_sc_results: dict[str, list[EvalResult]],
    mas_results: dict[str, list[EvalResult]],
    n_repeats: int,
    tau_acc: float = TAU_ACC,
) -> dict[str, str]:
    """Label tasks from repeated per-task accuracy gaps."""
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be at least 1, got {n_repeats}")
    task_ids = set(cot_sc_results) & set(mas_results)
    if not task_ids:
        raise ValueError("No common task_ids found between repeated CoT-SC and MAS results.")
    if set(cot_sc_results) != set(mas_results):
        raise ValueError("Repeated CoT-SC and MAS results must contain the same task_ids.")

    labels: dict[str, str] = {}
    for task_id in sorted(task_ids):
        cot_runs = cot_sc_results[task_id]
        mas_runs = mas_results[task_id]
        if len(cot_runs) != n_repeats or len(mas_runs) != n_repeats:
            raise ValueError(
                f"Missing repeats for task_id={task_id!r}: "
                f"expected {n_repeats}, got CoT-SC={len(cot_runs)}, MAS={len(mas_runs)}"
            )
        if any(result.is_correct is None for result in (*cot_runs, *mas_runs)):
            raise ValueError(f"Missing correctness value for task_id={task_id!r}")

        cot_accuracy = sum(result.is_correct is True for result in cot_runs) / n_repeats
        mas_accuracy = sum(result.is_correct is True for result in mas_runs) / n_repeats
        labels[task_id] = "ESCALATE" if mas_accuracy - cot_accuracy > tau_acc else "STOP"

    n_escalate = sum(label == "ESCALATE" for label in labels.values())
    logger.info(
        f"Repeated label rule applied: {len(labels)} tasks -> "
        f"{n_escalate} ESCALATE ({100*n_escalate/len(labels):.1f}%), "
        f"{len(labels)-n_escalate} STOP"
    )
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper (used during sweep)
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_classifier(
    gate: GateClassifier,
    features: list[GateFeatures],
    labels: list[str],
    k: int,
) -> dict[str, float]:
    """Compute gate performance metrics on a labeled set.

    Returns:
        Dict with accuracy, precision, recall, f1, escalation_rate.
    """
    preds = [gate.predict(f, k=k, probe_tokens=f.probe_tokens).decision for f in features]

    tp = sum(
        1 for p, label in zip(preds, labels, strict=True) if p == "ESCALATE" and label == "ESCALATE"
    )
    fp = sum(
        1 for p, label in zip(preds, labels, strict=True) if p == "ESCALATE" and label == "STOP"
    )
    tn = sum(1 for p, label in zip(preds, labels, strict=True) if p == "STOP" and label == "STOP")
    fn = sum(
        1 for p, label in zip(preds, labels, strict=True) if p == "STOP" and label == "ESCALATE"
    )

    accuracy = (tp + tn) / len(labels) if labels else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    escalation_rate = sum(1 for p in preds if p == "ESCALATE") / len(preds) if preds else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "escalation_rate": round(escalation_rate, 4),
        "n": len(labels),
    }


def evaluate_validation_policy(
    gate: GateClassifier,
    features: list[GateFeatures],
    cot_results: dict[str, EvalResult],
    mas_results_by_k: dict[int, dict[str, EvalResult]],
    k: int,
) -> dict[str, float]:
    """Evaluate routed validation answers and token usage for one k."""
    mas_results = mas_results_by_k[k]
    if not features:
        return {
            "accuracy": 0.0,
            "avg_tokens": 0.0,
            "token_savings_pct": 0.0,
            "escalation_rate": 0.0,
            "n": 0,
        }

    baseline_tokens = [result.tokens_spent for result in mas_results.values()]
    baseline_avg_tokens = sum(baseline_tokens) / len(baseline_tokens)
    correct = 0
    total_tokens = 0
    escalated = 0

    for feature in features:
        cot_result = cot_results[feature.task_id]
        decision = gate.predict(feature, k=k, probe_tokens=feature.probe_tokens)
        if decision.decision == "ESCALATE":
            result = mas_results[feature.task_id]
            escalated += 1
            tokens = cot_result.probe_tokens or feature.probe_tokens
            tokens += result.mas_tokens or result.tokens_spent
        else:
            result = cot_result
            tokens = result.tokens_spent
        correct += result.is_correct is True
        total_tokens += tokens

    avg_tokens = total_tokens / len(features)
    savings = 0.0
    if baseline_avg_tokens > 0:
        savings = (1.0 - (avg_tokens / baseline_avg_tokens)) * 100.0
    return {
        "accuracy": correct / len(features),
        "avg_tokens": avg_tokens,
        "token_savings_pct": savings,
        "escalation_rate": escalated / len(features),
        "n": len(features),
    }


def select_calibration_candidate(
    candidates: list[dict[str, Any]], epsilon: float
) -> dict[str, Any]:
    """Select highest-accuracy candidates, then prefer savings deterministically."""
    if not candidates:
        raise ValueError("No calibration candidates were produced.")
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")

    best_accuracy = max(float(candidate["accuracy"]) for candidate in candidates)
    eligible = [
        candidate
        for candidate in candidates
        if float(candidate["accuracy"]) >= best_accuracy - epsilon
    ]
    return min(
        eligible,
        key=lambda candidate: (
            -float(candidate["token_savings_pct"]),
            float(candidate["avg_tokens"]),
            int(candidate["k"]),
            float(candidate["tau_acc"]),
            str(candidate["classifier"]),
        ),
    )


def calibrate_gate(
    *,
    train_features: list[GateFeatures],
    train_repeated_cot: dict[str, list[EvalResult]],
    train_repeated_mas: dict[str, list[EvalResult]],
    val_features: list[GateFeatures],
    val_repeated_cot: dict[str, list[EvalResult]],
    val_repeated_mas: dict[str, list[EvalResult]],
    val_cot_results: dict[str, EvalResult],
    val_mas_results_by_k: dict[int, dict[str, EvalResult]],
    n_repeats: int,
    tau_values: list[float],
    k_values: list[int],
    classifier_names: list[str] | None = None,
    save_path: Path | None = None,
    epsilon: float | None = None,
) -> tuple[GateClassifier, dict[str, Any], list[dict[str, Any]]]:
    """Train and select gates using end-to-end validation accuracy and savings."""
    classifier_names = classifier_names or CLASSIFIER_NAMES
    save_path = save_path or BEST_MODEL_PATH
    epsilon = epsilon if epsilon is not None else max(0.01, 1 / len(val_features))
    candidates: list[dict[str, Any]] = []
    gates: dict[tuple[str, float], GateClassifier] = {}

    for tau_acc in tau_values:
        train_labels = apply_repeated_label_rule(
            train_repeated_cot, train_repeated_mas, n_repeats, tau_acc
        )
        val_labels = apply_repeated_label_rule(
            val_repeated_cot, val_repeated_mas, n_repeats, tau_acc
        )

        train_label_set = set(train_labels.values())
        if train_label_set != {"STOP", "ESCALATE"}:
            print(
                f"[gate-training] skipping tau_acc={tau_acc}: "
                f"training labels are {sorted(train_label_set)}",
                flush=True,
            )
            continue

        for classifier_name in classifier_names:
            gate = make_classifier(classifier_name)
            ordered_train_features = [
                feature for feature in train_features if feature.task_id in train_labels
            ]

            ordered_train_labels = [
                train_labels[feature.task_id] for feature in ordered_train_features
            ]

            gate.train(ordered_train_features, ordered_train_labels)
            gates[(classifier_name, tau_acc)] = gate

            for k in k_values:
                metrics = evaluate_validation_policy(
                    gate,
                    val_features,
                    val_cot_results,
                    val_mas_results_by_k,
                    k,
                )

                candidates.append(
                    {
                        "classifier": classifier_name,
                        "tau_acc": tau_acc,
                        "k": k,
                        **metrics,
                        "train_label_counts": {
                            "STOP": list(train_labels.values()).count("STOP"),
                            "ESCALATE": list(train_labels.values()).count("ESCALATE"),
                        },
                        "val_label_counts": {
                            "STOP": list(val_labels.values()).count("STOP"),
                            "ESCALATE": list(val_labels.values()).count("ESCALATE"),
                        },
                    }
                )

    if not candidates:
        raise ValueError(
            "Insufficient training data for gate calibration: no tau_acc "
            "produced both STOP and ESCALATE training labels."
        )

    selected = select_calibration_candidate(candidates, epsilon)
    selected_gate = gates[(selected["classifier"], selected["tau_acc"])]
    selected = {**selected, "epsilon": epsilon}
    selected_gate.save(save_path)
    return selected_gate, selected, candidates


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────


def train_gate(
    train_features: list[GateFeatures],
    train_labels: list[str],
    val_features: list[GateFeatures],
    val_labels: list[str],
    classifier_names: list[str] | None = None,
    k_values: list[int] | None = None,
    save_path: Path | None = None,
) -> tuple[GateClassifier, dict]:
    """Train gate classifiers and select the best one on the validation set.

    Args:
        train_features:   Training GateFeatures.
        train_labels:     Training labels ("STOP" / "ESCALATE").
        val_features:     Validation GateFeatures.
        val_labels:       Validation labels.
        classifier_names: Which classifiers to try. Default: all 3.
        k_values:         Token multiplier values to sweep. Default: K_VALUES.
        save_path:        Where to save the best model.

    Returns:
        (best_gate, best_metrics) — the best classifier and its val metrics.
    """
    classifier_names = classifier_names or CLASSIFIER_NAMES
    k_values = k_values or K_VALUES
    save_path = save_path or BEST_MODEL_PATH

    best_gate: GateClassifier | None = None
    best_metrics: dict = {}
    best_f1 = -1.0

    results_log: list[dict] = []

    for clf_name in classifier_names:
        logger.info(f"\n{'='*50}\nTraining {clf_name}...")
        gate = make_classifier(clf_name)
        gate.train(train_features, train_labels)

        for k in k_values:
            metrics = evaluate_classifier(gate, val_features, val_labels, k)
            row = {"classifier": clf_name, "k": k, **metrics}
            results_log.append(row)
            logger.info(f"  {clf_name} | k={k} | {metrics}")

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_gate = gate
                best_metrics = row

    if best_gate is None:
        raise RuntimeError("No classifiers trained successfully.")

    logger.info(
        f"\n{'='*50}\nBest: {best_metrics['classifier']} | k={best_metrics['k']} | "
        f"F1={best_metrics['f1']:.4f} | Acc={best_metrics['accuracy']:.4f}"
    )

    best_gate.save(save_path)
    logger.info(f"Best model saved to {save_path}")

    return best_gate, best_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────


def load_eval_results_from_jsonl(path: Path) -> dict[str, EvalResult]:
    """Load EvalResult objects from a JSONL file (Person 2's baseline logs)."""
    results: dict[str, EvalResult] = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                result = EvalResult(**obj)
                results[result.task_id] = result
    return results

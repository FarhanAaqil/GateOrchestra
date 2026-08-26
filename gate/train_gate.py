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

from gate.classifier import GateClassifier, make_classifier
from shared.config import (
    BEST_MODEL_PATH,
    CLASSIFIER_NAMES,
    K_VALUES,
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
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                result = EvalResult(**obj)
                results[result.task_id] = result
    return results

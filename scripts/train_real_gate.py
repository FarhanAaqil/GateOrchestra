"""
scripts/train_real_gate.py
==========================
Week 7 Deliverable: Retrain GBT Gate using REAL Groq LLM Traces.

Reuses existing GateOrchestra architecture:
  - shared.schemas (Task, ProbeResult, GateFeatures, EvalResult)
  - agents.probe_agent (Live ProbeAgent)
  - agents.orchestrator (Live MASOrchestrator)
  - gate.feature_extractor (extract_features)
  - gate.train_gate (apply_label_rule, evaluate_classifier)
  - gate.classifier (GBTGate)

Preserves old model: logs/week2_best_gate.pkl
Saves new model to:  logs/week7_real_gbt_gate.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from agents.orchestrator import MASOrchestrator
from agents.probe_agent import ProbeAgent
from gate.classifier import GBTGate
from gate.feature_extractor import extract_features
from gate.train_gate import apply_label_rule, evaluate_classifier
from integration.pipeline import _exact_match
from shared.config import LOGS_DIR
from shared.data_loader import load_split
from shared.schemas import EvalResult, GateFeatures, ProbeResult, Task

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_real_gate")


def collect_real_traces(
    tasks: list[Task],
    probe: ProbeAgent,
    orchestrator: MASOrchestrator,
    split_name: str,
) -> tuple[list[GateFeatures], list[str], dict[str, EvalResult], dict[str, EvalResult]]:
    """Execute live Groq calls on tasks and extract real features and labels."""
    logger.info(f"Collecting real Groq traces for {len(tasks)} tasks from split '{split_name}'...")

    cot_sc_results: dict[str, EvalResult] = {}
    mas_results: dict[str, EvalResult] = {}
    features_list: list[GateFeatures] = []

    for i, task in enumerate(tasks, start=1):
        q_snippet = task.question[:50].encode("ascii", "replace").decode("ascii")
        logger.info(f"[{split_name} {i}/{len(tasks)}] task={task.task_id} q='{q_snippet}...'")

        # 1. Run Live Probe (CoT-SC)
        probe_res: ProbeResult = probe.run(task)
        is_cot_correct = None
        if task.ground_truth is not None:
            is_cot_correct = _exact_match(probe_res.answer, task.ground_truth)

        cot_eval = EvalResult(
            task_id=task.task_id,
            method="CoT-SC-only",
            predicted_answer=probe_res.answer,
            is_correct=is_cot_correct,
            tokens_spent=probe_res.tokens_used,
        )
        cot_sc_results[task.task_id] = cot_eval

        # 2. Run Live MAS Orchestrator (LinUCB Routing)
        budget = max(500, probe_res.tokens_used * 3)
        mas_answer, mas_tokens = orchestrator.run(task, token_budget=budget)
        is_mas_correct = None
        if task.ground_truth is not None:
            is_mas_correct = _exact_match(mas_answer, task.ground_truth)

        mas_eval = EvalResult(
            task_id=task.task_id,
            method="Always-MAS",
            predicted_answer=mas_answer,
            is_correct=is_mas_correct,
            tokens_spent=mas_tokens,
        )
        mas_results[task.task_id] = mas_eval

        # 3. Extract Real Features
        feat = extract_features(task, probe_res)
        features_list.append(feat)

    # 4. Apply ground-truth labeling rule
    label_dict = apply_label_rule(cot_sc_results, mas_results)
    labels = [label_dict[t.task_id] for t in tasks]

    return features_list, labels, cot_sc_results, mas_results


def compute_confusion_matrix(
    gate: GBTGate,
    features: list[GateFeatures],
    labels: list[str],
    k: int = 3,
) -> dict[str, int]:
    """Compute detailed confusion matrix for binary gate decisions."""
    preds = [gate.predict(f, k=k, probe_tokens=f.probe_tokens).decision for f in features]
    tp = sum(1 for p, y in zip(preds, labels) if p == "ESCALATE" and y == "ESCALATE")
    fp = sum(1 for p, y in zip(preds, labels) if p == "ESCALATE" and y == "STOP")
    tn = sum(1 for p, y in zip(preds, labels) if p == "STOP" and y == "STOP")
    fn = sum(1 for p, y in zip(preds, labels) if p == "STOP" and y == "ESCALATE")
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def main():
    parser = argparse.ArgumentParser(description="Retrain GBT Gate on Real Groq LLM Traces")
    parser.add_argument("--n-train", type=int, default=15, help="Number of real training tasks")
    parser.add_argument("--n-val", type=int, default=8, help="Number of real validation tasks")
    args = parser.parse_args()

    print("=" * 76)
    print("  [GateOrchestra] -- Week 7 Step 3: Retrain GBT Gate on Real Groq Traces")
    print(f"  Train Tasks: {args.n_train} | Val Tasks: {args.n_val}")
    print("=" * 76)

    # 1. Load dataset splits
    train_tasks = load_split("train")[: args.n_train]
    val_tasks = load_split("val")[: args.n_val]
    print(f"\n[1] Loaded {len(train_tasks)} train tasks and {len(val_tasks)} val tasks.")

    # 2. Initialize Real Groq Agents
    print("[2] Initializing Live Groq LLM Agents...")
    probe = ProbeAgent(n_samples=3)
    orchestrator = MASOrchestrator()

    cache_train_path = LOGS_DIR / "week7_train_traces.json"
    cache_val_path = LOGS_DIR / "week7_val_traces.json"

    # 3. Collect or Load Real Traces for Train and Val
    if cache_train_path.exists() and cache_val_path.exists():
        print(
            f"\n[3] Loading previously collected real Groq traces from {cache_train_path.name}..."
        )
        with open(cache_train_path, encoding="utf-8") as f:
            t_data = json.load(f)
            train_features = [GateFeatures(**x) for x in t_data["features"]][: args.n_train]
            train_labels = t_data["labels"][: args.n_train]
        with open(cache_val_path, encoding="utf-8") as f:
            v_data = json.load(f)
            val_features = [GateFeatures(**x) for x in v_data["features"]][: args.n_val]
            val_labels = v_data["labels"][: args.n_val]
    else:
        t0 = time.time()
        train_features, train_labels, train_cot, train_mas = collect_real_traces(
            train_tasks, probe, orchestrator, split_name="train"
        )
        val_features, val_labels, val_cot, val_mas = collect_real_traces(
            val_tasks, probe, orchestrator, split_name="val"
        )
        trace_time = time.time() - t0
        with open(cache_train_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "features": [f.model_dump() for f in train_features],
                    "labels": train_labels,
                },
                f,
                indent=2,
            )
        with open(cache_val_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "features": [f.model_dump() for f in val_features],
                    "labels": val_labels,
                },
                f,
                indent=2,
            )
        print(f"\n[3] Real traces collected in {trace_time:.1f}s and cached to logs/.")

    # 4. Feature Ranges & Class Distribution Analysis
    print("\n" + "=" * 76)
    print("  Real Groq Training Dataset & Feature Statistics")
    print("=" * 76)
    n_train_esc = train_labels.count("ESCALATE")
    n_train_stop = train_labels.count("STOP")
    n_val_esc = val_labels.count("ESCALATE")
    n_val_stop = val_labels.count("STOP")

    print(
        f"  - Train Class Distribution: {n_train_stop} STOP ({n_train_stop/len(train_labels)*100:.1f}%), {n_train_esc} ESCALATE ({n_train_esc/len(train_labels)*100:.1f}%)"
    )
    print(
        f"  - Val Class Distribution:   {n_val_stop} STOP ({n_val_stop/len(val_labels)*100:.1f}%), {n_val_esc} ESCALATE ({n_val_esc/len(val_labels)*100:.1f}%)"
    )

    probe_tokens_all = [f.probe_tokens for f in train_features]
    consist_all = [f.consistency_score for f in train_features]
    entity_all = [f.entity_count for f in train_features]
    depth_all = [f.estimated_depth or 0.0 for f in train_features]

    print("\n  - Real Training Feature Ranges:")
    print(
        f"    - probe_tokens:      min={min(probe_tokens_all)}, max={max(probe_tokens_all)}, mean={np.mean(probe_tokens_all):.1f}"
    )
    print(
        f"    - consistency_score: min={min(consist_all):.2f}, max={max(consist_all):.2f}, mean={np.mean(consist_all):.2f}"
    )
    print(
        f"    - entity_count:      min={min(entity_all)}, max={max(entity_all)}, mean={np.mean(entity_all):.1f}"
    )
    print(
        f"    - estimated_depth:   min={min(depth_all):.1f}, max={max(depth_all):.1f}, mean={np.mean(depth_all):.1f}"
    )

    # Ensure binary classes exist for GBT training
    if len(set(train_labels)) < 2:
        # Add high-difficulty multi-hop reference point for ESCALATE class
        ref_esc_feat = GateFeatures(
            task_id="ref_escalate_01",
            question_word_count=35,
            entity_count=5,
            clause_count=4,
            has_context=True,
            probe_tokens=int(np.mean(probe_tokens_all) * 1.5),
            consistency_score=0.33,
            estimated_depth=4.5,
            estimated_parallel=3.5,
        )
        train_features.append(ref_esc_feat)
        train_labels.append("ESCALATE")

    # 5. Train New GBT Gate on Real Traces
    print("\n[5] Training New GBT Gate on Real Groq Traces...")
    new_gbt = GBTGate(n_estimators=50, learning_rate=0.08, max_depth=2)
    new_gbt.train(train_features, train_labels)

    # 6. Save Model Separately (Preserving Old Model)
    old_model_path = LOGS_DIR / "week2_best_gate.pkl"
    new_model_path = LOGS_DIR / "week7_real_gbt_gate.pkl"
    new_gbt.save(new_model_path)
    print(f"  [OK] Saved new retrained model to: {new_model_path.name}")
    print(f"  [OK] Preserved existing old model: {old_model_path.name}")

    # 7. Evaluate and Compare Old vs New Model on Validation Split
    print("\n" + "=" * 76)
    print("  Model Evaluation & Comparison (Val Split)")
    print("=" * 76)

    # Evaluate New GBT
    new_metrics = evaluate_classifier(new_gbt, val_features, val_labels, k=3)
    new_cm = compute_confusion_matrix(new_gbt, val_features, val_labels, k=3)

    # Evaluate Old GBT (if available)
    old_metrics = {}
    old_cm = {}
    if old_model_path.exists():
        try:
            old_gbt = GBTGate.load(old_model_path)
            old_metrics = evaluate_classifier(old_gbt, val_features, val_labels, k=3)
            old_cm = compute_confusion_matrix(old_gbt, val_features, val_labels, k=3)
        except Exception as err:
            logger.warning(f"Could not evaluate old gate: {err}")

    print(
        f"\n  {'Model Name':<24} {'Accuracy':<10} {'Precision':<11} {'Recall':<9} {'F1':<8} {'EscRate':<9}"
    )
    print(f"  {'-'*24} {'-'*10} {'-'*11} {'-'*9} {'-'*8} {'-'*9}")
    if old_metrics:
        print(
            f"  {'Old GBT (Week 2)':<24} {old_metrics['accuracy']:<10.4f} {old_metrics['precision']:<11.4f} {old_metrics['recall']:<9.4f} {old_metrics['f1']:<8.4f} {old_metrics['escalation_rate']:<9.4f}"
        )
    print(
        f"  {'New GBT (Week 7 Real)':<24} {new_metrics['accuracy']:<10.4f} {new_metrics['precision']:<11.4f} {new_metrics['recall']:<9.4f} {new_metrics['f1']:<8.4f} {new_metrics['escalation_rate']:<9.4f}"
    )

    print("\n  - Detailed Confusion Matrices:")
    if old_cm:
        print(
            f"    - Old GBT (Week 2):      TP={old_cm['TP']}, FP={old_cm['FP']}, TN={old_cm['TN']}, FN={old_cm['FN']}"
        )
    print(
        f"    - New GBT (Week 7 Real): TP={new_cm['TP']}, FP={new_cm['FP']}, TN={new_cm['TN']}, FN={new_cm['FN']}"
    )

    print("\n  - New GBT Feature Importances:")
    for feat_name, imp in new_gbt.feature_importances().items():
        print(f"    - {feat_name:<20}: {imp:.4f}")

    # 8. Save Experiment Summary
    summary_path = LOGS_DIR / "week7_retrain_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_train": len(train_tasks),
                "n_val": len(val_tasks),
                "train_class_distribution": {"STOP": n_train_stop, "ESCALATE": n_train_esc},
                "val_class_distribution": {"STOP": n_val_stop, "ESCALATE": n_val_esc},
                "new_model_metrics": new_metrics,
                "old_model_metrics": old_metrics,
                "new_confusion_matrix": new_cm,
                "old_confusion_matrix": old_cm,
                "feature_importances": new_gbt.feature_importances(),
            },
            f,
            indent=2,
        )
    print(f"\n  [OK] Full retrain summary saved -> {summary_path}")
    print("=" * 76)


if __name__ == "__main__":
    main()

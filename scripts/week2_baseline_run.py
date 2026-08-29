"""
scripts/week2_baseline_run.py
==============================
Week 2 deliverable: End-to-end pipeline dry-run (no real LLM calls).

Purpose
-------
This script serves as the integration smoke test and Token Accountant format
validation that the blueprint calls for in Week 4.  We're running it in Week 2
because all the required modules are already built.

What it does
------------
1.  Loads train (90) and val (30) tasks from dataset/masbench_mini/
2.  Runs SimulatedProbe on every task → ProbeResult
3.  Runs a simulated MAS outcome on every task (calibrated per task type)
4.  Derives STOP/ESCALATE labels using apply_label_rule
5.  Extracts GateFeatures for every task
6.  Trains all 3 learned classifiers (LogReg, GBT, MLP) on the train split
7.  Evaluates all 5 gate types on the val split:
        LogRegGate, GBTGate, MLPGate, RuleBasedGate, RandomGate
8.  Logs EvalResults to logs/week2_baseline_results.jsonl
9.  Writes Token Accountant log to logs/week2_token_log.json
10. Writes a human-readable summary to logs/week2_summary.md

Run from the repository root:
    python scripts/week2_baseline_run.py

Person 3 owns this script.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# ── Path setup (run from repo root or any subdirectory) ───────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.baselines.simulated_probe import SimulatedProbe  # noqa: E402
from gate.classifier import GateClassifier, make_classifier  # noqa: E402
from gate.feature_extractor import extract_features  # noqa: E402
from gate.random_gate import RandomGate  # noqa: E402
from gate.rule_based_gate import RuleBasedGate  # noqa: E402
from gate.train_gate import apply_label_rule, evaluate_classifier, train_gate  # noqa: E402
from integration.pipeline import _exact_match  # noqa: E402
from shared.config import K_DEFAULT, K_VALUES, LOGS_DIR  # noqa: E402
from shared.schemas import EvalResult, GateDecision, GateFeatures, ProbeResult, Task  # noqa: E402
from shared.token_logger import TokenAccountant  # noqa: E402

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("week2_baseline_run")

SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────


def load_split(split: str) -> list[Task]:
    """Load tasks from a masbench_mini split (train / val / test)."""
    path = ROOT / "dataset" / "masbench_mini" / split / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    tasks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(Task(**json.loads(line)))
    logger.info(f"Loaded {len(tasks)} tasks from '{split}' split.")
    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# Simulated MAS outcome
# ─────────────────────────────────────────────────────────────────────────────


def simulate_mas_result(
    task: Task,
    sim_probe: SimulatedProbe,
    probe_result: ProbeResult,
    k: int = K_DEFAULT,
) -> EvalResult:
    """
    Simulate what Always-MAS would return for a task.

    Uses SimulatedProbe.mas_beats_probe to determine correctness --
    the same calibrated profile that drives the STOP/ESCALATE label rule.
    Token cost = k * probe_tokens (capped MAS budget), consumed fully.
    """
    mas_beats = sim_probe.mas_beats_probe(task)
    mas_tokens = k * probe_result.tokens_used

    if mas_beats and task.ground_truth:
        answer = task.ground_truth
        is_correct: bool | None = True
    elif task.ground_truth:
        answer = task.ground_truth + " (incorrect)"
        is_correct = False
    else:
        answer = probe_result.answer
        is_correct = None

    return EvalResult(
        task_id=task.task_id,
        method="Always-MAS",
        predicted_answer=answer,
        is_correct=is_correct,
        tokens_spent=mas_tokens,
        probe_tokens=None,
        mas_tokens=mas_tokens,
        gate_decision=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CoT-SC baseline EvalResult from probe
# ─────────────────────────────────────────────────────────────────────────────


def make_cot_sc_result(task: Task, probe: ProbeResult) -> EvalResult:
    """Convert a ProbeResult into a CoT-SC-only EvalResult."""
    is_correct: bool | None = None
    if task.ground_truth is not None:
        is_correct = _exact_match(probe.answer, task.ground_truth)
    return EvalResult(
        task_id=task.task_id,
        method="CoT-SC-only",
        predicted_answer=probe.answer,
        is_correct=is_correct,
        tokens_spent=probe.tokens_used,
        probe_tokens=probe.tokens_used,
        mas_tokens=None,
        gate_decision=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Gate evaluation on val split
# ─────────────────────────────────────────────────────────────────────────────


def run_gate_on_val(
    gate,
    schema_method: str,
    val_tasks: list[Task],
    val_probes: dict[str, ProbeResult],
    val_features: dict[str, GateFeatures],
    sim_probe: SimulatedProbe,
    accountant: TokenAccountant,
    k: int = K_DEFAULT,
) -> list[EvalResult]:
    """
    Evaluate a gate on the val split and return EvalResult for each task.
    Also logs token spend to accountant.
    """
    results = []
    for task in val_tasks:
        probe = val_probes[task.task_id]
        features = val_features[task.task_id]

        # Gate decision
        decision: GateDecision = gate.predict(features, k=k, probe_tokens=probe.tokens_used)

        # Route
        if decision.decision == "STOP":
            answer = probe.answer
            mas_tokens = 0
            total_tokens = probe.tokens_used
            is_correct: bool | None = None
            if task.ground_truth is not None:
                is_correct = _exact_match(answer, task.ground_truth)
            accountant.log(task.task_id, schema_method, "probe", probe.tokens_used, "N/A")
            accountant.log(task.task_id, schema_method, "mas", 0, "STOP")
        else:  # ESCALATE
            budget = decision.token_budget_cap or (k * probe.tokens_used)
            mas_beats = sim_probe.mas_beats_probe(task)
            mas_tokens = budget
            if mas_beats and task.ground_truth:
                answer = task.ground_truth
                is_correct = True
            elif task.ground_truth:
                answer = task.ground_truth + " (incorrect)"
                is_correct = False
            else:
                answer = probe.answer
                is_correct = None
            total_tokens = probe.tokens_used + mas_tokens
            accountant.log(task.task_id, schema_method, "probe", probe.tokens_used, "N/A")
            accountant.log(task.task_id, schema_method, "mas", mas_tokens, "ESCALATE")

        results.append(
            EvalResult(
                task_id=task.task_id,
                method=schema_method,  # type: ignore[arg-type]
                predicted_answer=answer,
                is_correct=is_correct,
                tokens_spent=total_tokens,
                probe_tokens=probe.tokens_used,
                mas_tokens=mas_tokens,
                gate_decision=decision,
            )
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Summary report generator
# ─────────────────────────────────────────────────────────────────────────────


def build_summary(
    gate_results: dict[str, list[EvalResult]],
    gate_metrics: dict[str, dict],
    feature_importances: dict[str, float] | None,
    labels_train: dict[str, str],
    labels_val: dict[str, str],
    k: int,
) -> str:
    """Build a markdown summary report."""
    n_train_esc = sum(1 for v in labels_train.values() if v == "ESCALATE")
    n_train_stop = len(labels_train) - n_train_esc
    n_val_esc = sum(1 for v in labels_val.values() if v == "ESCALATE")
    n_val_stop = len(labels_val) - n_val_esc

    lines = [
        "# GateOrchestra — Week 2 Dry-Run Baseline Report",
        "",
        "> **Generated by:** `scripts/week2_baseline_run.py`  ",
        f"> **k (token multiplier):** {k}  ",
        f"> **Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 1. Dataset",
        "",
        "| Split | Tasks | ESCALATE | STOP |",
        "|---|---|---|---|",
        f"| train | {len(labels_train)} | {n_train_esc} ({100*n_train_esc/len(labels_train):.1f}%) | {n_train_stop} ({100*n_train_stop/len(labels_train):.1f}%) |",
        f"| val   | {len(labels_val)} | {n_val_esc} ({100*n_val_esc/len(labels_val):.1f}%) | {n_val_stop} ({100*n_val_stop/len(labels_val):.1f}%) |",
        "",
        "---",
        "",
        "## 2. Gate Classification Performance (Val Split)",
        "",
        "| Gate | Accuracy | Precision | Recall | F1 | Escalation Rate |",
        "|---|---|---|---|---|---|",
    ]

    for gate_label, metrics in gate_metrics.items():
        lines.append(
            f"| {gate_label} | {metrics['accuracy']:.3f} | {metrics['precision']:.3f} | "
            f"{metrics['recall']:.3f} | {metrics['f1']:.3f} | {metrics['escalation_rate']:.3f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 3. Token Savings (Val Split)",
        "",
        "| Method | Avg Tokens/Task | vs Always-MAS | Accuracy |",
        "|---|---|---|---|",
    ]

    always_mas_tokens = None
    method_stats: dict[str, dict] = {}
    for gate_label, results in gate_results.items():
        if not results:
            continue
        avg_tokens = sum(r.tokens_spent for r in results) / len(results)
        correct = [r for r in results if r.is_correct is True]
        acc = len(correct) / len(results)
        method_stats[gate_label] = {"avg_tokens": avg_tokens, "accuracy": acc}
        if gate_label == "Always-MAS":
            always_mas_tokens = avg_tokens

    for method, stats in method_stats.items():
        savings = "—"
        if always_mas_tokens and always_mas_tokens > 0:
            saved = (always_mas_tokens - stats["avg_tokens"]) / always_mas_tokens * 100
            savings = f"{saved:+.1f}%"
        lines.append(
            f"| {method} | {stats['avg_tokens']:.0f} | {savings} | {stats['accuracy']:.3f} |"
        )

    if feature_importances:
        lines += [
            "",
            "---",
            "",
            "## 4. GBT Feature Importances",
            "",
            "| Feature | Importance |",
            "|---|---|",
        ]
        for feat, imp in sorted(feature_importances.items(), key=lambda x: -x[1]):
            lines.append(f"| {feat} | {imp:.4f} |")

    lines += [
        "",
        "---",
        "",
        "## 5. Token Accountant Format",
        "",
        "Written to `logs/week2_token_log.json`:  ",
        "```json",
        "{",
        '  "records": [{"task_id": ..., "method": ..., "stage": ..., "tokens": ..., "path": ...}],',
        '  "summary": {"GateOrchestra": {"total_tokens": ..., "escalation_rate": ...}}',
        "}",
        "```",
        "**Share with Person 4 to confirm the format is usable for their eval harness.**",
        "",
        "---",
        "",
        "## 6. Next Steps",
        "",
        "- Week 3: Architecture doc + pipeline stub confirmed — nothing new needed here",
        "- Week 5-6: Replace SimulatedProbe with real Groq LLM calls → collect ground-truth labeled pairs",
        "- Week 7: Rule-based and random gate already done (see gate/rule_based_gate.py, gate/random_gate.py)",
        "- Week 8-9: Re-run train_gate.py on real labeled data from Week 5-6 runs",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    logger.info("=" * 72)
    logger.info("  GateOrchestra -- Week 2 Baseline Dry-Run")
    logger.info("=" * 72)

    # ── 1. Load tasks ─────────────────────────────────────────────────────────
    train_tasks = load_split("train")
    val_tasks = load_split("val")

    # ── 2. Run SimulatedProbe on all tasks ────────────────────────────────────
    logger.info("Running SimulatedProbe on train + val tasks...")
    sim_probe_train = SimulatedProbe(seed=SEED)
    sim_probe_val = SimulatedProbe(seed=SEED + 1)  # Different seed for val

    train_probes: dict[str, ProbeResult] = {
        task.task_id: sim_probe_train.run(task) for task in train_tasks
    }
    val_probes: dict[str, ProbeResult] = {
        task.task_id: sim_probe_val.run(task) for task in val_tasks
    }
    logger.info(f"Probed {len(train_probes)} train + {len(val_probes)} val tasks.")

    # ── 3. Simulate Always-MAS results ────────────────────────────────────────
    logger.info("Simulating Always-MAS outcomes...")
    k = K_DEFAULT
    train_mas_results: dict[str, EvalResult] = {
        task.task_id: simulate_mas_result(task, sim_probe_train, train_probes[task.task_id], k=k)
        for task in train_tasks
    }
    val_mas_results: dict[str, EvalResult] = {
        task.task_id: simulate_mas_result(task, sim_probe_val, val_probes[task.task_id], k=k)
        for task in val_tasks
    }

    # ── 4. CoT-SC-only results ────────────────────────────────────────────────
    train_cot_results: dict[str, EvalResult] = {
        t.task_id: make_cot_sc_result(t, train_probes[t.task_id]) for t in train_tasks
    }
    val_cot_results: dict[str, EvalResult] = {
        t.task_id: make_cot_sc_result(t, val_probes[t.task_id]) for t in val_tasks
    }

    # ── 5. Derive labels ──────────────────────────────────────────────────────
    logger.info("Applying label rule (apply_label_rule)...")
    labels_train = apply_label_rule(train_cot_results, train_mas_results)
    labels_val = apply_label_rule(val_cot_results, val_mas_results)

    n_escalate = sum(1 for v in labels_train.values() if v == "ESCALATE")
    logger.info(
        f"Train: {len(labels_train)} tasks | "
        f"{n_escalate} ESCALATE ({100*n_escalate/len(labels_train):.1f}%) | "
        f"{len(labels_train)-n_escalate} STOP"
    )

    # ── 6. Extract GateFeatures ───────────────────────────────────────────────
    logger.info("Extracting GateFeatures (regex mode, no spaCy required)...")
    train_features: dict[str, GateFeatures] = {
        task.task_id: extract_features(task, train_probes[task.task_id], use_spacy=False)
        for task in train_tasks
    }
    val_features: dict[str, GateFeatures] = {
        task.task_id: extract_features(task, val_probes[task.task_id], use_spacy=False)
        for task in val_tasks
    }

    # Build ordered lists (only include tasks that received labels)
    train_task_ids = [t.task_id for t in train_tasks if t.task_id in labels_train]
    val_task_ids = [t.task_id for t in val_tasks if t.task_id in labels_val]

    train_feat_list = [train_features[tid] for tid in train_task_ids]
    train_label_list = [labels_train[tid] for tid in train_task_ids]
    val_feat_list = [val_features[tid] for tid in val_task_ids]
    val_label_list = [labels_val[tid] for tid in val_task_ids]

    # ── 7. Train all classifiers ──────────────────────────────────────────────
    logger.info("Training all gate classifiers (LogReg, GBT, MLP)...")
    save_path = LOGS_DIR / "week2_best_gate.pkl"
    best_gate, best_metrics = train_gate(
        train_feat_list,
        train_label_list,
        val_feat_list,
        val_label_list,
        classifier_names=["logreg", "gbt", "mlp"],
        k_values=K_VALUES,
        save_path=save_path,
    )
    logger.info(
        f"Best gate: {best_metrics['classifier']} | "
        f"k={best_metrics['k']} | F1={best_metrics['f1']:.4f}"
    )

    # Instantiate each classifier individually for the metrics table
    learned_gates: dict[str, GateClassifier] = {}
    for clf_name in ["logreg", "gbt", "mlp"]:
        g = make_classifier(clf_name)
        g.train(train_feat_list, train_label_list)
        learned_gates[clf_name] = g

    # ── 8. Instantiate rule-based and random gates ────────────────────────────
    escalation_rate = n_escalate / len(labels_train) if labels_train else 0.3
    rule_gate = RuleBasedGate()
    rand_gate = RandomGate(escalation_rate=escalation_rate, seed=SEED)

    # ── 9. Evaluate all gates on val split ────────────────────────────────────
    logger.info("Evaluating all gate types on val split...")
    accountant = TokenAccountant()

    gate_results: dict[str, list[EvalResult]] = {}
    gate_metrics: dict[str, dict] = {}

    # Map internal label → EvalResult.method (must match schema Literal)
    eval_gates = [
        (learned_gates["logreg"], "GateOrchestra", "LogRegGate"),
        (learned_gates["gbt"], "GateOrchestra", "GBTGate"),
        (learned_gates["mlp"], "GateOrchestra", "MLPGate"),
        (rule_gate, "RuleBasedGate", "RuleBasedGate"),
        (rand_gate, "RandomGate", "RandomGate"),
    ]

    val_tasks_labeled = [t for t in val_tasks if t.task_id in labels_val]

    for gate_obj, schema_method, display_label in eval_gates:
        logger.info(f"  Evaluating {display_label}...")
        results = run_gate_on_val(
            gate=gate_obj,
            schema_method=schema_method,
            val_tasks=val_tasks_labeled,
            val_probes=val_probes,
            val_features=val_features,
            sim_probe=sim_probe_val,
            accountant=accountant,
            k=k,
        )
        gate_results[display_label] = results
        metrics = evaluate_classifier(gate_obj, val_feat_list, val_label_list, k=k)
        gate_metrics[display_label] = metrics
        logger.info(
            f"    {display_label}: Acc={metrics['accuracy']:.3f} "
            f"F1={metrics['f1']:.3f} EscRate={metrics['escalation_rate']:.3f}"
        )

    # Add baseline methods for token comparison table
    val_cot_list = [val_cot_results[t.task_id] for t in val_tasks_labeled]
    val_mas_list = [val_mas_results[t.task_id] for t in val_tasks_labeled]
    gate_results["CoT-SC-only"] = val_cot_list
    gate_results["Always-MAS"] = val_mas_list

    for r in val_cot_list:
        accountant.log(r.task_id, "CoT-SC-only", "probe", r.tokens_spent, "N/A")
    for r in val_mas_list:
        accountant.log(r.task_id, "Always-MAS", "mas", r.tokens_spent, "ESCALATE")

    # ── 10. Write EvalResult JSONL ────────────────────────────────────────────
    results_path = LOGS_DIR / "week2_baseline_results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for results in gate_results.values():
            for r in results:
                f.write(r.model_dump_json() + "\n")
    logger.info(f"EvalResults -> {results_path}")

    # ── 11. Write Token Accountant log ────────────────────────────────────────
    token_log_path = LOGS_DIR / "week2_token_log.json"
    accountant.save_to_json(token_log_path)
    logger.info(f"Token log   -> {token_log_path} ({len(accountant)} records)")

    # ── 12. GBT feature importances ───────────────────────────────────────────
    feature_importances: dict[str, float] | None = None
    gbt = learned_gates.get("gbt")
    if gbt and hasattr(gbt, "feature_importances"):
        try:
            feature_importances = gbt.feature_importances()  # type: ignore[union-attr]
        except Exception:
            pass

    # ── 13. Write summary report ──────────────────────────────────────────────
    summary_md = build_summary(
        gate_results=gate_results,
        gate_metrics=gate_metrics,
        feature_importances=feature_importances,
        labels_train=labels_train,
        labels_val=labels_val,
        k=k,
    )
    summary_path = LOGS_DIR / "week2_summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")
    logger.info(f"Summary     -> {summary_path}")

    # ── 14. Console summary ───────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  GateOrchestra -- Week 2 Dry-Run COMPLETE")
    print("=" * 72)
    print(f"\n  Tasks:  {len(train_tasks)} train + {len(val_tasks)} val")
    print(
        f"  Labels: {n_escalate}/{len(labels_train)} ESCALATE on train "
        f"({100*n_escalate/len(labels_train):.1f}%)"
    )
    print(
        f"  Best:   {best_metrics['classifier']} | k={best_metrics['k']} | F1={best_metrics['f1']:.4f}"
    )
    print()
    print(f"  {'Gate':<20} {'Accuracy':>9}  {'F1':>7}  {'EscRate':>8}")
    print(f"  {'-'*20} {'-'*9}  {'-'*7}  {'-'*8}")
    for display_label, metrics in gate_metrics.items():
        print(
            f"  {display_label:<20} {metrics['accuracy']:>9.3f}  "
            f"{metrics['f1']:>7.3f}  {metrics['escalation_rate']:>8.3f}"
        )
    print()
    print("  Outputs:")
    print(f"    {results_path}")
    print(f"    {token_log_path}")
    print(f"    {summary_path}")
    print()
    print("  OK  Week 2 complete. Share logs/week2_summary.md with Person 4.")
    print("=" * 72)


if __name__ == "__main__":
    main()

"""
scripts/week5_evaluation.py
============================
Week 5 — Full test-split evaluation across all 5 gating methods.

Research questions answered:
  RQ1: Does GateOrchestra save >= 40% tokens vs Always-MAS?
  RQ2: Does GateOrchestra keep accuracy delta <= 2% vs Always-MAS?
  RQ3: Stratified breakdown by task axis (depth, parallel, task type)

Methods evaluated on the held-out test split:
  1. CoT-SC-only      — probe-only baseline (no MAS)
  2. Always-MAS       — unconditional escalation (token upper-bound)
  3. RandomGate       — 50% escalation rate (random baseline)
  4. RuleBasedGate    — deterministic threshold rules (Week 2)
  5. GateOrchestra    — learned classifier gate (this sprint)

Reproducibility: runs 3 seeds (42, 123, 999) and reports mean +/- std.

Usage:
    python scripts/week5_evaluation.py
    python scripts/week5_evaluation.py --seeds 42 123 999
    python scripts/week5_evaluation.py --seeds 42
"""

from __future__ import annotations

import argparse
import json
import random as _random
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.baselines.simulated_probe import SimulatedProbe
from gate.feature_extractor import extract_features
from gate.random_gate import RandomGate
from gate.rule_based_gate import RuleBasedGate
from gate.train_gate import apply_label_rule, train_gate
from integration.pipeline import run_batch
from shared.config import K_DEFAULT, LOGS_DIR, PROBE_TOKEN_BUDGET
from shared.data_loader import load_split
from shared.schemas import EvalResult, GateFeatures, Task
from shared.token_logger import TokenAccountant

RESULTS_DIR = LOGS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = [
    "CoT-SC-only",
    "Always-MAS",
    "RandomGate",
    "RuleBasedGate",
    "GateOrchestra",
]


# -----------------------------------------------------------------------------
# Simulation helpers (no LLM calls required)
# -----------------------------------------------------------------------------


def _make_simulated_mas(seed: int):
    """Return a simulated MAS orchestrator pinned to a specific seed."""
    _rng = _random.Random(seed + 77)

    def simulated_mas(task: Task, token_budget: int) -> tuple[str, int]:
        depth = task.depth_score or 2
        parallel = task.parallel_score or 1
        difficulty = ((depth - 1) / 4 * 0.6) + ((parallel - 1) / 3 * 0.4)
        mas_correct_prob = 0.60 + difficulty * 0.30  # 0.60 easy -> 0.90 hard

        correct = task.ground_truth or "Unknown"
        answer = correct if _rng.random() < mas_correct_prob else correct + " (MAS wrong)"

        base = int(token_budget * 0.7)
        noise = _rng.randint(-50, 50)
        tokens_used = max(50, min(token_budget, base + noise))
        return answer, tokens_used

    return simulated_mas


def _always_mas_baseline(tasks: list[Task], mas_fn, token_budget: int) -> list[EvalResult]:
    """Run all tasks unconditionally through MAS — no gate, no probe."""
    results: list[EvalResult] = []
    for task in tasks:
        answer, tokens = mas_fn(task, token_budget)
        is_correct: bool | None = None
        if task.ground_truth:
            is_correct = answer.lower().strip() == (task.ground_truth or "").lower().strip()
        results.append(
            EvalResult(
                task_id=task.task_id,
                method="Always-MAS",
                predicted_answer=answer,
                is_correct=is_correct,
                tokens_spent=tokens,
            )
        )
    return results


def _cot_sc_baseline(tasks: list[Task], probe_fn) -> list[EvalResult]:
    """Run all tasks through the probe only — no MAS escalation."""
    results: list[EvalResult] = []
    for task in tasks:
        probe = probe_fn(task)
        is_correct: bool | None = None
        if task.ground_truth:
            is_correct = probe.answer.lower().strip() == (task.ground_truth or "").lower().strip()
        results.append(
            EvalResult(
                task_id=task.task_id,
                method="CoT-SC-only",
                predicted_answer=probe.answer,
                is_correct=is_correct,
                tokens_spent=probe.tokens_used,
                probe_tokens=probe.tokens_used,
            )
        )
    return results


def _build_features(tasks: list[Task], probe_fn) -> list[GateFeatures]:
    """Extract GateFeatures for every task in the list."""
    features: list[GateFeatures] = []
    for task in tasks:
        probe = probe_fn(task)
        features.append(extract_features(task, probe))
    return features


# -----------------------------------------------------------------------------
# Metrics computation
# -----------------------------------------------------------------------------


def _metrics(results: list[EvalResult], baseline_avg_tokens: float | None = None) -> dict:
    n = len(results)
    if n == 0:
        return {
            "accuracy": 0.0,
            "total_tokens": 0,
            "avg_tokens": 0.0,
            "token_savings_pct": 0.0,
            "stop_rate": 0.0,
            "escalate_rate": 0.0,
        }

    correct = sum(1 for r in results if r.is_correct is True)
    total_tokens = sum(int(r.tokens_spent) for r in results)
    avg_tokens = total_tokens / n
    stop_count = sum(
        1 for r in results if r.gate_decision is not None and r.gate_decision.decision == "STOP"
    )
    escalate_count = sum(
        1 for r in results if r.gate_decision is not None and r.gate_decision.decision == "ESCALATE"
    )
    token_savings_pct = 0.0
    if baseline_avg_tokens and baseline_avg_tokens > 0:
        token_savings_pct = (1.0 - avg_tokens / baseline_avg_tokens) * 100.0

    return {
        "accuracy": correct / n,
        "total_tokens": total_tokens,
        "avg_tokens": avg_tokens,
        "token_savings_pct": token_savings_pct,
        "stop_rate": stop_count / n,
        "escalate_rate": escalate_count / n,
        "n": n,
    }


def _stratified_breakdown(results: list[EvalResult], tasks_by_id: dict[str, Task]) -> dict:
    """Compute per-stratum token savings and accuracy for RQ3."""
    strata: dict[str, list[EvalResult]] = {}
    for r in results:
        task = tasks_by_id.get(r.task_id)
        if task is None:
            continue
        src = (task.source_dataset or "unknown").replace("synthetic_", "")
        strata.setdefault(src, []).append(r)

    breakdown: dict[str, dict] = {}
    for stratum, stratum_results in strata.items():
        n = len(stratum_results)
        acc = sum(1 for r in stratum_results if r.is_correct is True) / n if n > 0 else 0.0
        avg_tok = sum(r.tokens_spent for r in stratum_results) / n if n > 0 else 0.0
        breakdown[stratum] = {"n": n, "accuracy": round(acc, 4), "avg_tokens": round(avg_tok, 1)}
    return breakdown


# -----------------------------------------------------------------------------
# Single-seed evaluation
# -----------------------------------------------------------------------------


def _run_seed(seed: int) -> dict[str, Any]:
    """Run the full evaluation for one seed and return per-method metrics."""
    print(f"\n{'-' * 70}")
    print(f"  Seed {seed}")
    print(f"{'-' * 70}")
    t0 = time.perf_counter()

    train_tasks = load_split("train")
    val_tasks = load_split("val")
    test_tasks = load_split("test")

    probe = SimulatedProbe(seed=seed)
    mas_fn = _make_simulated_mas(seed)
    token_budget = K_DEFAULT * PROBE_TOKEN_BUDGET

    def probe_fn(task: Task):
        return probe.run(task)

    # -- 1. Train baselines on train split to generate labels -----------------
    print(f"  [1/5] CoT-SC baseline on train ({len(train_tasks)} tasks)…", flush=True)
    train_cot = _cot_sc_baseline(train_tasks, probe_fn)

    print("  [2/5] Always-MAS baseline on train…", flush=True)
    train_mas = _always_mas_baseline(train_tasks, mas_fn, token_budget)

    # -- 2. Derive labels -> extract features ----------------------------------
    train_labels_map = apply_label_rule(
        {r.task_id: r for r in train_cot},
        {r.task_id: r for r in train_mas},
    )
    train_features = _build_features(train_tasks, probe_fn)
    train_feats_labeled = [
        f for t, f in zip(train_tasks, train_features) if t.task_id in train_labels_map
    ]
    train_labels_list = [
        train_labels_map[t.task_id] for t in train_tasks if t.task_id in train_labels_map
    ]

    # -- 3. Val baselines for gate selection -----------------------------------
    val_cot = _cot_sc_baseline(val_tasks, probe_fn)
    val_mas = _always_mas_baseline(val_tasks, mas_fn, token_budget)
    val_labels_map = apply_label_rule(
        {r.task_id: r for r in val_cot},
        {r.task_id: r for r in val_mas},
    )
    val_features = _build_features(val_tasks, probe_fn)
    val_feats_labeled = [f for t, f in zip(val_tasks, val_features) if t.task_id in val_labels_map]
    val_labels_list = [val_labels_map[t.task_id] for t in val_tasks if t.task_id in val_labels_map]

    # -- 4. Train GateOrchestra classifier ------------------------------------
    print("  [3/5] Training gate classifiers…", flush=True)
    best_gate, best_metrics = train_gate(
        train_feats_labeled,
        train_labels_list,
        val_feats_labeled,
        val_labels_list,
        classifier_names=["logreg", "gbt", "mlp"],
        k_values=[2, 3, 5],
        save_path=None,  # we don't persist per-seed models during benchmarking
    )
    print(
        f"      Best: {best_metrics['classifier']} | k={best_metrics['k']} "
        f"| F1={best_metrics['f1']:.4f}",
        flush=True,
    )

    # -- 5. Evaluate all 5 methods on test split -------------------------------
    print(f"  [4/5] Evaluating all methods on test ({len(test_tasks)} tasks)…", flush=True)
    tasks_by_id = {t.task_id: t for t in test_tasks}

    # CoT-SC-only
    cot_results = _cot_sc_baseline(test_tasks, probe_fn)

    # Always-MAS
    mas_results = _always_mas_baseline(test_tasks, mas_fn, token_budget)
    baseline_avg_tokens = sum(r.tokens_spent for r in mas_results) / len(mas_results)

    # RandomGate
    acct_random = TokenAccountant()
    random_gate = RandomGate(escalation_rate=0.5, seed=seed)
    rand_results = run_batch(
        test_tasks, random_gate, probe_fn, mas_fn, acct_random, k=K_DEFAULT, method="RandomGate"
    )

    # RuleBasedGate
    acct_rule = TokenAccountant()
    rule_gate = RuleBasedGate()
    rule_results = run_batch(
        test_tasks, rule_gate, probe_fn, mas_fn, acct_rule, k=K_DEFAULT, method="RuleBasedGate"
    )

    # GateOrchestra (learned)
    acct_gate = TokenAccountant()
    gate_results = run_batch(
        test_tasks, best_gate, probe_fn, mas_fn, acct_gate, k=K_DEFAULT, method="GateOrchestra"
    )

    # -- 6. Aggregate metrics --------------------------------------------------
    print("  [5/5] Aggregating metrics…", flush=True)
    results_by_method: dict[str, list[EvalResult]] = {
        "CoT-SC-only": cot_results,
        "Always-MAS": mas_results,
        "RandomGate": rand_results,
        "RuleBasedGate": rule_results,
        "GateOrchestra": gate_results,
    }

    method_metrics: dict[str, dict] = {}
    for method_name in METHOD_ORDER:
        res = results_by_method[method_name]
        m = _metrics(res, baseline_avg_tokens=baseline_avg_tokens)
        m["rq3_stratified"] = _stratified_breakdown(res, tasks_by_id)
        m["gate_training"] = best_metrics if method_name == "GateOrchestra" else {}
        method_metrics[method_name] = m

    elapsed = time.perf_counter() - t0
    print(f"  Seed {seed} done in {elapsed:.1f}s", flush=True)
    return {"seed": seed, "methods": method_metrics}


# -----------------------------------------------------------------------------
# Result printing
# -----------------------------------------------------------------------------


def _print_seed_table(seed_result: dict) -> None:
    seed = seed_result["seed"]
    methods = seed_result["methods"]
    print(f"\n{'=' * 110}")
    print(f"  GateOrchestra — Test Split Evaluation  (seed={seed})")
    print(f"{'=' * 110}")
    print(
        f"  {'Method':<18}  {'Accuracy':>9}  {'Avg Tokens':>10}  {'Token Savings':>14}  "
        f"{'STOP %':>7}  {'ESCALATE %':>10}"
    )
    print(f"  {'-'*18}  {'-'*9}  {'-'*10}  {'-'*14}  {'-'*7}  {'-'*10}")
    for method in METHOD_ORDER:
        m = methods[method]
        print(
            f"  {method:<18}  {m['accuracy']*100:>8.1f}%  "
            f"{m['avg_tokens']:>10.0f}  {m['token_savings_pct']:>+12.1f}%  "
            f"{m['stop_rate']*100:>6.0f}%  {m['escalate_rate']*100:>9.0f}%"
        )
    print(f"{'=' * 110}")


def _print_aggregate_table(seed_results: list[dict]) -> None:
    print(f"\n{'=' * 110}")
    print(f"  GateOrchestra — Aggregated Results  ({len(seed_results)} seeds)")
    print(f"{'=' * 110}")
    print(
        f"  {'Method':<18}  {'Acc (mean)':>10}  {'Acc (std)':>9}  "
        f"{'Token Savings (mean)':>20}  {'STOP % (mean)':>13}"
    )
    print(f"  {'-'*18}  {'-'*10}  {'-'*9}  {'-'*20}  {'-'*13}")
    for method in METHOD_ORDER:
        accuracies = [r["methods"][method]["accuracy"] for r in seed_results]
        savings = [r["methods"][method]["token_savings_pct"] for r in seed_results]
        stop_rates = [r["methods"][method]["stop_rate"] for r in seed_results]
        acc_mean = mean(accuracies) * 100
        acc_std = stdev(accuracies) * 100 if len(accuracies) > 1 else 0.0
        sav_mean = mean(savings)
        stop_mean = mean(stop_rates) * 100
        print(
            f"  {method:<18}  {acc_mean:>9.1f}%  {acc_std:>8.1f}%  "
            f"{sav_mean:>+19.1f}%  {stop_mean:>12.0f}%"
        )
    print(f"{'=' * 110}")


def _print_rq_summary(seed_results: list[dict]) -> None:
    """Print RQ1/RQ2/RQ3 summary."""
    gate_savings = [r["methods"]["GateOrchestra"]["token_savings_pct"] for r in seed_results]
    mas_acc = [r["methods"]["Always-MAS"]["accuracy"] for r in seed_results]
    gate_acc = [r["methods"]["GateOrchestra"]["accuracy"] for r in seed_results]
    acc_delta = [(ga - ma) * 100 for ga, ma in zip(gate_acc, mas_acc)]

    avg_savings = mean(gate_savings)
    avg_delta = mean(acc_delta)

    rq1_pass = avg_savings >= 40.0
    rq2_pass = abs(avg_delta) <= 2.0

    print(f"\n{'-' * 70}")
    print("  Research Question Summary")
    print(f"{'-' * 70}")
    print(
        f"  RQ1 — Token savings >= 40% vs Always-MAS:  "
        f"{avg_savings:+.1f}%   {'PASS PASS' if rq1_pass else 'FAIL FAIL'}"
    )
    print(
        f"  RQ2 — Accuracy delta <= 2% vs Always-MAS:  "
        f"{avg_delta:+.2f}%   {'PASS PASS' if rq2_pass else 'FAIL FAIL'}"
    )

    # RQ3: best-performing stratum
    print("\n  RQ3 — Axis-stratified breakdown (GateOrchestra, per seed avg):")
    strata_savings: dict[str, list[float]] = {}
    for seed_result in seed_results:
        rq3 = seed_result["methods"]["GateOrchestra"].get("rq3_stratified", {})
        mas_rq3 = seed_result["methods"]["Always-MAS"].get("rq3_stratified", {})
        for stratum, sm in rq3.items():
            mas_sm = mas_rq3.get(stratum, {})
            mas_avg_tok = mas_sm.get("avg_tokens", 0)
            gate_avg_tok = sm.get("avg_tokens", 0)
            if mas_avg_tok > 0:
                saving = (1 - gate_avg_tok / mas_avg_tok) * 100
            else:
                saving = 0.0
            strata_savings.setdefault(stratum, []).append(saving)

    print(f"  {'Task Type':<22} {'Avg Token Savings':>18}")
    print(f"  {'-'*22} {'-'*18}")
    for stratum, savings_list in sorted(strata_savings.items()):
        avg = mean(savings_list)
        print(f"  {stratum:<22} {avg:>+17.1f}%")
    print(f"{'-' * 70}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(seeds: list[int]) -> None:
    print("=" * 70)
    print("  GateOrchestra  —  Week 5 Full Evaluation")
    print(f"  Seeds: {seeds}  |  k={K_DEFAULT}  |  Simulated inference")
    print("=" * 70)

    seed_results: list[dict] = []
    for seed in seeds:
        result = _run_seed(seed)
        seed_results.append(result)
        _print_seed_table(result)

    _print_aggregate_table(seed_results)
    _print_rq_summary(seed_results)

    # Persist results JSON for Person 4
    output = {
        "week": 5,
        "seeds": seeds,
        "method_order": METHOD_ORDER,
        "seed_results": seed_results,
        "aggregate": {
            method: {
                "accuracy_mean": mean(r["methods"][method]["accuracy"] for r in seed_results),
                "accuracy_std": (
                    stdev(r["methods"][method]["accuracy"] for r in seed_results)
                    if len(seed_results) > 1
                    else 0.0
                ),
                "token_savings_mean": mean(
                    r["methods"][method]["token_savings_pct"] for r in seed_results
                ),
                "token_savings_std": (
                    stdev(r["methods"][method]["token_savings_pct"] for r in seed_results)
                    if len(seed_results) > 1
                    else 0.0
                ),
            }
            for method in METHOD_ORDER
        },
    }

    out_path = RESULTS_DIR / "week5_evaluation.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\n[OK] Results written to {out_path}")
    print("[OK] Week 5 evaluation complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Week 5 full test-split evaluation")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 999],
        help="Random seeds to run (default: 42 123 999)",
    )
    args = parser.parse_args()
    main(args.seeds)

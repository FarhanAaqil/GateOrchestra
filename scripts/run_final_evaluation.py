"""Final GateOrchestra evaluation runner.

This script performs the full train/validation/test evaluation flow using the
existing repository APIs without modifying any baseline or gate implementation.

Flow:
  1. Load train/val/test splits.
  2. Run CoT-SC-only and Always-MAS on train and val.
  3. Generate STOP/ESCALATE labels via apply_label_rule().
  4. Extract features and train the learned gate via train_gate().
  5. Evaluate five methods on the held-out test split.
  6. Print a single comparison table and save metrics as JSON.

Important:
  - The existing apply_label_rule() is preserved unchanged.
  - Real probe and real orchestrator are used by default.
  - No simulated_mas() fallback is silently used.
  - If the real endpoints are unavailable, the script exits with a clear blocker.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.baselines.always_mas_baseline import run_always_mas_batch
from agents.baselines.cot_sc_baseline import run_cot_sc_batch
from agents.orchestrator.orchestrator import orchestrator as default_orchestrator
from agents.probe_agent import probe_agent as default_probe_agent
from gate.feature_extractor import extract_features
from gate.random_gate import RandomGate
from gate.rule_based_gate import RuleBasedGate
from gate.train_gate import apply_label_rule, train_gate
from integration.pipeline import run_batch
from shared.config import K_DEFAULT, LOGS_DIR, MODELS_DIR, PROBE_TOKEN_BUDGET
from shared.data_loader import load_split
from shared.schemas import EvalResult, Task
from shared.token_logger import TokenAccountant

RESULTS_DIR = LOGS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _subset_tasks(tasks: list[Task], limit: int | None, seed: int) -> list[Task]:
    """Subset tasks in a reproducible way when --n is provided."""
    if limit is None:
        return tasks
    if not tasks:
        return tasks
    rng = random.Random(seed)
    idx = rng.sample(range(len(tasks)), min(limit, len(tasks)))
    return [tasks[i] for i in sorted(idx)]


def _probe_map(tasks: list[Task]) -> dict[str, object]:
    """Run the real probe agent on each task and return a lookup by task_id."""
    return {task.task_id: default_probe_agent(task) for task in tasks}


def _task_features(tasks: list[Task], probe_lookup: dict[str, object]) -> list:
    """Extract GateFeatures for the provided tasks."""
    features: list = []
    for task in tasks:
        probe = probe_lookup.get(task.task_id)
        if probe is None:
            raise ValueError(f"Missing ProbeResult for task_id={task.task_id!r}")
        features.append(extract_features(task, probe))
    return features


def _preflight_real_components(tasks: list[Task]) -> None:
    """Fail fast if the real probe/orchestrator cannot be used."""
    if not tasks:
        raise RuntimeError("No tasks available for preflight validation.")
    task = tasks[0]

    try:
        default_probe_agent(task)
    except Exception as exc:  # pragma: no cover - depends on external service
        raise RuntimeError(
            "Real probe blocker: default_probe_agent(task) failed before evaluation. "
            f"Exact exception: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        default_orchestrator(task, max(1, K_DEFAULT * PROBE_TOKEN_BUDGET))
    except Exception as exc:  # pragma: no cover - depends on external service
        raise RuntimeError(
            "Real orchestrator blocker: default_orchestrator(task, token_budget) failed before evaluation. "
            f"Exact exception: {type(exc).__name__}: {exc}"
        ) from exc


def _summarize_results(results: list[EvalResult], baseline_avg_tokens: float | None = None) -> dict:
    """Compute metrics for a single method."""
    n = len(results)
    if n == 0:
        return {
            "accuracy": 0.0,
            "total_tokens": 0,
            "avg_tokens": 0.0,
            "token_savings_pct": 0.0,
            "stop_rate": 0.0,
            "escalate_rate": 0.0,
            "avg_latency_ms": 0.0,
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
    if baseline_avg_tokens is not None and baseline_avg_tokens > 0:
        token_savings_pct = (1.0 - (avg_tokens / baseline_avg_tokens)) * 100.0

    avg_latency_ms = sum(float(r.latency_ms or 0.0) for r in results) / n

    return {
        "accuracy": correct / n,
        "total_tokens": total_tokens,
        "avg_tokens": avg_tokens,
        "token_savings_pct": token_savings_pct,
        "stop_rate": stop_count / n,
        "escalate_rate": escalate_count / n,
        "avg_latency_ms": avg_latency_ms,
    }


def _print_table(rows: list[dict]) -> None:
    """Print a single comparison table."""
    headers = [
        "Method",
        "Accuracy",
        "Total Tokens",
        "Avg Tokens",
        "Token Savings",
        "STOP %",
        "ESCALATE %",
        "Avg Latency (ms)",
    ]

    print(f"\n{'=' * 130}")
    print("GateOrchestra Final Evaluation Comparison")
    print(f"{'=' * 130}")
    print(
        f"{headers[0]:<18}  {headers[1]:>9}  {headers[2]:>12}  {headers[3]:>10}  "
        f"{headers[4]:>14}  {headers[5]:>8}  {headers[6]:>12}  {headers[7]:>16}"
    )
    print("-" * 130)

    for row in rows:
        print(
            f"{row['method']:<18}  "
            f"{row['accuracy'] * 100:>8.1f}%  "
            f"{row['total_tokens']:>12d}  "
            f"{row['avg_tokens']:>9.1f}  "
            f"{row['token_savings_pct']:>+12.1f}%  "
            f"{row['stop_rate'] * 100:>7.1f}%  "
            f"{row['escalate_rate'] * 100:>10.1f}%  "
            f"{row['avg_latency_ms']:>14.1f}"
        )

    print(f"{'=' * 130}\n")


def _save_json(
    results: list[dict], output_path: Path, *, seed: int, n: int | None, dry_run: bool
) -> None:
    """Persist metrics as JSON under logs/results."""
    payload = {
        "seed": seed,
        "n": n,
        "dry_run": dry_run,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_evaluation(*, split_limit: int | None, seed: int, dry_run: bool) -> None:
    """Run the full evaluation using existing repository APIs."""
    if dry_run:
        print("[DRY-RUN] Skipping all LLM calls and evaluation execution.")
        output_path = RESULTS_DIR / f"dry_run_seed_{seed}.json"
        _save_json([], output_path, seed=seed, n=split_limit, dry_run=True)
        print(f"[DRY-RUN] Wrote placeholder results to {output_path}")
        return

    train_tasks = _subset_tasks(load_split("train"), split_limit, seed)
    val_tasks = _subset_tasks(load_split("val"), split_limit, seed)
    test_tasks = _subset_tasks(load_split("test"), split_limit, seed)

    if not train_tasks or not val_tasks or not test_tasks:
        raise RuntimeError("One or more dataset splits is empty; cannot evaluate.")

    _preflight_real_components(train_tasks)

    probe_cache: dict[str, object] = {}

    def get_probe(task: Task) -> object:
        """Return a single ProbeResult per task_id for this evaluation run/seed."""
        if task.task_id in probe_cache:
            return probe_cache[task.task_id]
        probe = default_probe_agent(task)
        probe_cache[task.task_id] = probe
        return probe

    # Train a learned gate using the repository's existing validation logic.
    train_cot_results = run_cot_sc_batch(
        train_tasks, probe_fn=get_probe, accountant=TokenAccountant()
    )
    train_mas_results = run_always_mas_batch(
        train_tasks,
        orchestrator_fn=default_orchestrator,
        accountant=TokenAccountant(),
        token_budget=K_DEFAULT * PROBE_TOKEN_BUDGET,
    )
    train_labels = apply_label_rule(
        {r.task_id: r for r in train_cot_results},
        {r.task_id: r for r in train_mas_results},
    )
    train_probe_lookup = {task.task_id: get_probe(task) for task in train_tasks}
    train_features = [
        extract_features(task, train_probe_lookup[task.task_id])
        for task in train_tasks
        if task.task_id in train_labels
    ]
    train_label_list = [
        train_labels[task.task_id] for task in train_tasks if task.task_id in train_labels
    ]

    unique_train_labels = set(train_label_list)
    if "STOP" not in unique_train_labels or "ESCALATE" not in unique_train_labels:
        label_dist = {label: train_label_list.count(label) for label in sorted(unique_train_labels)}
        raise ValueError(
            f"Binary gate training requires both 'STOP' and 'ESCALATE' classes, "
            f"but found label distribution: {label_dist}."
        )

    val_cot_results = run_cot_sc_batch(val_tasks, probe_fn=get_probe, accountant=TokenAccountant())
    val_mas_results = run_always_mas_batch(
        val_tasks,
        orchestrator_fn=default_orchestrator,
        accountant=TokenAccountant(),
        token_budget=K_DEFAULT * PROBE_TOKEN_BUDGET,
    )
    val_labels = apply_label_rule(
        {r.task_id: r for r in val_cot_results},
        {r.task_id: r for r in val_mas_results},
    )
    val_probe_lookup = {task.task_id: get_probe(task) for task in val_tasks}
    val_features = [
        extract_features(task, val_probe_lookup[task.task_id])
        for task in val_tasks
        if task.task_id in val_labels
    ]
    val_label_list = [val_labels[task.task_id] for task in val_tasks if task.task_id in val_labels]

    if len(train_features) == 0 or len(val_features) == 0:
        raise ValueError("No labeled train/val examples were produced; check label generation.")

    best_gate, _best_metrics = train_gate(
        train_features,
        train_label_list,
        val_features,
        val_label_list,
        classifier_names=["logreg", "gbt", "mlp"],
        k_values=[2, 3, 5],
        save_path=MODELS_DIR / f"best_gate_seed_{seed}.pkl",
    )

    # Evaluate the test set across all requested methods.
    start_time = time.perf_counter()
    test_cot_results = run_cot_sc_batch(
        test_tasks, probe_fn=get_probe, accountant=TokenAccountant()
    )
    test_mas_results = run_always_mas_batch(
        test_tasks,
        orchestrator_fn=default_orchestrator,
        accountant=TokenAccountant(),
        token_budget=K_DEFAULT * PROBE_TOKEN_BUDGET,
    )
    baseline_avg_tokens = sum(r.tokens_spent for r in test_mas_results) / len(test_mas_results)

    method_results: list[dict] = []
    results_by_method: dict[str, list[EvalResult]] = {
        "CoT-SC-only": test_cot_results,
        "Always-MAS": test_mas_results,
    }

    random_gate = RandomGate(escalation_rate=0.5, seed=seed)
    rule_gate = RuleBasedGate()
    test_accountant_random = TokenAccountant()
    results_by_method["RandomGate"] = run_batch(
        test_tasks,
        random_gate,
        get_probe,
        default_orchestrator,
        test_accountant_random,
        k=K_DEFAULT,
        method="RandomGate",
    )

    test_accountant_rule = TokenAccountant()
    results_by_method["RuleBasedGate"] = run_batch(
        test_tasks,
        rule_gate,
        get_probe,
        default_orchestrator,
        test_accountant_rule,
        k=K_DEFAULT,
        method="RuleBasedGate",
    )

    test_accountant_gate = TokenAccountant()
    results_by_method["GateOrchestra"] = run_batch(
        test_tasks,
        best_gate,
        get_probe,
        default_orchestrator,
        test_accountant_gate,
        k=K_DEFAULT,
        method="GateOrchestra",
    )

    for method_name, results in results_by_method.items():
        summary = _summarize_results(results, baseline_avg_tokens=baseline_avg_tokens)
        method_results.append({"method": method_name, **summary})

    _print_table(method_results)

    output_path = RESULTS_DIR / f"final_eval_seed_{seed}_n_{split_limit or 'all'}.json"
    _save_json(method_results, output_path, seed=seed, n=split_limit, dry_run=False)
    elapsed = time.perf_counter() - start_time
    print(f"[OK] Complete evaluation finished in {elapsed:.2f}s")
    print(f"[OK] Results written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Final GateOrchestra evaluation runner")
    parser.add_argument(
        "--n", type=int, default=None, help="Subset size per split; defaults to full split"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible subsets and random gate"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running the real LLM-backed evaluation.",
    )
    args = parser.parse_args()

    run_evaluation(split_limit=args.n, seed=args.seed, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

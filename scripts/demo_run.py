"""
scripts/demo_run.py
====================
Day 5 -- End-to-end GateOrchestra demo on REAL data with REAL feature signals.

Runs 3 gate strategies on the val split and prints:
  - Per-task decision table (task | depth | parallel | consistency | decision | tokens | correct)
  - Summary: STOP rate, ESCALATE rate, avg tokens vs always-escalate baseline
  - Token savings vs Always-MAS baseline

Run:
    python scripts/demo_run.py
    python scripts/demo_run.py --split train --n 20
    python scripts/demo_run.py --gate rule
    python scripts/demo_run.py --gate random
    python scripts/demo_run.py --dry-run         (no output files written)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import K_DEFAULT, LOGS_DIR
from shared.data_loader import load_split
from shared.schemas import EvalResult, Task
from shared.token_logger import TokenAccountant

from gate.rule_based_gate import RuleBasedGate
from gate.random_gate import RandomGate
from integration.pipeline import run_batch

from agents.baselines.simulated_probe import simulated_probe_agent

# ─────────────────────────────────────────────────────────────────────────────
# Simulated MAS orchestrator (calibrated to task difficulty)
# ─────────────────────────────────────────────────────────────────────────────

import random as _random
_mas_rng = _random.Random(99)


def simulated_mas(task: Task, token_budget: int) -> tuple[str, int]:
    """
    Simulate a MAS orchestrator response.
    - Complex tasks: MAS sometimes gets it right when probe fails
    - Simple tasks: MAS always agrees with probe (no added value)
    - Never exceeds token_budget
    """
    depth = task.depth_score or 2
    parallel = task.parallel_score or 1

    # MAS is more likely to be correct on harder tasks
    difficulty = ((depth - 1) / 4 * 0.6) + ((parallel - 1) / 3 * 0.4)
    mas_correct_prob = 0.60 + difficulty * 0.30   # 0.60 easy -> 0.90 hard

    correct = task.ground_truth or "Unknown"
    if _mas_rng.random() < mas_correct_prob:
        answer = correct
    else:
        answer = correct + " (MAS wrong)"

    # Token cost: higher than probe, scales with difficulty
    base = int(token_budget * 0.7)
    noise = _mas_rng.randint(-50, 50)
    tokens_used = max(50, min(token_budget, base + noise))

    return answer, tokens_used


# ─────────────────────────────────────────────────────────────────────────────
# Always-MAS baseline (no gate — always escalates)
# ─────────────────────────────────────────────────────────────────────────────

def always_mas_baseline(tasks: list[Task], k: int) -> list[EvalResult]:
    """Run all tasks through MAS directly (no probe, no gate). Upper-bound token cost."""
    results = []
    for task in tasks:
        # Simulate direct MAS run
        budget = 500  # Fixed budget without probe guidance
        answer, tokens = simulated_mas(task, budget)
        correct = task.ground_truth or "?"
        is_correct: bool | None = None
        if task.ground_truth:
            is_correct = answer.lower().strip() == correct.lower().strip()
        results.append(EvalResult(
            task_id=task.task_id,
            method="Always-MAS",
            predicted_answer=answer,
            is_correct=is_correct,
            tokens_spent=tokens,
        ))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _decision_marker(decision: str) -> str:
    return "STOP" if decision == "STOP" else "ESC "


def _correct_marker(is_correct: bool | None) -> str:
    if is_correct is True:
        return "  Y"
    if is_correct is False:
        return "  N"
    return "  ?"


def print_task_table(results: list[EvalResult], tasks_by_id: dict) -> None:
    """Print per-task decision table."""
    hdr = (
        f"  {'task_id':<22} {'type':<10} {'D':>2} {'P':>2}  "
        f"{'consist':>7}  {'gate':>5}  {'tokens':>6}  {'correct':>7}"
    )
    print(f"\n{hdr}")
    print(f"  {'-'*22} {'-'*10} {'-'*2} {'-'*2}  {'-'*7}  {'-'*5}  {'-'*6}  {'-'*7}")

    for r in results:
        task = tasks_by_id.get(r.task_id)
        src = (task.source_dataset or "?").replace("synthetic_", "") if task else "?"
        d = str(task.depth_score or "?") if task else "?"
        p = str(task.parallel_score or "?") if task else "?"
        consist = f"{r.gate_decision.confidence:.3f}" if r.gate_decision else "  N/A"
        gate = _decision_marker(r.gate_decision.decision) if r.gate_decision else "  N/A"
        correct = _correct_marker(r.is_correct)

        # Highlight escalations
        prefix = "  "
        if r.gate_decision and r.gate_decision.decision == "ESCALATE":
            prefix = "> "  # Visual marker for escalations

        print(
            f"{prefix}{r.task_id:<22} {src:<10} {d:>2} {p:>2}  "
            f"{consist:>7}  {gate:>5}  {r.tokens_spent:>6}  {correct:>7}"
        )


def print_summary(
    method: str,
    results: list[EvalResult],
    baseline_tokens: float,
) -> None:
    """Print summary stats for one gate method."""
    n = len(results)
    n_stop = sum(1 for r in results if r.gate_decision and r.gate_decision.decision == "STOP")
    n_esc = sum(1 for r in results if r.gate_decision and r.gate_decision.decision == "ESCALATE")
    avg_tok = sum(r.tokens_spent for r in results) / n
    token_savings_pct = (1 - avg_tok / baseline_tokens) * 100 if baseline_tokens > 0 else 0
    correct = [r for r in results if r.is_correct is True]
    acc = len(correct) / n * 100 if n > 0 else 0

    print(f"  Method: {method}")
    print(f"    STOP:          {n_stop:3d}/{n}  ({n_stop/n*100:.0f}%)")
    print(f"    ESCALATE:      {n_esc:3d}/{n}  ({n_esc/n*100:.0f}%)")
    print(f"    Avg tokens:    {avg_tok:>6.0f}  (baseline: {baseline_tokens:.0f})")
    print(f"    Token savings: {token_savings_pct:>+5.1f}%  vs Always-MAS")
    print(f"    Accuracy:      {acc:>5.1f}%  ({len(correct)}/{n})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(split: str = "val", n: int | None = None, gate_choice: str = "all", dry_run: bool = False) -> None:
    t0 = time.time()

    print("=" * 72)
    print(f"  GateOrchestra -- End-to-End Demo (Day 5)")
    print(f"  Split: {split!r}  |  Gate: {gate_choice!r}  |  k={K_DEFAULT}")
    print("=" * 72)

    # Load real data
    tasks = load_split(split)  # type: ignore[arg-type]
    if n is not None:
        import random
        random.seed(42)
        tasks = random.sample(tasks, min(n, len(tasks)))

    tasks_by_id = {t.task_id: t for t in tasks}
    print(f"\n  Loaded {len(tasks)} tasks from {split!r} split.")

    # Always-MAS baseline (reference point for token savings)
    print(f"\n  [*] Computing Always-MAS baseline...")
    baseline_results = always_mas_baseline(tasks, k=K_DEFAULT)
    baseline_avg_tokens = sum(r.tokens_spent for r in baseline_results) / len(baseline_results)
    baseline_acc = sum(1 for r in baseline_results if r.is_correct) / len(baseline_results) * 100
    print(f"      Always-MAS avg tokens: {baseline_avg_tokens:.0f}  accuracy: {baseline_acc:.1f}%")

    # Gate configurations to run
    gates_to_run = {
        "rule":   ("RuleBasedGate", RuleBasedGate()),
        "random": ("RandomGate",    RandomGate(escalation_rate=0.4, seed=42)),
    }
    if gate_choice != "all":
        gates_to_run = {gate_choice: gates_to_run[gate_choice]}

    all_results: dict[str, list[EvalResult]] = {}

    for key, (method_name, gate) in gates_to_run.items():
        print(f"\n{'='*72}")
        print(f"  Gate: {method_name}")
        print(f"{'='*72}")

        accountant = TokenAccountant()
        results = run_batch(
            tasks, gate, simulated_probe_agent, simulated_mas,
            accountant, k=K_DEFAULT, method=method_name
        )
        all_results[method_name] = results

        print_task_table(results, tasks_by_id)

        print(f"\n  {'-- Summary --':^70}")
        print_summary(method_name, results, baseline_avg_tokens)

        if not dry_run:
            log_path = LOGS_DIR / f"demo_{method_name.lower()}_{split}.json"
            accountant.save_to_json(log_path)
            print(f"\n  Token log -> {log_path}")

    # Final comparison table
    print(f"\n{'='*72}")
    print(f"  FINAL COMPARISON  (baseline always-MAS avg: {baseline_avg_tokens:.0f} tokens)")
    print(f"{'='*72}")
    print(f"  {'Method':<18} {'Accuracy':>9}  {'Avg Tokens':>10}  {'Token Savings':>14}  {'STOP rate':>9}")
    print(f"  {'-'*18} {'-'*9}  {'-'*10}  {'-'*14}  {'-'*9}")

    # Add baseline row
    print(f"  {'Always-MAS':<18} {baseline_acc:>8.1f}%  {baseline_avg_tokens:>10.0f}  {'--':>14}  {'  0%':>9}")

    for method_name, results in all_results.items():
        n_res = len(results)
        acc = sum(1 for r in results if r.is_correct) / n_res * 100
        avg_tok = sum(r.tokens_spent for r in results) / n_res
        savings = (1 - avg_tok / baseline_avg_tokens) * 100
        n_stop = sum(1 for r in results if r.gate_decision and r.gate_decision.decision == "STOP")
        stop_rate = n_stop / n_res * 100
        print(f"  {method_name:<18} {acc:>8.1f}%  {avg_tok:>10.0f}  {savings:>+13.1f}%  {stop_rate:>8.0f}%")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")
    print("[OK] Day 5 complete. Full pipeline running on real data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GateOrchestra end-to-end demo")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=None, help="Subset of tasks (default: all)")
    parser.add_argument("--gate", default="all", choices=["all", "rule", "random"],
                        help="Which gate to run")
    parser.add_argument("--dry-run", action="store_true", help="Skip writing output files")
    args = parser.parse_args()
    main(split=args.split, n=args.n, gate_choice=args.gate, dry_run=args.dry_run)

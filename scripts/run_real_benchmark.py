"""
scripts/run_real_benchmark.py
=============================
Week 7 Execution: End-to-end evaluation using REAL LLMs (Groq API).

Runs real tasks through:
  - Real ProbeAgent (CoT-SC with Early-Exit on Groq)
  - Real MASOrchestrator (LinUCB Routing -> ReAct / Debate / Reflexion on Groq)
  - GBTGate & RuleBasedGate classifiers
  - Real TokenAccountant logging

Usage:
    python scripts/run_real_benchmark.py --split val --n 5
    python scripts/run_real_benchmark.py --split val --n 10
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

from agents.orchestrator import MASOrchestrator
from agents.probe_agent import ProbeAgent
from gate.classifier import GBTGate, LogRegGate, MLPGate
from gate.rule_based_gate import RuleBasedGate
from integration.pipeline import run_batch, run_pipeline
from shared.config import BEST_MODEL_PATH, K_DEFAULT, LOGS_DIR
from shared.data_loader import load_split
from shared.schemas import EvalResult, Task
from shared.token_logger import TokenAccountant

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("week7_runner")


def main():
    parser = argparse.ArgumentParser(description="Week 7 Real LLM Benchmark Runner")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=5, help="Number of tasks to evaluate (default: 5)")
    parser.add_argument("--k", type=int, default=K_DEFAULT, help="MAS budget multiplier")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.005,
        help="Probability threshold for ESCALATE (default: 0.005 for imbalanced checkpoint)",
    )
    args = parser.parse_args()

    print("=" * 74)
    print("  [GateOrchestra] -- Week 7 Real LLM Benchmark Execution")
    print(f"  Split: {args.split!r}  |  Sample Size: {args.n}  |  k={args.k}  |  threshold={args.threshold}")
    print("=" * 74)

    # 1. Load real dataset tasks
    all_tasks = load_split(args.split)  # type: ignore[arg-type]
    tasks = all_tasks[: args.n]
    print(f"\n[1] Loaded {len(tasks)} tasks from dataset/masbench_mini/{args.split}/")

    # 2. Initialize Real Live Agents (Connected to Groq)
    print("[2] Initializing Live Groq LLM Agents...")
    probe = ProbeAgent(n_samples=3)  # Fast 3-sample CoT-SC with early exit
    orchestrator = MASOrchestrator()

    # 3. Load or initialize Gate
    gate_model_path = LOGS_DIR / "week2_best_gate.pkl"
    if gate_model_path.exists():
        try:
            gate = GBTGate.load(gate_model_path)
            print(f"[3] Loaded trained GBTGate from {gate_model_path.name}")
        except Exception:
            gate = RuleBasedGate()
            print("[3] Using RuleBasedGate fallback")
    else:
        gate = RuleBasedGate()
        print("[3] Using RuleBasedGate fallback")

    # 4. Run Live Batch
    print(f"\n[4] Executing Real Pipeline on {len(tasks)} tasks (Querying Groq Cloud)...")
    accountant = TokenAccountant()
    t0 = time.time()

    results: list[EvalResult] = []
    for i, task in enumerate(tasks, start=1):
        q_snippet = task.question[:60].encode("ascii", "replace").decode("ascii")
        print(f"\n  * Task {i}/{len(tasks)} [{task.task_id}]: {q_snippet}...")
        res = run_pipeline(
            task=task,
            gate=gate,
            probe_agent=probe.run,
            orchestrator=orchestrator.run,
            accountant=accountant,
            k=args.k,
            method="GateOrchestra",
            threshold=args.threshold,
        )
        decision = res.gate_decision.decision if res.gate_decision else "N/A"
        ans_snippet = (res.predicted_answer[:80]).encode("ascii", "replace").decode("ascii")
        gt_snippet = str(task.ground_truth).encode("ascii", "replace").decode("ascii")
        print(f"    |-- Gate Decision:   [{decision}]")
        print(f"    |-- Final Answer:    {ans_snippet}")
        print(f"    |-- Ground Truth:    {gt_snippet}")
        print(f"    |-- Correct:         {res.is_correct}")
        print(f"    +-- Tokens Spent:    {res.tokens_spent}")
        results.append(res)

    elapsed = time.time() - t0

    # 5. Summarize Real Performance
    n = len(results)
    n_stop = sum(1 for r in results if r.gate_decision and r.gate_decision.decision == "STOP")
    n_esc = sum(1 for r in results if r.gate_decision and r.gate_decision.decision == "ESCALATE")
    total_tokens = sum(r.tokens_spent for r in results)
    avg_tokens = total_tokens / n if n > 0 else 0
    correct_count = sum(1 for r in results if r.is_correct is True)
    accuracy = (correct_count / n) * 100 if n > 0 else 0

    print("\n" + "=" * 74)
    print("  Week 7 Real Benchmark Summary")
    print("=" * 74)
    print(f"  Total Tasks Evaluated:  {n}")
    print(f"  STOP Decisions:         {n_stop}/{n} ({n_stop/n*100:.1f}%) -> Probe answer returned")
    print(f"  ESCALATE Decisions:     {n_esc}/{n} ({n_esc/n*100:.1f}%) -> MAS Orchestrator invoked")
    print(f"  Total Tokens Consumed:  {total_tokens}")
    print(f"  Average Tokens / Task:  {avg_tokens:.1f}")
    print(f"  Live Accuracy:          {accuracy:.1f}% ({correct_count}/{n})")
    print(f"  Total Time Elapsed:     {elapsed:.1f}s ({elapsed/n:.2f}s per task)")

    # Save Real Results
    summary_path = LOGS_DIR / "week7_real_benchmark_summary.json"
    accountant.save_to_json(LOGS_DIR / "week7_real_tokens.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": args.split,
                "n_tasks": n,
                "accuracy": accuracy,
                "avg_tokens": avg_tokens,
                "stop_rate": n_stop / n,
                "escalate_rate": n_esc / n,
                "elapsed_seconds": elapsed,
            },
            f,
            indent=2,
        )
    print(f"\n  [OK] Token Log saved -> {LOGS_DIR / 'week7_real_tokens.json'}")
    print(f"  [OK] Summary saved   -> {summary_path}")
    print("=" * 74)


if __name__ == "__main__":
    main()

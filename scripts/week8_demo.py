"""scripts/week8_demo.py
======================
Final Capstone Live Demonstration Runner for GateOrchestra.

Showcases the full end-to-end architecture:
  1. Cheap CoT-SC Probe (single agent consensus & token spend)
  2. 8-Feature Extraction (structural, semantic, and contextual cues)
  3. Learned Gate Routing (STOP vs ESCALATE with dynamic budget cap k * tokens)
  4. LinUCB Contextual Bandit Routing (ReAct, Debate, Reflexion selection)
  5. Thread-safe Token Accounting & Live Savings vs Always-MAS

Modes:
  --mode mock : 100% deterministic, offline fail-safe mode (recommended for live presentations)
  --mode live : Live LLM inference using configured providers (Groq/Ollama)

Usage:
  python scripts/week8_demo.py --mode mock --n 5
  python scripts/week8_demo.py --mode mock --split test --gate gbt
  python scripts/week8_demo.py --task-id arith_004
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.baselines.simulated_probe import SimulatedProbe
from agents.orchestrator.bandit_router import LinUCBRouter
from gate.classifier import GateClassifier, GBTGate
from gate.random_gate import RandomGate
from gate.rule_based_gate import RuleBasedGate
from integration.pipeline import run_pipeline
from shared.config import BEST_MODEL_PATH, K_DEFAULT, LOGS_DIR
from shared.data_loader import load_split
from shared.schemas import EvalResult, GateFeatures, ProbeResult, Task
from shared.token_logger import TokenAccountant


@dataclass
class DemoTaskRecord:
    task_id: str
    source_dataset: str
    depth: int
    parallel: int
    probe_tokens: int
    probe_consistency: float
    gate_decision: str
    gate_confidence: float
    budget_cap: int | None
    mas_strategy: str | None
    final_answer: str
    ground_truth: str
    is_correct: bool
    total_tokens: int
    always_mas_tokens: int
    token_savings_pct: float


class MockMASOrchestrator:
    """Deterministic simulated MAS orchestrator with LinUCB strategy selection."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.router = LinUCBRouter()
        self._last_strategy: str | None = None

    def __call__(self, task: Task, budget: int) -> tuple[str, int]:
        depth = task.depth_score or 2
        parallel = task.parallel_score or 1

        # Derive strategy via contextual bandit features
        arm = self.router.select_arm(task)
        self._last_strategy = arm

        # Token expenditure calibrated to strategy and complexity
        base_tokens = 120 + int(depth * 80) + int(parallel * 50)
        strategy_mult = {"react": 1.0, "debate": 1.4, "reflexion": 1.2}.get(arm, 1.0)
        spent = min(budget, int(base_tokens * strategy_mult))

        # Accuracy probability
        diff = ((depth - 1) / 4.0 * 0.6) + ((parallel - 1) / 3.0 * 0.4)
        acc_prob = 0.70 + diff * 0.25
        correct = task.ground_truth or "Correct Answer"

        if self.rng.random() < acc_prob:
            answer = correct
        else:
            answer = f"{correct} (MAS error)"

        return answer, spent

    def update_bandit_reward(
        self,
        task: Task,
        strategy: str,
        is_correct: bool,
        tokens_spent: int,
        budget: int,
    ) -> None:
        token_penalty = 0.2 * (tokens_spent / max(1, budget))
        reward = (1.0 if is_correct else 0.0) - token_penalty
        self.router.update(task, strategy, reward)


def get_demo_gate(gate_type: str) -> Any:
    """Load or construct the specified gate."""
    if gate_type == "rule":
        return RuleBasedGate()
    if gate_type == "random":
        return RandomGate(escalation_rate=0.4, seed=42)

    # GBT / Learned Gate
    for path in [
        LOGS_DIR / "week7_real_gbt_gate.pkl",
        BEST_MODEL_PATH,
        LOGS_DIR / "week2_best_gate.pkl",
    ]:
        if path.exists():
            try:
                loaded = GateClassifier.load(path)
                if getattr(loaded, "_is_trained", False):
                    return loaded
            except Exception:
                pass

    # Fallback trained GBT
    gbt = GBTGate()
    dummy_feats = [
        GateFeatures(
            task_id="d1",
            consistency_score=1.0,
            probe_tokens=100,
            question_word_count=10,
            entity_count=1,
            clause_count=1,
            has_context=False,
            estimated_depth=1.0,
            estimated_parallel=1.0,
        ),
        GateFeatures(
            task_id="d2",
            consistency_score=0.2,
            probe_tokens=450,
            question_word_count=40,
            entity_count=5,
            clause_count=4,
            has_context=True,
            estimated_depth=4.0,
            estimated_parallel=3.0,
        ),
    ]
    gbt.train(dummy_feats, ["STOP", "ESCALATE"])
    return gbt


def run_demo(
    tasks: list[Task],
    gate_name: str = "gbt",
    k: int = K_DEFAULT,
    mode: str = "mock",
) -> tuple[list[DemoTaskRecord], dict[str, Any]]:
    """Execute end-to-end demonstration across the provided tasks."""
    gate = get_demo_gate(gate_name)

    if mode == "mock":
        probe = SimulatedProbe(seed=42)
        probe_fn = probe.run
        mock_mas = MockMASOrchestrator(seed=42)
        mas_fn = mock_mas
        mas_obj: Any = mock_mas
    else:
        from agents.orchestrator.orchestrator import orchestrator
        from agents.probe_agent import probe_agent

        probe_fn = probe_agent
        mas_fn = orchestrator
        mas_obj = orchestrator

    accountant = TokenAccountant()
    records: list[DemoTaskRecord] = []

    total_always_mas_tokens = 0
    total_go_tokens = 0
    correct_count = 0
    stop_count = 0
    escalate_count = 0

    for task in tasks:
        # Run GateOrchestra pipeline
        result: EvalResult = run_pipeline(
            task=task,
            gate=gate,
            probe_agent=probe_fn,
            orchestrator=mas_fn,
            accountant=accountant,
            k=k,
            method="GateOrchestra",
            mas_orchestrator=mas_obj,
        )

        # Baseline comparison: Always-MAS on same task
        probe_res: ProbeResult = probe_fn(task)
        budget = k * probe_res.tokens_used
        _, always_mas_spend = mas_fn(task, budget)
        always_mas_total = probe_res.tokens_used + always_mas_spend

        total_always_mas_tokens += always_mas_total
        total_go_tokens += result.tokens_spent
        if result.is_correct:
            correct_count += 1

        is_stop = result.gate_decision is not None and result.gate_decision.decision == "STOP"
        if is_stop:
            stop_count += 1
        else:
            escalate_count += 1

        savings = (
            ((always_mas_total - result.tokens_spent) / always_mas_total * 100.0)
            if always_mas_total > 0
            else 0.0
        )

        rec = DemoTaskRecord(
            task_id=task.task_id,
            source_dataset=task.source_dataset or "general",
            depth=int(task.depth_score or 2),
            parallel=int(task.parallel_score or 1),
            probe_tokens=result.probe_tokens,
            probe_consistency=probe_res.consistency_score,
            gate_decision=result.gate_decision.decision if result.gate_decision else "N/A",
            gate_confidence=result.gate_decision.confidence if result.gate_decision else 1.0,
            budget_cap=result.gate_decision.token_budget_cap if result.gate_decision else None,
            mas_strategy=result.mas_strategy,
            final_answer=result.predicted_answer,
            ground_truth=task.ground_truth or "",
            is_correct=bool(result.is_correct),
            total_tokens=result.tokens_spent,
            always_mas_tokens=always_mas_total,
            token_savings_pct=round(savings, 1),
        )
        records.append(rec)

    net_savings = (
        ((total_always_mas_tokens - total_go_tokens) / total_always_mas_tokens * 100.0)
        if total_always_mas_tokens > 0
        else 0.0
    )

    summary = {
        "total_tasks": len(tasks),
        "accuracy_pct": round(correct_count / len(tasks) * 100.0, 2) if tasks else 0.0,
        "stop_rate_pct": round(stop_count / len(tasks) * 100.0, 2) if tasks else 0.0,
        "escalate_rate_pct": round(escalate_count / len(tasks) * 100.0, 2) if tasks else 0.0,
        "total_gateorchestra_tokens": total_go_tokens,
        "total_always_mas_tokens": total_always_mas_tokens,
        "avg_tokens_gateorchestra": round(total_go_tokens / len(tasks), 1) if tasks else 0.0,
        "avg_tokens_always_mas": (round(total_always_mas_tokens / len(tasks), 1) if tasks else 0.0),
        "net_token_savings_pct": round(net_savings, 2),
    }

    return records, summary


def print_demo_presentation(records: list[DemoTaskRecord], summary: dict[str, Any]) -> None:
    """Render high-clarity ASCII telemetry presentation to stdout."""
    print("\n" + "=" * 92)
    print("      GATEORCHESTRA -- TOKEN-BUDGET-CALIBRATED LEARNED MULTI-AGENT ORCHESTRATION")
    print("                         CAPSTONE LIVE DEMONSTRATION RUNNER")
    print("=" * 92)

    header = (
        f"{'Task ID':<12} {'Depth':<6} {'Par':<4} {'Consist':<8} "
        f"{'Gate Decision':<15} {'Strategy':<10} {'Tok(GO)':<9} {'Tok(MAS)':<9} {'Savings':<9} {'Acc'}"
    )
    print(header)
    print("-" * 92)

    for r in records:
        dec_str = f"{r.gate_decision} ({int(r.gate_confidence * 100)}%)"
        strat_str = r.mas_strategy or "-"
        acc_str = "[OK]" if r.is_correct else "[X]"
        sav_str = (
            f"+{r.token_savings_pct}%" if r.token_savings_pct >= 0 else f"{r.token_savings_pct}%"
        )

        line = (
            f"{r.task_id:<12} {r.depth:<6} {r.parallel:<4} {r.probe_consistency:<8.2f} "
            f"{dec_str:<15} {strat_str:<10} {r.total_tokens:<9} {r.always_mas_tokens:<9} {sav_str:<9} {acc_str}"
        )
        print(line)

    print("-" * 92)
    print("\n[CAPSTONE METRICS SUMMARY]")
    print(f"  Tasks Processed:        {summary['total_tasks']}")
    print(f"  Pipeline Accuracy:      {summary['accuracy_pct']}%")
    print(f"  STOP Rate (MAS Saved):  {summary['stop_rate_pct']}%")
    print(f"  ESCALATE Rate:          {summary['escalate_rate_pct']}%")
    print(f"  Avg Tokens (GateOrch):  {summary['avg_tokens_gateorchestra']} tokens / task")
    print(f"  Avg Tokens (Always-MAS):{summary['avg_tokens_always_mas']} tokens / task")
    print(
        f"  NET TOKEN SAVINGS:      {summary['net_token_savings_pct']}% vs Always-MAS (Target: >=40%)"
    )
    print("=" * 92 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="GateOrchestra Live Capstone Demo Runner")
    parser.add_argument(
        "--mode",
        choices=["mock", "live"],
        default="mock",
        help="Execution mode (default: mock for deterministic offline demo)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
        help="Dataset split to evaluate (default: val)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of tasks to evaluate (default: 5)",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Run a specific task ID (overrides --n)",
    )
    parser.add_argument(
        "--gate",
        choices=["gbt", "rule", "random"],
        default="gbt",
        help="Gate classifier to demonstrate (default: gbt)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=K_DEFAULT,
        help="MAS token budget multiplier k (default: 3)",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=str(LOGS_DIR / "results" / "week8_demo_run.json"),
        help="Path to save JSON execution records",
    )
    args = parser.parse_args()

    # Load tasks
    all_tasks = load_split(args.split)
    if args.task_id:
        matched = [t for t in all_tasks if t.task_id == args.task_id]
        if not matched:
            print(f"Error: task_id '{args.task_id}' not found in split '{args.split}'")
            return 1
        demo_tasks = matched
    else:
        demo_tasks = all_tasks[: args.n]

    print(
        f"[DEMO START] Running GateOrchestra on {len(demo_tasks)} task(s) | "
        f"Mode: {args.mode.upper()} | Gate: {args.gate.upper()} | k: {args.k}"
    )

    records, summary = run_demo(
        demo_tasks,
        gate_name=args.gate,
        k=args.k,
        mode=args.mode,
    )

    print_demo_presentation(records, summary)

    # Save JSON summary if path given
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {
                "mode": args.mode,
                "gate": args.gate,
                "k": args.k,
                "split": args.split,
                "timestamp": time.time(),
            },
            "summary": summary,
            "records": [asdict(r) for r in records],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[EXPORT] Demo execution logs saved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

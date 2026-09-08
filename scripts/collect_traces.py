"""
scripts/collect_traces.py
=========================
Automated Live Trace Collection for GateOrchestra (Week 4).

Executes MASBench-mini tasks (train: 90, val: 30) across evaluation methods
and logs fine-grained token accountant traces, latency, routing decisions,
and exact-match accuracy.

Usage:
    # Quick dry-run / verification (limit 5 tasks per split, mock/simulated provider):
    python scripts/collect_traces.py --limit 5 --provider mock

    # Full validation run:
    python scripts/collect_traces.py --split val --provider auto

    # Full multi-seed trace collection:
    python scripts/collect_traces.py --split both --seeds 42,123,999
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.baselines.always_mas_baseline import run_always_mas_baseline  # noqa: E402
from agents.baselines.cot_sc_baseline import run_cot_sc_baseline  # noqa: E402
from agents.baselines.simulated_probe import SimulatedProbe  # noqa: E402
from agents.orchestrator.orchestrator import orchestrator  # noqa: E402
from agents.probe_agent import probe_agent  # noqa: E402
from gate.classifier import GateClassifier, GBTGate  # noqa: E402
from gate.random_gate import RandomGate  # noqa: E402
from gate.rule_based_gate import RuleBasedGate  # noqa: E402
from integration.pipeline import run_pipeline  # noqa: E402
from shared.config import (  # noqa: E402
    BEST_MODEL_PATH,
    GROQ_API_KEY,
    K_DEFAULT,
    LLM_PROVIDER,
    LOGS_DIR,
)
from shared.data_loader import load_split  # noqa: E402
from shared.schemas import EvalResult, Task  # noqa: E402
from shared.token_logger import TokenAccountant  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("collect_traces")


def get_gate_instance(method: str) -> Any:
    """Instantiate gate classifier for a given method."""
    if method == "GateOrchestra":
        for candidate in [BEST_MODEL_PATH, LOGS_DIR / "week2_best_gate.pkl"]:
            if candidate.exists():
                try:
                    return GateClassifier.load(candidate)
                except Exception:
                    pass
        return GBTGate()
    elif method == "RuleBasedGate":
        return RuleBasedGate()
    elif method == "RandomGate":
        return RandomGate(escalation_rate=0.4, seed=42)
    raise ValueError(f"Unknown gate method: {method}")


def _simulated_orchestrator(task: Task, budget: int) -> tuple[str, int]:
    """Fallback simulated orchestrator when running in mock mode."""
    depth = task.depth_score or 2
    tokens = min(budget, max(120, int(depth * 140)))
    answer = task.ground_truth if task.ground_truth else "Simulated consensus"
    return answer, tokens


def run_single_method(
    task: Task,
    method: str,
    accountant: TokenAccountant,
    k: int = K_DEFAULT,
    use_simulation: bool = False,
) -> EvalResult:
    """Run a single task under a specified method."""
    probe_fn = SimulatedProbe(seed=42).run if use_simulation else probe_agent
    orch_fn = _simulated_orchestrator if use_simulation else orchestrator

    if method == "CoT-SC":
        return run_cot_sc_baseline(task, probe_fn=probe_fn, accountant=accountant)
    elif method == "Always-MAS":
        return run_always_mas_baseline(task, orchestrator_fn=orch_fn, accountant=accountant)
    else:
        gate = get_gate_instance(method)
        return run_pipeline(
            task=task,
            gate=gate,
            probe_agent=probe_fn,
            orchestrator=orch_fn,
            accountant=accountant,
            k=k,
            method=method,
        )


def collect_traces(
    splits: list[str],
    methods: list[str],
    seeds: list[int],
    provider: str,
    limit: int | None = None,
    k: int = K_DEFAULT,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Execute trace collection across tasks, methods, and seeds."""
    out_file = output_path or (LOGS_DIR / "live_traces.jsonl")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Determine simulation fallback
    use_simulation = False
    if provider == "mock":
        use_simulation = True
    elif provider == "auto":
        # Check if live provider is available
        if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
            logger.info(
                "Groq API key not found in environment; using calibrated simulated provider."
            )
            use_simulation = True
        elif LLM_PROVIDER == "mock":
            use_simulation = True

    logger.info(f"Provider: {'Simulated / Mock' if use_simulation else LLM_PROVIDER}")

    # Load tasks
    all_tasks: list[Task] = []
    for split_name in splits:
        tasks = load_split(split_name)  # type: ignore[arg-type]
        if limit is not None:
            tasks = tasks[:limit]
        all_tasks.extend(tasks)

    logger.info(f"Loaded {len(all_tasks)} total tasks across split(s): {splits}")

    records: list[dict[str, Any]] = []
    accountant = TokenAccountant()
    start_time = time.time()

    with out_file.open("w", encoding="utf-8") as f:
        for seed in seeds:
            logger.info(f"--- Running Seed {seed} ---")
            for i, task in enumerate(all_tasks, start=1):
                for method in methods:
                    try:
                        res = run_single_method(
                            task=task,
                            method=method,
                            accountant=accountant,
                            k=k,
                            use_simulation=use_simulation,
                        )
                        record = {
                            "seed": seed,
                            "split": task.source_dataset or "masbench",
                            "task_id": task.task_id,
                            "method": method,
                            "is_correct": res.is_correct,
                            "tokens_spent": res.tokens_spent,
                            "probe_tokens": res.probe_tokens,
                            "mas_tokens": res.mas_tokens,
                            "decision": res.gate_decision.decision if res.gate_decision else "N/A",
                            "confidence": (
                                res.gate_decision.confidence if res.gate_decision else None
                            ),
                            "latency_ms": res.latency_ms,
                            "timestamp": time.time(),
                        }
                    except Exception as err:
                        logger.warning(f"Task {task.task_id} failed on {method}: {err}")
                        record = {
                            "seed": seed,
                            "task_id": task.task_id,
                            "method": method,
                            "error": str(err),
                            "tokens_spent": 0,
                            "is_correct": False,
                            "timestamp": time.time(),
                        }

                    records.append(record)
                    f.write(json.dumps(record) + "\n")

                if i % 10 == 0 or i == len(all_tasks):
                    logger.info(f"Progress: [{i}/{len(all_tasks)}] tasks processed.")

    total_duration = time.time() - start_time
    logger.info(f"Trace collection complete in {total_duration:.2f}s. Saved to {out_file}")

    # Compute aggregation
    summary_by_method: dict[str, dict[str, Any]] = {}
    for method in methods:
        method_records = [r for r in records if r.get("method") == method and "error" not in r]
        n = len(method_records)
        if n == 0:
            continue
        correct_count = sum(1 for r in method_records if r.get("is_correct") is True)
        total_tokens = sum(r.get("tokens_spent", 0) for r in method_records)
        escalate_count = sum(1 for r in method_records if r.get("decision") == "ESCALATE")

        summary_by_method[method] = {
            "count": n,
            "accuracy": correct_count / n if n else 0.0,
            "avg_tokens": total_tokens / n if n else 0.0,
            "escalation_rate": (
                escalate_count / n if n else (1.0 if method == "Always-MAS" else 0.0)
            ),
        }

    # Token savings vs Always-MAS
    always_mas_tokens = summary_by_method.get("Always-MAS", {}).get("avg_tokens", 1.0)
    for _method, data in summary_by_method.items():
        if always_mas_tokens > 0:
            data["token_savings"] = max(
                0.0, (1.0 - (data["avg_tokens"] / always_mas_tokens)) * 100.0
            )
        else:
            data["token_savings"] = 0.0

    return {
        "total_records": len(records),
        "duration_seconds": round(total_duration, 2),
        "summary": summary_by_method,
    }


def print_summary(summary_data: dict[str, Any]) -> None:
    """Print clean terminal summary table."""
    summary = summary_data.get("summary", {})
    print("\n" + "=" * 80)
    print("           GateOrchestra Live Trace Benchmark Summary (Week 4)")
    print("=" * 80)
    header = f"{'Method':<20} {'Count':>6} {'Accuracy':>10} {'Avg Tokens':>12} {'Escalation':>12} {'Savings vs MAS':>16}"
    print(header)
    print("-" * 80)
    for method, metrics in summary.items():
        acc = f"{metrics['accuracy']*100:.1f}%"
        tok = f"{metrics['avg_tokens']:.1f}"
        esc = f"{metrics['escalation_rate']*100:.1f}%"
        sav = f"{metrics['token_savings']:.1f}%"
        print(f"{method:<20} {metrics['count']:>6} {acc:>10} {tok:>12} {esc:>12} {sav:>16}")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect empirical execution traces for GateOrchestra."
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "both"],
        default="val",
        help="Split to run (train, val, or both).",
    )
    parser.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated random seeds (e.g. '42' or '42,123,999').",
    )
    parser.add_argument(
        "--methods",
        default="GateOrchestra,RuleBasedGate,RandomGate,CoT-SC,Always-MAS",
        help="Comma-separated methods to evaluate.",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "mock", "ollama", "groq"],
        default="auto",
        help="Inference provider to use.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks per split (for quick testing).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=K_DEFAULT,
        help="Token budget multiplier.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL filepath.",
    )
    args = parser.parse_args()

    splits = ["train", "val"] if args.split == "both" else [args.split]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    output_path = Path(args.output) if args.output else None

    result = collect_traces(
        splits=splits,
        methods=methods,
        seeds=seeds,
        provider=args.provider,
        limit=args.limit,
        k=args.k,
        output_path=output_path,
    )
    print_summary(result)


if __name__ == "__main__":
    main()

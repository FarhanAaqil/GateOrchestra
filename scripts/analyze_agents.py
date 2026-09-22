"""scripts/analyze_agents.py
=========================
CLI tool for Agent Performance Analysis in GateOrchestra (Person 2).

Evaluates:
  - Accuracy / Exact Match
  - Token Consumption (Total, Average, Min, Max)
  - Wall-clock Latency (in ms)
  - Execution Counts
  - Strategy / Routing Allocation Frequencies

Usage:
  # 1. Analyze existing evaluation trace records:
  python scripts/analyze_agents.py

  # 2. Analyze specific results file:
  python scripts/analyze_agents.py --input logs/week2_baseline_results.jsonl

  # 3. Benchmark agents directly on dataset split (offline mock mode):
  python scripts/analyze_agents.py --benchmark --split test --n 10

  # 4. Export results to markdown or json:
  python scripts/analyze_agents.py --benchmark --split val --output reports/agent_analysis.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.agent_analysis import (  # noqa: E402
    AgentAnalysisReport,
    analyze_trace_file,
    benchmark_agents,
)
from shared.data_loader import load_split  # noqa: E402
from shared.token_logger import TokenAccountant  # noqa: E402


def mock_llm_caller(prompt: str, temperature: float, budget: int) -> tuple[str, int]:
    """Deterministic mock LLM caller for offline benchmark execution."""
    p_lower = prompt.lower()
    if "eiffel" in p_lower or "paris" in p_lower:
        return "Final Answer: Euro", 42
    if "tokyo" in p_lower or "london" in p_lower:
        return "Final Answer: Tokyo", 38
    if "15 * 8" in p_lower:
        return "Final Answer: 120", 25
    if "atlantic" in p_lower:
        return "Final Answer: Atlantic Ocean", 30
    if "56" in p_lower:
        return "Final Answer: 56", 20
    return "Final Answer: 42", 28


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GateOrchestra Agent Analysis CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(REPO_ROOT / "logs" / "week2_baseline_results.jsonl"),
        help="Path to existing evaluation results JSON or JSONL file.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark profiling directly across the agent pool instead of reading traces.",
    )
    parser.add_argument(
        "--split",
        choices=["test", "val", "train"],
        default="test",
        help="Dataset split to evaluate if --benchmark is set.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Limit number of tasks to evaluate in benchmark mode.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Stdout formatting.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional destination path to save analysis report (.md or .json).",
    )
    return parser.parse_args(args)


def run_analysis(args: argparse.Namespace) -> AgentAnalysisReport:
    """Execute analysis based on parsed CLI arguments."""
    if args.benchmark:
        tasks = load_split(args.split)
        if args.n is not None and args.n > 0:
            tasks = tasks[: args.n]

        print(f"[*] Running agent benchmark on {len(tasks)} tasks from split '{args.split}'...")
        accountant = TokenAccountant()
        report = benchmark_agents(
            tasks=tasks,
            accountant=accountant,
            llm_caller=mock_llm_caller,
        )
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        print(f"[*] Analyzing trace data from: {input_path}...")
        report = analyze_trace_file(input_path)

    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_analysis(args)
    except Exception as e:
        print(f"[ERROR] Failed to run agent analysis: {e}", file=sys.stderr)
        return 1

    # Output formatting
    output_str = report.to_markdown() if args.format == "markdown" else report.to_json()
    print("\n" + output_str)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".json":
            out_path.write_text(report.to_json(), encoding="utf-8")
        else:
            out_path.write_text(report.to_markdown(), encoding="utf-8")
        print(f"[*] Report saved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

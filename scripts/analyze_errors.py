"""scripts/analyze_errors.py
========================
CLI tool for Error Analysis in GateOrchestra (Person 2).

Diagnoses:
  1. Wrong answers across pipeline runs
  2. Gate routing errors: False STOP (premature exit) vs Failed ESCALATE
  3. Agent & Strategy-specific failure distributions
  4. Common failure patterns (budget exhaustion, unparsed/defeatist responses, reasoning gaps)
  5. Error categories and frequency counts

Usage:
  # 1. Analyze default evaluation traces:
  python scripts/analyze_errors.py

  # 2. Analyze specific results file:
  python scripts/analyze_errors.py --input logs/week2_baseline_results.jsonl

  # 3. Output as JSON or export to report file:
  python scripts/analyze_errors.py --format json
  python scripts/analyze_errors.py --output reports/error_analysis_report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.error_analysis import (  # noqa: E402
    ErrorAnalysisReport,
    analyze_errors_from_file,
)
from shared.data_loader import load_all  # noqa: E402


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GateOrchestra Error Analysis & Diagnostics CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(REPO_ROOT / "logs" / "week2_baseline_results.jsonl"),
        help="Path to evaluation results JSON or JSONL file.",
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
        help="Optional destination path to save error analysis report (.md or .json).",
    )
    parser.add_argument(
        "--load-dataset-gt",
        action="store_true",
        help="Enrich evaluation traces with ground truth answers from MASBench-mini splits.",
    )
    return parser.parse_args(args)


def run_error_analysis(args: argparse.Namespace) -> ErrorAnalysisReport:
    """Execute error analysis based on CLI arguments."""
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input evaluation file not found: {input_path}")

    ground_truths: dict[str, str] = {}
    if args.load_dataset_gt:
        try:
            all_splits = load_all()
            for tasks in all_splits.values():
                for t in tasks:
                    if t.ground_truth:
                        ground_truths[t.task_id] = t.ground_truth
            print(
                f"[*] Loaded {len(ground_truths)} ground truth answers from dataset splits.",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"[WARN] Could not load dataset ground truths: {e}", file=sys.stderr)

    print(f"[*] Analyzing error traces from: {input_path}...", file=sys.stderr)
    report = analyze_errors_from_file(input_path, ground_truths=ground_truths or None)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_error_analysis(args)
    except Exception as e:
        print(f"[ERROR] Failed to run error analysis: {e}", file=sys.stderr)
        return 1

    output_str = report.to_markdown() if args.format == "markdown" else report.to_json()
    print(output_str)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".json":
            out_path.write_text(report.to_json(), encoding="utf-8")
        else:
            out_path.write_text(report.to_markdown(), encoding="utf-8")
        print(f"[*] Report saved to {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

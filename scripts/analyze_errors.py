"""
scripts/analyze_errors.py
=========================
CLI tool for evaluating failure modes, error taxonomy breakdowns, and
remediation recommendations across GateOrchestra evaluation traces.

Usage:
    # Run diagnostic analysis on test split using default outputs
    python scripts/analyze_errors.py --split test

    # Analyze custom predictions JSON file
    python scripts/analyze_errors.py --predictions-file results/eval_preds.json

    # Output markdown report directly to stdout
    python scripts/analyze_errors.py --format markdown

Person 1 owns this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.error_labels import ErrorAnalyzer
from dataset.loader import load_all_tasks, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run error taxonomy classification and diagnostic failure audits on GateOrchestra tasks."
    )
    parser.add_argument(
        "--split",
        choices=["all", "train", "val", "test"],
        default="test",
        help="Split to analyze if predictions file is not provided (default: test).",
    )
    parser.add_argument(
        "--predictions-file",
        type=str,
        default=None,
        help="Path to JSON or JSONL file mapping task_id to predicted answer.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(REPO_ROOT / "reports" / "dataset_quality" / "error_analysis_report.json"),
        help="Output JSON report file.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=str(REPO_ROOT / "reports" / "dataset_quality" / "error_analysis_report.md"),
        help="Output Markdown report file.",
    )
    parser.add_argument(
        "--format",
        choices=["console", "markdown", "json"],
        default="console",
        help="Stdout output format (default: console).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save report files to disk.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analyzer = ErrorAnalyzer()

    # 1. Load tasks
    try:
        if args.split == "all":
            tasks = load_all_tasks()
        else:
            tasks = load_dataset(args.split)
    except FileNotFoundError as e:
        print(f"\n[ERROR] Failed to load dataset: {e}")
        return 2

    predictions: dict[str, str] = {}

    if args.predictions_file:
        p_path = Path(args.predictions_file)
        if not p_path.exists():
            print(f"[ERROR] Predictions file not found: {p_path}", file=sys.stderr)
            return 1
        with p_path.open("r", encoding="utf-8") as f:
            if p_path.suffix == ".jsonl":
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        t_id = record.get("task_id") or record.get("id")
                        pred = record.get("predicted_answer") or record.get("prediction", "")
                        if t_id:
                            predictions[t_id] = pred
            else:
                data = json.load(f)
                if isinstance(data, dict):
                    predictions = data
                elif isinstance(data, list):
                    for item in data:
                        t_id = item.get("task_id") or item.get("id")
                        pred = item.get("predicted_answer") or item.get("prediction", "")
                        if t_id:
                            predictions[t_id] = pred
    else:
        # Default diagnostic simulation: 80% correct, 20% varied error injection
        for i, t in enumerate(tasks):
            if i % 5 == 0:
                predictions[t.task_id] = "Wrong answer sample"
            else:
                predictions[t.task_id] = str(t.ground_truth)

    # 2. Run error diagnosis
    report = analyzer.analyze_predictions(tasks, predictions)

    # 3. Print according to format
    if args.format == "console":
        print(analyzer.render_ascii_summary(report))
    elif args.format == "markdown":
        print(analyzer.render_markdown_report(report))
    elif args.format == "json":
        print(json.dumps(report.to_dict(), indent=2))

    # 4. Save report
    if not args.no_save:
        j_out = Path(args.output_json)
        m_out = Path(args.output_md)
        analyzer.export_report(report, json_path=j_out, md_path=m_out)
        print("\n[OK] Error analysis reports saved:")
        print(f"  -> JSON     : {j_out}")
        print(f"  -> Markdown : {m_out}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
scripts/dataset_stats.py
========================
CLI tool for computing dataset statistics, distribution tables,
multi-dimensional cross-tabulations, lexical metrics, and imbalance audits.

Usage:
    # Run full analysis on all splits and save reports (JSON + Markdown)
    python scripts/dataset_stats.py

    # Analyze a specific split (e.g. train)
    python scripts/dataset_stats.py --split train

    # Strict mode (exit with error code if critical imbalance detected)
    python scripts/dataset_stats.py --strict

    # Print markdown format directly to stdout
    python scripts/dataset_stats.py --format markdown

Person 1 owns this script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.analysis import DatasetAnalyzer
from dataset.loader import load_all_splits, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute descriptive statistics, distribution matrices, and imbalance checks for GateOrchestra dataset."
    )
    parser.add_argument(
        "--split",
        choices=["all", "train", "val", "test"],
        default="all",
        help="Which split to analyze (default: all).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(REPO_ROOT / "reports" / "dataset_quality" / "analysis_report.json"),
        help="Output path for structured JSON analysis report.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=str(REPO_ROOT / "reports" / "dataset_quality" / "analysis_report.md"),
        help="Output path for GitHub Flavored Markdown report.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving report files to disk.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero code if CRITICAL imbalance alerts are triggered.",
    )
    parser.add_argument(
        "--format",
        choices=["console", "markdown", "json"],
        default="console",
        help="Stdout output format (default: console).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analyzer = DatasetAnalyzer()

    # Load data
    try:
        if args.split == "all":
            splits = load_all_splits()
        else:
            splits = {args.split: load_dataset(args.split)}
    except FileNotFoundError as e:
        print(f"\n[ERROR] Could not load dataset splits: {e}")
        print("Please ensure splits have been created: python scripts/create_splits.py\n")
        return 2

    # Run analysis
    report = analyzer.analyze_splits(splits, dataset_name="masbench_mini")

    # Output to stdout according to format
    if args.format == "console":
        print(analyzer.render_ascii_report(report))
    elif args.format == "markdown":
        print(analyzer.render_markdown_report(report))
    elif args.format == "json":
        import json
        print(json.dumps(report.to_dict(), indent=2))

    # Save reports unless --no-save
    if not args.no_save:
        json_path = Path(args.output_json)
        md_path = Path(args.output_md)
        analyzer.export_report(report, json_path=json_path, md_path=md_path)
        print(f"\n[OK] Reports saved successfully:")
        print(f"  -> JSON     : {json_path}")
        print(f"  -> Markdown : {md_path}\n")

    # Check strict status
    critical_alerts = [a for a in report.imbalance_alerts if a.severity == "CRITICAL"]
    if args.strict and critical_alerts:
        print(f"[FAIL] Strict mode failed with {len(critical_alerts)} CRITICAL alerts.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

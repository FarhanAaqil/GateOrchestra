"""
scripts/validate_dataset.py
===========================
Dataset validation and structural auditing CLI script — Person 1, Week 3.

Validates datasets against project requirements, schema bounds, and constraints:
  1. Field schema and type adherence
  2. Length boundaries (min/max characters)
  3. Score bounds (depth in [1, 5], parallel in [1, 4])
  4. Task ID uniqueness & valid formatting
  5. Minimum dataset size assertion (>= min_acceptable_size)
  6. Source and type representation balance
  7. Outputs validation report JSON and returns appropriate exit code (0=Pass, 1=Fail)

Usage:
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --input dataset/processed/tasks_cleaned.jsonl

Person 1 owns this file.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure repo root is on path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dataset.validation.validator import DatasetValidator  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GateOrchestra Dataset Validation CLI")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input JSONL dataset file (defaults to cleaned, fallback to raw)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=str(REPO_ROOT / "reports" / "dataset_quality" / "validation_report.json"),
        help="Path to save validation report JSON",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs" / "dataset.yaml"),
        help="Path to dataset config YAML",
    )
    return parser.parse_args()


def resolve_default_input_path() -> Path:
    cleaned = REPO_ROOT / "dataset" / "processed" / "tasks_cleaned.jsonl"
    if cleaned.exists():
        return cleaned
    raw = REPO_ROOT / "dataset" / "raw" / "tasks_raw.jsonl"
    return raw


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    report_path = Path(args.report)

    if args.input:
        input_path = Path(args.input)
    else:
        input_path = resolve_default_input_path()

    print("\n" + "=" * 65)
    print("  GateOrchestra — Dataset Validation (Week 3)")
    print("=" * 65 + "\n")

    logger.info(f"Target dataset file : {input_path}")
    logger.info(f"Config file         : {config_path}")

    validator = DatasetValidator(config_path=config_path)
    report = validator.validate_file(input_path)

    # Print summary text
    print("\n" + report.summary_text() + "\n")

    # Save validation report
    DatasetValidator.save_report(report, report_path)
    logger.info(f"Report saved to: {report_path}")

    if report.is_valid:
        print("  [PASSED] VALIDATION PASSED — Dataset is structurally sound.\n")
        sys.exit(0)
    else:
        print("  [FAILED] VALIDATION FAILED — Errors detected in dataset.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

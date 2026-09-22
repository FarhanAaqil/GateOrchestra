"""
scripts/clean_dataset.py
========================
Dataset cleaning and deduplication CLI script — Person 1, Week 3.

Pipeline steps:
  1. Load raw labeled tasks from dataset/raw/tasks_raw.jsonl
  2. Perform unicode sanitization, whitespace collapsing, and answer canonicalization
  3. Perform exact and fuzzy deduplication (Jaccard + tri-gram similarity)
  4. Save cleaned dataset to dataset/processed/tasks_cleaned.jsonl
  5. Save detailed audit report to reports/dataset_quality/cleaning_report.json
  6. Print summary statistics

Usage:
    python scripts/clean_dataset.py
    python scripts/clean_dataset.py --input dataset/raw/tasks_raw.jsonl --threshold 0.85

Person 1 owns this file.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure repo root is on path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from dataset.cleaning.cleaner import clean_dataset_records  # noqa: E402
from dataset.cleaning.deduplicator import FuzzyDeduplicator  # noqa: E402
from shared.schemas import Task  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GateOrchestra Dataset Cleaning & Deduplication CLI"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(REPO_ROOT / "dataset" / "raw" / "tasks_raw.jsonl"),
        help="Path to input raw tasks JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "dataset" / "processed" / "tasks_cleaned.jsonl"),
        help="Path to output cleaned tasks JSONL file",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=str(REPO_ROOT / "reports" / "dataset_quality" / "cleaning_report.json"),
        help="Path to write JSON cleaning report",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fuzzy deduplication similarity threshold (0.0 - 1.0)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs" / "dataset.yaml"),
        help="Path to dataset config YAML",
    )
    return parser.parse_args()


def load_tasks(file_path: Path) -> list[Task]:
    tasks: list[Task] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                tasks.append(Task(**json.loads(line)))
            except Exception as e:
                logger.error(f"Error parsing line {line_no} in {file_path.name}: {e}")
    return tasks


def save_tasks(tasks: list[Task], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task.model_dump(), ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)
    config_path = Path(args.config)

    print("\n" + "=" * 65)
    print("  GateOrchestra — Dataset Cleaning & Deduplication (Week 3)")
    print("=" * 65 + "\n")

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.info("Run `python scripts/build_dataset.py` first.")
        sys.exit(1)

    # Load configuration
    cleaning_cfg: dict = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            cfg_data = yaml.safe_load(f) or {}
            cleaning_cfg = cfg_data.get("cleaning", {})

    threshold = args.threshold
    if threshold is None:
        threshold = cleaning_cfg.get("fuzzy_dedup", {}).get("similarity_threshold", 0.85)

    # Step 1: Load tasks
    logger.info(f"Loading raw tasks from {input_path}...")
    raw_tasks = load_tasks(input_path)
    logger.info(f"  Loaded {len(raw_tasks)} tasks")

    # Step 2: Clean and sanitize tasks
    logger.info("Applying text normalization and answer canonicalization...")
    sanitized_tasks, clean_summary = clean_dataset_records(
        raw_tasks,
        config=cleaning_cfg,
    )
    logger.info(f"  Tasks modified by sanitizer: {clean_summary['tasks_modified_count']}")
    for mod_type, count in clean_summary["modifications_by_type"].items():
        logger.info(f"    - {mod_type}: {count}")

    # Step 3: Deduplicate
    logger.info(f"Performing exact and fuzzy deduplication (threshold={threshold})...")
    deduplicator = FuzzyDeduplicator(similarity_threshold=threshold)
    cleaned_tasks, exact_dups, fuzzy_dups = deduplicator.deduplicate(sanitized_tasks)
    logger.info(f"  Unique tasks retained : {len(cleaned_tasks)}")
    logger.info(f"  Exact duplicates found: {len(exact_dups)}")
    logger.info(f"  Fuzzy duplicates found: {len(fuzzy_dups)}")

    # Step 4: Save cleaned dataset
    logger.info(f"Saving cleaned tasks to {output_path}...")
    save_tasks(cleaned_tasks, output_path)
    logger.info(f"  [OK] Successfully wrote {len(cleaned_tasks)} tasks -> {output_path.name}")

    # Step 5: Save audit report
    report_data = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "total_input_tasks": len(raw_tasks),
        "total_cleaned_tasks": len(cleaned_tasks),
        "total_removed": len(raw_tasks) - len(cleaned_tasks),
        "sanitization_summary": {
            "tasks_modified": clean_summary["tasks_modified_count"],
            "breakdown": clean_summary["modifications_by_type"],
        },
        "deduplication_summary": {
            "exact_duplicates_count": len(exact_dups),
            "fuzzy_duplicates_count": len(fuzzy_dups),
            "similarity_threshold": threshold,
            "exact_duplicates": exact_dups,
            "fuzzy_duplicates": fuzzy_dups,
        },
        "task_audit_log": clean_summary["task_audit_log"],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    logger.info(f"  [OK] Saved cleaning report -> {report_path.name}")

    # Summary
    print("\n" + "=" * 65)
    print("  CLEANING & DEDUPLICATION SUMMARY")
    print("=" * 65)
    print(f"  Input tasks              : {len(raw_tasks)}")
    print(f"  Modifications applied    : {clean_summary['tasks_modified_count']}")
    print(f"  Duplicates removed       : {len(exact_dups) + len(fuzzy_dups)}")
    print(f"  Final clean tasks count  : {len(cleaned_tasks)}")
    print("=" * 65)
    print("  [OK] Dataset cleaning complete.")
    print("       Next: python scripts/validate_dataset.py\n")


if __name__ == "__main__":
    main()

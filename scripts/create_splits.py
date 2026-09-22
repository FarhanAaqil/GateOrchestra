"""
scripts/create_splits.py
========================
Stratified dataset splitting and leakage audit CLI script — Person 1, Week 4.

Loads cleaned tasks from dataset/processed/tasks_cleaned.jsonl,
partitions them into stratified train/val/test splits (60% / 20% / 20%),
audits for data contamination, and saves the partitioned datasets.

Person 1 owns this file.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from dataset.splits.leakage_auditor import (  # noqa: E402
    LeakageAuditor,
    save_leakage_report,
)
from dataset.splits.stratified_splitter import (  # noqa: E402
    StratifiedSplitter,
    compute_split_statistics,
)
from shared.config import (  # noqa: E402
    CONFIGS_DIR,
    DATASET_DIR,
    RANDOM_SEED,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
)
from shared.schemas import Task  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("create_splits")


def load_tasks_from_jsonl(path: Path) -> list[Task]:
    """Read tasks from a JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"Input task file not found: {path}")

    tasks: list[Task] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(Task(**json.loads(line)))
    return tasks


def save_split_jsonl(tasks: list[Task], path: Path) -> None:
    """Save tasks to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t.model_dump(), ensure_ascii=False) + "\n")


def run_split_pipeline(
    input_file: Path | None = None,
    output_dir: Path | None = None,
    config_file: Path | None = None,
) -> int:
    """Main execution workflow for Week 4 splitting and leakage audit."""
    if config_file is None:
        config_file = CONFIGS_DIR / "dataset.yaml"

    # Default ratios & seed
    train_ratio = TRAIN_SPLIT
    val_ratio = VAL_SPLIT
    test_ratio = TEST_SPLIT
    seed = RANDOM_SEED

    if config_file.exists():
        with config_file.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            splits_cfg = cfg.get("splits", {})
            train_ratio = splits_cfg.get("train", train_ratio)
            val_ratio = splits_cfg.get("val", val_ratio)
            test_ratio = splits_cfg.get("test", test_ratio)
            seed = cfg.get("random_seed", seed)

    if input_file is None:
        cleaned_path = REPO_ROOT / "dataset" / "processed" / "tasks_cleaned.jsonl"
        raw_path = REPO_ROOT / "dataset" / "raw" / "tasks_raw.jsonl"
        input_file = cleaned_path if cleaned_path.exists() else raw_path

    if output_dir is None:
        output_dir = DATASET_DIR

    logger.info(f"Loading input tasks from: {input_file}")
    tasks = load_tasks_from_jsonl(input_file)
    logger.info(f"Loaded {len(tasks)} tasks.")

    logger.info(
        f"Partitioning with StratifiedSplitter (train={train_ratio:.0%}, "
        f"val={val_ratio:.0%}, test={test_ratio:.0%}, seed={seed})..."
    )
    splitter = StratifiedSplitter(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    result = splitter.split(tasks)
    splits_dict = result.as_dict()

    # Save to masbench_mini (both root flat files and subdirectories for backward compatibility)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_tasks in splits_dict.items():
        # Flat format: dataset/masbench_mini/train.jsonl
        flat_path = output_dir / f"{split_name}.jsonl"
        save_split_jsonl(split_tasks, flat_path)

        # Nested format: dataset/masbench_mini/train/train.jsonl
        nested_path = output_dir / split_name / f"{split_name}.jsonl"
        save_split_jsonl(split_tasks, nested_path)

        logger.info(f"  [OK] Saved {len(split_tasks)} tasks -> {flat_path.name} & {nested_path}")

    # Run cross-split leakage audit
    logger.info("Running cross-split data leakage audit...")
    auditor = LeakageAuditor(ngram_n=3, ngram_threshold=0.80)
    leakage_report = auditor.audit(splits_dict)

    report_path = REPO_ROOT / "reports" / "dataset_quality" / "leakage_report.json"
    save_leakage_report(leakage_report, report_path)

    stats = compute_split_statistics(splits_dict)
    logger.info(f"Split distributions computed across {len(stats)} partitions.")

    print("\n" + "=" * 65)
    print("  GateOrchestra — Dataset Stratified Splitting (Week 4)")
    print("=" * 65)
    print(f"  Total Input Tasks : {len(tasks)}")
    print(f"  Train Split       : {len(result.train)} tasks ({len(result.train)/len(tasks):.1%})")
    print(f"  Validation Split  : {len(result.val)} tasks ({len(result.val)/len(tasks):.1%})")
    print(f"  Test Split        : {len(result.test)} tasks ({len(result.test)/len(tasks):.1%})")
    print("-" * 65)
    print("  Leakage Audit Status:")
    print(
        f"    Is Clean        : {'[PASS] Clean' if leakage_report.is_clean else '[FAIL] Leakage Detected'}"
    )
    print(f"    Violations Count: {len(leakage_report.violations)}")
    print(f"    Pairwise Checks : {leakage_report.pair_comparisons} comparisons")
    print(f"    Max N-gram Sim  : {leakage_report.max_ngram_similarity:.4f}")
    print(f"    Report Saved    : {report_path}")
    print("=" * 65 + "\n")

    return 0 if leakage_report.is_clean else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Partition GateOrchestra dataset into stratified train/val/test splits."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to tasks_cleaned.jsonl input file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for split JSONL files (default: dataset/masbench_mini)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to dataset.yaml configuration file",
    )
    args = parser.parse_args()

    sys.exit(run_split_pipeline(args.input, args.output_dir, args.config))


if __name__ == "__main__":
    main()

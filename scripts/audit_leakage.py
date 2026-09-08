"""
scripts/audit_leakage.py
========================
Standalone CLI for auditing data leakage across dataset partitions — Person 1, Week 4.

Person 1 owns this file.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.loader import load_dataset  # noqa: E402
from dataset.splits.leakage_auditor import (  # noqa: E402
    LeakageAuditor,
    save_leakage_report,
)
from shared.schemas import Task  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("audit_leakage")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit dataset partitions for train/val/test data leakage."
    )
    parser.add_argument(
        "--ngram",
        type=int,
        default=3,
        help="N-gram size for overlap check (default: 3)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Similarity threshold to flag high overlap (default: 0.80)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "reports" / "dataset_quality" / "leakage_report.json",
        help="Output JSON file for leakage report",
    )
    args = parser.parse_args()

    logger.info("Loading train, val, and test splits...")
    try:
        splits: dict[str, list[Task]] = {
            "train": load_dataset("train"),
            "val": load_dataset("val"),
            "test": load_dataset("test"),
        }
    except FileNotFoundError as e:
        logger.error(f"Failed to load splits: {e}")
        sys.exit(1)

    logger.info(
        f"Loaded splits: train={len(splits['train'])}, "
        f"val={len(splits['val'])}, test={len(splits['test'])}"
    )

    auditor = LeakageAuditor(ngram_n=args.ngram, ngram_threshold=args.threshold)
    report = auditor.audit(splits)

    save_leakage_report(report, args.output)

    print("\n" + "=" * 60)
    print("  GateOrchestra — Data Leakage Audit (Week 4)")
    print("=" * 60)
    print(
        f"  Status          : {'[PASS] No Leakage Detected' if report.is_clean else '[FAIL] Leakage Detected'}"
    )
    print(f"  Total Inspected : {report.total_inspected} tasks")
    print(f"  Pairwise Checks : {report.pair_comparisons}")
    print(f"  Max N-gram Sim  : {report.max_ngram_similarity:.4f}")
    print(f"  Violations      : {len(report.violations)}")
    if report.violations:
        for v in report.violations[:5]:
            print(
                f"    - [{v.violation_type}] {v.task_id_a} ({v.split_a}) <-> {v.task_id_b} ({v.split_b}): {v.details}"
            )
    print(f"  Report saved to : {args.output}")
    print("=" * 60 + "\n")

    sys.exit(0 if report.is_clean else 1)


if __name__ == "__main__":
    main()

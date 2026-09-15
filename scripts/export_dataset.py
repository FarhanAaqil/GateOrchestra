"""
scripts/export_dataset.py
==========================
Dataset Export CLI — Week 5 Deliverable (Person 1).

Exports the GateOrchestra MASBench-mini dataset splits to portable formats
for external analysis (Google Sheets, Excel, Pandas, Hugging Face).

Supported export formats:
    --format csv        One CSV file per split (default)
    --format hf         Hugging Face DatasetDict (requires `datasets` library)
    --format combined   All splits merged into a single CSV

Usage::

    # Export all splits to CSV (default output dir: exports/dataset/)
    python scripts/export_dataset.py

    # Export to a custom directory
    python scripts/export_dataset.py --output-dir reports/exports

    # Export all splits to a single merged CSV
    python scripts/export_dataset.py --format combined

    # Export to Hugging Face DatasetDict (saved locally as Arrow files)
    python scripts/export_dataset.py --format hf --output-dir exports/hf_dataset

    # Export only specific splits
    python scripts/export_dataset.py --splits train val

Person 1 owns this file.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

# ── Bootstrap path so we can import shared.* and dataset.* ────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.repository import JSONLTaskRepository  # noqa: E402
from shared.schemas import Task  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# CSV column order — matches Task schema field order
_TASK_FIELDS = [
    "task_id",
    "question",
    "context",
    "ground_truth",
    "depth_score",
    "parallel_score",
    "source_dataset",
]

DEFAULT_OUTPUT_DIR = ROOT / "exports" / "dataset"


# ─────────────────────────────────────────────────────────────────────────────
# Export helpers
# ─────────────────────────────────────────────────────────────────────────────


def _task_to_row(task: Task, split: str | None = None) -> dict:
    """Convert a Task pydantic model to a flat dict for CSV export."""
    row = {field: getattr(task, field, None) for field in _TASK_FIELDS}
    if split is not None:
        row["split"] = split
    return row


def export_csv_per_split(
    repo: JSONLTaskRepository,
    splits: list[str],
    output_dir: Path,
) -> dict[str, Path]:
    """Export each split to its own CSV file.

    Args:
        repo:       Initialised ``JSONLTaskRepository``.
        splits:     List of split names to export.
        output_dir: Destination directory (created if necessary).

    Returns:
        Dict mapping split name to the Path of the exported CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}

    for split in splits:
        tasks = repo.list_by_split(split)
        if not tasks:
            logger.warning("[Export] Split '%s' is empty — skipping.", split)
            continue

        out_path = output_dir / f"{split}.csv"
        fieldnames = _TASK_FIELDS
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for task in tasks:
                writer.writerow(_task_to_row(task))

        exported[split] = out_path
        logger.info("[Export] CSV  %-6s  %4d tasks  ->  %s", split, len(tasks), out_path)

    return exported


def export_csv_combined(
    repo: JSONLTaskRepository,
    splits: list[str],
    output_dir: Path,
) -> Path:
    """Export all specified splits into a single CSV with a 'split' column.

    Args:
        repo:       Initialised ``JSONLTaskRepository``.
        splits:     List of split names to combine.
        output_dir: Destination directory (created if necessary).

    Returns:
        Path to the combined CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "all_splits_combined.csv"
    fieldnames = ["split"] + _TASK_FIELDS
    total = 0

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for split in splits:
            tasks = repo.list_by_split(split)
            for task in tasks:
                writer.writerow(_task_to_row(task, split=split))
            total += len(tasks)
            logger.info("[Export] combined <- %-6s  %d tasks", split, len(tasks))

    logger.info("[Export] Combined CSV  %d total tasks  ->  %s", total, out_path)
    return out_path


def export_huggingface(
    repo: JSONLTaskRepository,
    splits: list[str],
    output_dir: Path,
) -> Path:
    """Export dataset splits as a Hugging Face ``DatasetDict`` (Arrow format).

    Requires the ``datasets`` package (``pip install datasets``).

    Args:
        repo:       Initialised ``JSONLTaskRepository``.
        splits:     List of split names to export.
        output_dir: Destination directory for the saved HF dataset.

    Returns:
        Path to the saved Hugging Face dataset directory.
    """
    try:
        from datasets import Dataset, DatasetDict  # type: ignore[import]
    except ImportError:
        logger.error(
            "[Export] 'datasets' package not installed.  "
            "Run: pip install datasets"
        )
        sys.exit(1)

    split_dicts: dict[str, Dataset] = {}
    for split in splits:
        tasks = repo.list_by_split(split)
        if not tasks:
            logger.warning("[Export] Split '%s' is empty — skipping.", split)
            continue
        rows = [_task_to_row(task) for task in tasks]
        split_dicts[split] = Dataset.from_list(rows)
        logger.info(
            "[Export] HF %-6s  %4d tasks  (schema: %s)",
            split, len(tasks), list(rows[0].keys()),
        )

    ds_dict = DatasetDict(split_dicts)
    output_dir.mkdir(parents=True, exist_ok=True)
    ds_dict.save_to_disk(str(output_dir))
    logger.info("[Export] Hugging Face DatasetDict saved  ->  %s", output_dir)
    return output_dir


def export_metadata_json(
    repo: JSONLTaskRepository,
    splits: list[str],
    output_dir: Path,
) -> Path:
    """Write a JSON metadata summary alongside exported files.

    Args:
        repo:       Initialised ``JSONLTaskRepository``.
        splits:     Splits that were exported.
        output_dir: Directory to write ``export_metadata.json`` into.

    Returns:
        Path to the metadata JSON file.
    """
    from collections import Counter

    meta: dict = {"splits": {}}
    for split in splits:
        tasks = repo.list_by_split(split)
        source_counter = Counter(
            t.source_dataset for t in tasks if t.source_dataset
        )
        depth_counter = Counter(
            t.depth_score for t in tasks if t.depth_score is not None
        )
        parallel_counter = Counter(
            t.parallel_score for t in tasks if t.parallel_score is not None
        )
        meta["splits"][split] = {
            "count": len(tasks),
            "source_distribution": dict(source_counter.most_common()),
            "depth_distribution": {str(k): v for k, v in sorted(depth_counter.items())},
            "parallel_distribution": {str(k): v for k, v in sorted(parallel_counter.items())},
        }

    meta["total"] = sum(v["count"] for v in meta["splits"].values())

    out_path = output_dir / "export_metadata.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    logger.info("[Export] Metadata JSON  ->  %s", out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export_dataset",
        description="Export GateOrchestra MASBench-mini dataset to portable formats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/export_dataset.py
  python scripts/export_dataset.py --format combined --output-dir reports/exports
  python scripts/export_dataset.py --format hf --output-dir exports/hf_dataset
  python scripts/export_dataset.py --splits train val
""",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "combined", "hf"],
        default="csv",
        help="Export format: 'csv' (one file per split), 'combined' (single merged CSV), "
             "or 'hf' (Hugging Face DatasetDict). Default: csv.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        metavar="SPLIT",
        help="Splits to export. Default: train val test.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip writing export_metadata.json alongside the exported files.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    repo = JSONLTaskRepository()
    available = repo.split_names()
    if not available:
        logger.error(
            "[Export] No dataset splits found in %s.\n"
            "         Run: python scripts/create_splits.py  first.",
            repo._root,
        )
        sys.exit(1)

    # Filter requested splits to those that actually exist
    splits = [s for s in args.splits if s in available]
    missing = [s for s in args.splits if s not in available]
    if missing:
        logger.warning("[Export] Splits not found on disk (skipping): %s", missing)
    if not splits:
        logger.error("[Export] No valid splits to export.  Aborting.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("GateOrchestra Dataset Export")
    logger.info("  Format  : %s", args.format)
    logger.info("  Splits  : %s", splits)
    logger.info("  Out dir : %s", args.output_dir)
    logger.info("=" * 60)

    if args.format == "csv":
        export_csv_per_split(repo, splits, args.output_dir)
    elif args.format == "combined":
        export_csv_combined(repo, splits, args.output_dir)
    elif args.format == "hf":
        export_huggingface(repo, splits, args.output_dir)

    if not args.no_metadata:
        export_metadata_json(repo, splits, args.output_dir)

    logger.info("[Export] Done.")


if __name__ == "__main__":
    main()

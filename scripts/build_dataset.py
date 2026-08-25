"""
scripts/build_dataset.py
=========================
Master dataset construction script — Person 1, Week 1.

Runs the complete raw dataset pipeline:
  1. Load raw task pool (160 tasks)
  2. Validate and deduplicate
  3. Assign task IDs
  4. Extract labeling features
  5. Assign depth_score (1–5)
  6. Assign parallel_score (1–4)
  7. Save to dataset/raw/tasks_raw.jsonl  (raw Tasks + features)
  8. Save feature matrix to dataset/processed/features.jsonl
  9. Print summary statistics

Run from repo root:
    python scripts/build_dataset.py

Output files:
    dataset/raw/tasks_raw.jsonl         — schema-valid Tasks with labels
    dataset/processed/features.jsonl    — intermediate feature matrix

Person 1 owns this file.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Ensure repo root is on path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared.schemas import Task
from dataset.generation.task_builder import build_tasks
from dataset.generation.task_pool import RAW_TASKS
from dataset.labeling.feature_extractor import extract_labeling_features
from dataset.labeling.depth_labeler import assign_depth
from dataset.labeling.parallel_labeler import assign_parallel

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Output paths
# ─────────────────────────────────────────────────────────────────────────────

RAW_DIR = REPO_ROOT / "dataset" / "raw"
PROCESSED_DIR = REPO_ROOT / "dataset" / "processed"
RAW_TASKS_FILE = RAW_DIR / "tasks_raw.jsonl"
FEATURES_FILE = PROCESSED_DIR / "features.jsonl"


def main() -> None:
    print("\n" + "=" * 65)
    print("  GateOrchestra — Dataset Build Script  (Person 1, Week 1)")
    print("=" * 65 + "\n")

    # ── Step 1 & 2: Build and validate raw tasks ───────────────────────────
    logger.info("STEP 1-3: Building tasks from raw pool...")
    tasks_unlabeled, rejected = build_tasks(raw_tasks=RAW_TASKS, verbose=True)

    if not tasks_unlabeled:
        logger.error("No tasks produced — aborting.")
        sys.exit(1)

    # ── Step 3: Feature extraction + labeling ─────────────────────────────
    logger.info(f"\nSTEP 4-6: Extracting features and assigning labels for {len(tasks_unlabeled)} tasks...")

    # We need the hop_hint from the raw pool — build a lookup by question fingerprint
    raw_hop_lookup = _build_hop_lookup()

    labeled_tasks: list[Task] = []
    feature_rows: list[dict] = []

    for task in tasks_unlabeled:
        hop_hint = raw_hop_lookup.get(task.question, 1)

        # Extract raw features
        features = extract_labeling_features(
            question=task.question,
            context=task.context,
            hop_hint=hop_hint,
        )

        # Assign labels
        depth_result = assign_depth(features)
        parallel_result = assign_parallel(features)

        depth_score = depth_result["depth_score"]
        parallel_score = parallel_result["parallel_score"]

        # Rebuild Task with labels (Task is frozen/immutable, must create new)
        labeled_task = Task(
            task_id=task.task_id,
            question=task.question,
            context=task.context,
            depth_score=depth_score,
            parallel_score=parallel_score,
            ground_truth=task.ground_truth,
            source_dataset=task.source_dataset,
        )
        labeled_tasks.append(labeled_task)

        # Feature row for analysis
        feature_row = {
            "task_id": task.task_id,
            "source_dataset": task.source_dataset,
            **features,
            "depth_score": depth_score,
            "depth_raw_score": depth_result["depth_raw_score"],
            "parallel_score": parallel_score,
            "parallel_raw_score": parallel_result["parallel_raw_score"],
        }
        feature_rows.append(feature_row)

    logger.info(f"  Labels assigned to {len(labeled_tasks)} tasks")

    # ── Step 4: Save raw tasks ─────────────────────────────────────────────
    logger.info(f"\nSTEP 7: Saving raw labeled tasks to {RAW_TASKS_FILE}...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    _save_tasks_jsonl(labeled_tasks, RAW_TASKS_FILE)
    logger.info(f"  ✅ Saved {len(labeled_tasks)} tasks → {RAW_TASKS_FILE.name}")

    # ── Step 5: Save feature matrix ────────────────────────────────────────
    logger.info(f"\nSTEP 8: Saving feature matrix to {FEATURES_FILE}...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    _save_jsonl(feature_rows, FEATURES_FILE)
    logger.info(f"  ✅ Saved {len(feature_rows)} feature rows → {FEATURES_FILE.name}")

    # ── Step 6: Summary statistics ─────────────────────────────────────────
    _print_summary(labeled_tasks, rejected)


def _build_hop_lookup() -> dict[str, int]:
    """Build a normalized-question → hop_hint lookup from the raw pool."""
    lookup: dict[str, int] = {}
    for raw in RAW_TASKS:
        q_normalized = " ".join(raw["question"].strip().split())
        lookup[q_normalized] = raw.get("hop_hint", 1)
    return lookup


def _save_tasks_jsonl(tasks: list[Task], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task.model_dump(), ensure_ascii=False) + "\n")


def _save_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _print_summary(tasks: list[Task], rejected: list[dict]) -> None:
    print("\n" + "=" * 65)
    print("  DATASET BUILD SUMMARY")
    print("=" * 65)
    print(f"  Total tasks built        : {len(tasks)}")
    print(f"  Tasks rejected/skipped   : {len(rejected)}")

    # Source distribution
    sources: dict[str, int] = {}
    for t in tasks:
        src = t.source_dataset or "unknown"
        sources[src] = sources.get(src, 0) + 1
    print("\n  Source distribution:")
    for src, cnt in sorted(sources.items()):
        print(f"    {src:30s}: {cnt}")

    # Depth distribution
    depth_counts: dict[int, int] = {}
    for t in tasks:
        if t.depth_score is not None:
            depth_counts[t.depth_score] = depth_counts.get(t.depth_score, 0) + 1
    print("\n  Depth score distribution (1=shallow, 5=deep):")
    for d in sorted(depth_counts):
        bar = "#" * depth_counts[d]
        print(f"    depth {d}: {depth_counts[d]:3d}  {bar}")

    # Parallel distribution
    par_counts: dict[int, int] = {}
    for t in tasks:
        if t.parallel_score is not None:
            par_counts[t.parallel_score] = par_counts.get(t.parallel_score, 0) + 1
    print("\n  Parallel score distribution (1=sequential, 4=highly parallel):")
    for p in sorted(par_counts):
        bar = "#" * par_counts[p]
        print(f"    parallel {p}: {par_counts[p]:3d}  {bar}")

    print("\n" + "=" * 65)
    print("  [OK] Week 1 build complete.")
    print(f"     Next: python scripts/clean_dataset.py")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

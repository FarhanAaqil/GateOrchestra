"""
dataset/generation/task_builder.py
====================================
Converts the raw task pool into schema-validated Task objects.

Pipeline:
  1. Load raw dicts from task_pool.py
  2. Normalize (strip whitespace, normalize answer strings)
  3. Deduplicate (exact + normalized)
  4. Validate (length, required fields)
  5. Assign stable task IDs
  6. Return list[Task] ready for labeling

Person 1 owns this file.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sys
from pathlib import Path

# Allow running as a standalone script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataset.generation.task_pool import RAW_TASKS
from shared.schemas import Task

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# ID generation
# ─────────────────────────────────────────────────────────────────────────────

# Prefix mapping for readable IDs
_TYPE_PREFIX: dict[str, str] = {
    "multihop_bridge": "bridge",
    "multihop_compositional": "comp",
    "arithmetic": "arith",
    "comparison": "cmp",
}


def _make_task_id(task_type: str, index: int) -> str:
    """Generate a stable, readable task ID.

    Format: <type_prefix>_<zero_padded_index>
    Example: bridge_001, arith_042
    """
    prefix = _TYPE_PREFIX.get(task_type, "task")
    return f"{prefix}_{index:03d}"


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────


def _normalize_text(text: str) -> str:
    """Strip, collapse whitespace."""
    return " ".join(text.strip().split())


def _normalize_for_dedup(text: str) -> str:
    """Aggressive normalization for duplicate detection.

    Lowercase, remove punctuation, collapse whitespace.
    """
    s = text.lower()
    s = re.sub(r"[^\w\s]", "", s)
    return " ".join(s.split())


def _question_fingerprint(question: str) -> str:
    """MD5 fingerprint of normalized question for fast dedup."""
    normalized = _normalize_for_dedup(question)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

_VALID_SOURCES = {
    "hotpotqa_style",
    "musique_style",
    "template_arithmetic",
    "template_comparison",
}

_VALID_TYPES = {
    "multihop_bridge",
    "multihop_compositional",
    "arithmetic",
    "comparison",
}


def _validate_raw(task: dict, idx: int) -> list[str]:
    """Return a list of validation errors (empty = valid)."""
    errors: list[str] = []
    q = task.get("question", "")
    a = task.get("answer", "")
    src = task.get("source", "")
    tt = task.get("task_type", "")

    if not q or len(q.strip()) < 10:
        errors.append(f"[{idx}] question too short: {q!r}")
    if len(q) > 600:
        errors.append(f"[{idx}] question too long ({len(q)} chars)")
    if not a or len(a.strip()) < 1:
        errors.append(f"[{idx}] missing answer")
    if src not in _VALID_SOURCES:
        errors.append(f"[{idx}] unknown source: {src!r}")
    if tt not in _VALID_TYPES:
        errors.append(f"[{idx}] unknown task_type: {tt!r}")
    return errors


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def build_tasks(
    raw_tasks: list[dict] | None = None,
    *,
    verbose: bool = True,
) -> tuple[list[Task], list[dict]]:
    """Build and validate Task objects from the raw task pool.

    Args:
        raw_tasks: Override the default task pool (for testing).
        verbose:   Log progress and validation results.

    Returns:
        (tasks, rejected)
        tasks    — list of valid Task objects (depth/parallel labels NOT yet assigned)
        rejected — list of dicts with keys: raw_task, errors
    """
    if raw_tasks is None:
        raw_tasks = RAW_TASKS

    if verbose:
        logger.info(f"[TaskBuilder] Starting build from {len(raw_tasks)} raw tasks")

    # ── Step 1: Validate raw dicts ────────────────────────────────────────────
    valid_raws: list[dict] = []
    rejected: list[dict] = []

    for i, raw in enumerate(raw_tasks):
        errors = _validate_raw(raw, i)
        if errors:
            for e in errors:
                logger.warning(f"  REJECT: {e}")
            rejected.append({"raw_task": raw, "errors": errors})
        else:
            valid_raws.append(raw)

    if verbose:
        logger.info(f"  After validation: {len(valid_raws)} valid, {len(rejected)} rejected")

    # ── Step 2: Deduplicate ───────────────────────────────────────────────────
    seen_fingerprints: set[str] = set()
    deduped_raws: list[dict] = []
    dup_count = 0

    for raw in valid_raws:
        fp = _question_fingerprint(raw["question"])
        if fp in seen_fingerprints:
            logger.warning(f"  DEDUP: duplicate question skipped: {raw['question'][:60]!r}")
            dup_count += 1
        else:
            seen_fingerprints.add(fp)
            deduped_raws.append(raw)

    if verbose:
        logger.info(f"  After dedup: {len(deduped_raws)} tasks ({dup_count} duplicates removed)")

    # ── Step 3: Normalize text and assign IDs ────────────────────────────────
    # Track per-type counters for stable IDs
    type_counters: dict[str, int] = {}
    tasks: list[Task] = []

    for raw in deduped_raws:
        tt = raw["task_type"]
        type_counters[tt] = type_counters.get(tt, 0) + 1
        task_id = _make_task_id(tt, type_counters[tt])

        question = _normalize_text(raw["question"])
        answer = _normalize_text(raw["answer"])
        context = _normalize_text(raw["context"]) if raw.get("context") else None

        # depth_score and parallel_score are None at this stage — set by labeler
        task = Task(
            task_id=task_id,
            question=question,
            answer=answer,
            context=context,
            depth_score=None,
            parallel_score=None,
            ground_truth=answer,
            source_dataset=raw["source"],
        )
        tasks.append(task)

    if verbose:
        logger.info(f"[TaskBuilder] Done — {len(tasks)} Task objects created")
        _log_type_distribution(tasks)

    return tasks, rejected


def _log_type_distribution(tasks: list[Task]) -> None:
    """Log task type distribution derived from task_id prefix."""
    counts: dict[str, int] = {}
    for t in tasks:
        prefix = t.task_id.split("_")[0]
        counts[prefix] = counts.get(prefix, 0) + 1
    logger.info("  Task type distribution:")
    for prefix, count in sorted(counts.items()):
        logger.info(f"    {prefix:10s}: {count}")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    tasks, rejected = build_tasks(verbose=True)
    print(f"\n✅ {len(tasks)} tasks built, {len(rejected)} rejected")
    print("\nSample tasks:")
    for t in tasks[:3]:
        print(f"  [{t.task_id}] {t.question[:70]}...")

"""
dataset/validation/structural_checks.py
======================================
Low-level structural and semantic validation rules for GateOrchestra Task objects.

Performs deterministic integrity checks:
  1. Field existence and type constraints
  2. Length boundaries for questions and answers
  3. Score range boundedness for depth_score and parallel_score
  4. Task ID format and uniqueness
  5. Source dataset validity

Person 1 owns this file.
"""

from __future__ import annotations

import re
from typing import Any

from shared.schemas import Task

# Allowed source datasets
_DEFAULT_VALID_SOURCES = {
    "hotpotqa_style",
    "musique_style",
    "template_arithmetic",
    "template_comparison",
}

# Task ID regex pattern (e.g. bridge_001, comp_042, arith_100, cmp_015)
_TASK_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]+$")


def check_task_fields(
    task: Task,
    *,
    valid_sources: set[str] | None = None,
    min_q_len: int = 10,
    max_q_len: int = 600,
    min_a_len: int = 1,
    max_a_len: int = 200,
    depth_range: tuple[int, int] = (1, 5),
    parallel_range: tuple[int, int] = (1, 4),
) -> list[str]:
    """Perform comprehensive structural checks on a single Task instance.

    Args:
        task: The Task to validate.
        valid_sources: Set of accepted source_dataset strings.
        min_q_len: Minimum question length in characters.
        max_q_len: Maximum question length in characters.
        min_a_len: Minimum answer/ground_truth length in characters.
        max_a_len: Maximum answer/ground_truth length in characters.
        depth_range: (min_depth, max_depth) inclusive.
        parallel_range: (min_parallel, max_parallel) inclusive.

    Returns:
        List of validation error messages (empty list if valid).
    """
    errors: list[str] = []
    tid = task.task_id

    # 1. Task ID checks
    if not tid or not tid.strip():
        errors.append("Task ID is missing or empty")
    elif " " in tid:
        errors.append(f"Task ID contains spaces: {tid!r}")
    elif not _TASK_ID_REGEX.match(tid):
        errors.append(f"Task ID contains invalid characters: {tid!r}")

    # 2. Question checks
    q = task.question
    if not q or not q.strip():
        errors.append("Question is missing or empty")
    else:
        q_len = len(q.strip())
        if q_len < min_q_len:
            errors.append(f"Question too short ({q_len} chars < min {min_q_len}): {q[:40]!r}")
        if q_len > max_q_len:
            errors.append(f"Question too long ({q_len} chars > max {max_q_len})")

    # 3. Answer / Ground truth checks
    gt = task.ground_truth
    if gt is None or not str(gt).strip():
        errors.append("Ground truth is missing or empty")
    else:
        gt_len = len(str(gt).strip())
        if gt_len < min_a_len:
            errors.append(f"Ground truth too short ({gt_len} chars < min {min_a_len})")
        if gt_len > max_a_len:
            errors.append(f"Ground truth too long ({gt_len} chars > max {max_a_len})")

    # 4. Source dataset check
    sources = valid_sources or _DEFAULT_VALID_SOURCES
    if task.source_dataset is None:
        errors.append("source_dataset is None")
    elif task.source_dataset not in sources:
        errors.append(f"Unknown source_dataset: {task.source_dataset!r} (expected one of {sorted(sources)})")

    # 5. Depth score checks
    if task.depth_score is not None:
        if not (depth_range[0] <= task.depth_score <= depth_range[1]):
            errors.append(
                f"depth_score {task.depth_score} outside valid range [{depth_range[0]}, {depth_range[1]}]"
            )

    # 6. Parallel score checks
    if task.parallel_score is not None:
        if not (parallel_range[0] <= task.parallel_score <= parallel_range[1]):
            errors.append(
                f"parallel_score {task.parallel_score} outside valid range [{parallel_range[0]}, {parallel_range[1]}]"
            )

    return errors


def check_collection_integrity(tasks: list[Task]) -> list[dict[str, Any]]:
    """Check dataset-wide integrity (uniqueness of IDs, uniqueness of questions).

    Args:
        tasks: Complete list of tasks.

    Returns:
        List of integrity violations dicts with keys: type, message, task_ids.
    """
    violations: list[dict[str, Any]] = []

    if not tasks:
        violations.append({
            "type": "empty_dataset",
            "message": "Dataset task list is completely empty",
            "task_ids": [],
        })
        return violations

    # Check Task ID collisions
    seen_ids: dict[str, list[int]] = {}
    for idx, t in enumerate(tasks):
        seen_ids.setdefault(t.task_id, []).append(idx)

    id_collisions = {tid: indices for tid, indices in seen_ids.items() if len(indices) > 1}
    if id_collisions:
        for tid, indices in id_collisions.items():
            violations.append({
                "type": "task_id_collision",
                "message": f"Duplicate task_id found: '{tid}' at indices {indices}",
                "task_ids": [tid],
            })

    # Check exact Question duplicates
    seen_questions: dict[str, list[str]] = {}
    for t in tasks:
        q_norm = " ".join(t.question.strip().lower().split())
        seen_questions.setdefault(q_norm, []).append(t.task_id)

    q_duplicates = {q: tids for q, tids in seen_questions.items() if len(tids) > 1}
    if q_duplicates:
        for q, tids in q_duplicates.items():
            violations.append({
                "type": "duplicate_question",
                "message": f"Duplicate question text across task IDs: {tids}",
                "task_ids": tids,
            })

    return violations

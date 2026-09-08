"""
dataset/cleaning/deduplicator.py
================================
Exact and fuzzy duplicate detection for GateOrchestra dataset tasks.

Supports:
  1. Exact duplicate detection using normalized hash fingerprinting
  2. Fuzzy duplicate detection using token-level Jaccard & char n-gram similarity
  3. Duplicate clustering and audit logging

Person 1 owns this file.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from shared.schemas import Task


def normalize_for_fingerprint(text: str) -> str:
    """Normalize text aggressively for fingerprinting.

    Lowercases, strips punctuation, collapses whitespace.
    """
    s = text.lower()
    s = re.sub(r"[^\w\s]", "", s)
    return " ".join(s.split())


def exact_fingerprint(text: str) -> str:
    """Compute MD5 hash fingerprint of aggressively normalized text."""
    normalized = normalize_for_fingerprint(text)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def _get_token_set(text: str) -> set[str]:
    """Tokenize normalized text into a word set."""
    norm = normalize_for_fingerprint(text)
    return set(norm.split())


def _get_char_ngrams(text: str, n: int = 3) -> set[str]:
    """Extract character n-grams from normalized text."""
    norm = normalize_for_fingerprint(text)
    if len(norm) < n:
        return {norm} if norm else set()
    return {norm[i : i + n] for i in range(len(norm) - n + 1)}


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def compute_question_similarity(q1: str, q2: str) -> float:
    """Compute hybrid fuzzy similarity score between two questions in [0.0, 1.0].

    Blends word-token Jaccard (60%) and character tri-gram Jaccard (40%).
    """
    if q1 == q2:
        return 1.0

    tokens1 = _get_token_set(q1)
    tokens2 = _get_token_set(q2)
    token_sim = jaccard_similarity(tokens1, tokens2)

    char_ngrams1 = _get_char_ngrams(q1, n=3)
    char_ngrams2 = _get_char_ngrams(q2, n=3)
    char_sim = jaccard_similarity(char_ngrams1, char_ngrams2)

    return round(0.60 * token_sim + 0.40 * char_sim, 4)


class FuzzyDeduplicator:
    """Deduplicates a collection of tasks using exact hashing and fuzzy similarity."""

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        self.similarity_threshold = similarity_threshold

    def deduplicate(
        self,
        tasks: list[Task],
    ) -> tuple[list[Task], list[dict[str, Any]], list[dict[str, Any]]]:
        """Find and remove exact and fuzzy duplicates.

        Keeps the first encountered task in each duplicate group.

        Args:
            tasks: List of Task objects.

        Returns:
            (unique_tasks, exact_duplicates, fuzzy_duplicates)
            - unique_tasks: Filtered list of unique Task objects.
            - exact_duplicates: List of exact duplicate records removed.
            - fuzzy_duplicates: List of fuzzy duplicate records removed (above threshold).
        """
        unique_tasks: list[Task] = []
        seen_fingerprints: dict[str, Task] = {}
        exact_duplicates: list[dict[str, Any]] = []
        fuzzy_duplicates: list[dict[str, Any]] = []

        for task in tasks:
            fp = exact_fingerprint(task.question)

            # 1. Exact duplicate check
            if fp in seen_fingerprints:
                orig_task = seen_fingerprints[fp]
                exact_duplicates.append(
                    {
                        "duplicate_task_id": task.task_id,
                        "original_task_id": orig_task.task_id,
                        "question": task.question,
                        "fingerprint": fp,
                    }
                )
                continue

            # 2. Fuzzy duplicate check against already accepted unique tasks
            is_fuzzy_dup = False
            for accepted in unique_tasks:
                sim = compute_question_similarity(task.question, accepted.question)
                if sim >= self.similarity_threshold:
                    fuzzy_duplicates.append(
                        {
                            "duplicate_task_id": task.task_id,
                            "original_task_id": accepted.task_id,
                            "similarity_score": sim,
                            "threshold": self.similarity_threshold,
                            "duplicate_question": task.question,
                            "original_question": accepted.question,
                        }
                    )
                    is_fuzzy_dup = True
                    break

            if not is_fuzzy_dup:
                seen_fingerprints[fp] = task
                unique_tasks.append(task)

        return unique_tasks, exact_duplicates, fuzzy_duplicates


def detect_duplicates(
    tasks: list[Task],
    similarity_threshold: float = 0.85,
) -> tuple[list[Task], dict[str, Any]]:
    """Convenience helper to deduplicate tasks and return clean summary dict.

    Args:
        tasks: Input task list.
        similarity_threshold: Threshold above which two questions are considered duplicates.

    Returns:
        (deduped_tasks, summary_dict)
    """
    deduplicator = FuzzyDeduplicator(similarity_threshold=similarity_threshold)
    unique_tasks, exact_dups, fuzzy_dups = deduplicator.deduplicate(tasks)

    summary = {
        "input_task_count": len(tasks),
        "unique_task_count": len(unique_tasks),
        "exact_duplicates_count": len(exact_dups),
        "fuzzy_duplicates_count": len(fuzzy_dups),
        "total_removed": len(exact_dups) + len(fuzzy_dups),
        "exact_duplicates": exact_dups,
        "fuzzy_duplicates": fuzzy_dups,
    }

    return unique_tasks, summary

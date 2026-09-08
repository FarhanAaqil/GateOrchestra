# dataset/cleaning/__init__.py
# Person 1 — Dataset cleaning package public API

from dataset.cleaning.cleaner import (
    canonicalize_answer,
    clean_dataset_records,
    clean_task,
    format_question,
    normalize_text,
)
from dataset.cleaning.deduplicator import (
    FuzzyDeduplicator,
    compute_question_similarity,
    detect_duplicates,
    exact_fingerprint,
)

__all__ = [
    "normalize_text",
    "canonicalize_answer",
    "format_question",
    "clean_task",
    "clean_dataset_records",
    "exact_fingerprint",
    "compute_question_similarity",
    "FuzzyDeduplicator",
    "detect_duplicates",
]

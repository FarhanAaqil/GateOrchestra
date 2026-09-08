"""
dataset/cleaning/cleaner.py
===========================
Text sanitization, unicode normalization, and answer canonicalization for GateOrchestra.

Core responsibilities:
  1. Unicode normalization (NFKC, smart quote/dash replacement, zero-width char stripping)
  2. Whitespace collapsing and invisible character cleaning
  3. Interrogative punctuation enforcement
  4. Answer string canonicalization (standardizing booleans, removing spurious punctuation)
  5. Audit log tracking of all transformations applied

Person 1 owns this file.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from shared.schemas import Task

# Zero-width / invisible unicode characters to strip
_ZERO_WIDTH_CHARS = re.compile(r"[\u200B\u200C\u200D\uFEFF\u200E\u200F\u202A-\u202E]")

# Smart quotes and dashes mapping
_CHAR_REPLACEMENTS: dict[str, str] = {
    "“": '"',
    "”": '"',
    "„": '"',
    "«": '"',
    "»": '"',
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‹": "'",
    "›": "'",
    "`": "'",
    "—": "-",
    "–": "-",
    "―": "-",
    "−": "-",
    "\u00a0": " ",  # Non-breaking space
    "\u202f": " ",  # Narrow no-break space
    "\u3000": " ",  # Ideographic space
}

# Regex pattern for smart characters to replace efficiently
_CHAR_REPLACE_PATTERN = re.compile("|".join(re.escape(k) for k in _CHAR_REPLACEMENTS.keys()))

# Interrogative keywords for question formatting
_INTERROGATIVE_STARTERS = (
    "who",
    "what",
    "when",
    "where",
    "which",
    "why",
    "how",
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "can",
    "could",
    "should",
    "would",
    "between",
    "if",
    "calculate",
    "find",
    "determine",
    "compare",
    "name",
    "list",
    "convert",
)


def normalize_text(
    text: str,
    *,
    unicode_norm: str = "NFKC",
    strip_zero_width: bool = True,
) -> str:
    """Sanitize and normalize text strings.

    Args:
        text: Input string.
        unicode_norm: Unicode normalization form ('NFKC', 'NFC', etc.).
        strip_zero_width: If True, remove invisible / zero-width characters.

    Returns:
        Cleaned, normalized string with collapsed whitespace.
    """
    if not text:
        return ""

    s = text

    # 1. Unicode normalization
    if unicode_norm:
        s = unicodedata.normalize(unicode_norm, s)

    # 2. Strip zero-width / control characters
    if strip_zero_width:
        s = _ZERO_WIDTH_CHARS.sub("", s)

    # 3. Replace smart quotes, apostrophes, dashes, and NBSP
    s = _CHAR_REPLACE_PATTERN.sub(lambda m: _CHAR_REPLACEMENTS[m.group(0)], s)

    # 4. Collapse internal whitespace and strip leading/trailing
    s = " ".join(s.split())

    return s


def canonicalize_answer(answer: str) -> str:
    """Standardize answer formatting without changing semantic meaning.

    Rules:
      - Collapses whitespace and normalizes unicode.
      - Strips surrounding quotes if whole string is quoted (e.g., '"Paris"' -> 'Paris').
      - Strips trailing period in short non-sentence answers (e.g., 'Paris.' -> 'Paris', '42.' -> '42').
      - Standardizes common boolean representations ('yes' -> 'Yes', 'no' -> 'No', 'true' -> 'True', 'false' -> 'False').

    Args:
        answer: Raw answer string.

    Returns:
        Canonicalized answer string.
    """
    if not answer:
        return ""

    s = normalize_text(answer)

    # Strip surrounding quotation marks if symmetric
    if (s.startswith('"') and s.endswith('"') and len(s) >= 2) or (
        s.startswith("'") and s.endswith("'") and len(s) >= 2
    ):
        s = s[1:-1].strip()

    # Canonicalize booleans
    s_lower = s.lower()
    if s_lower in ("true", "false", "yes", "no"):
        s = s_lower.capitalize()
        return s

    # Strip trailing period on short entities / numbers / dates (e.g. "Washington, D.C." keeps D.C., but "Paris." becomes "Paris")
    if s.endswith(".") and not s.endswith("..") and len(s) < 50:
        # Check if it looks like an acronym or abbreviation ending like "D.C." or "U.S."
        if not re.search(r"\b[A-Z]\.[A-Z]\.$", s):
            s = s[:-1].strip()

    return s


def format_question(question: str, *, enforce_question_mark: bool = True) -> str:
    """Format and clean a question string.

    Args:
        question: Raw question string.
        enforce_question_mark: If True, appends '?' if an interrogative question lacks ending punctuation.

    Returns:
        Cleaned question string with proper punctuation.
    """
    if not question:
        return ""

    s = normalize_text(question)

    if enforce_question_mark and s:
        # If question does not end in standard terminal punctuation
        if not s.endswith(("?", ".", "!")):
            first_word = s.split()[0].strip("\"'").lower() if s.split() else ""
            if first_word in _INTERROGATIVE_STARTERS:
                s = s + "?"

    return s


def clean_task(
    task: Task,
    *,
    config: dict[str, Any] | None = None,
) -> tuple[Task, list[str]]:
    """Clean a single Task object and track modifications.

    Args:
        task: Input Task instance.
        config: Optional cleaning configuration dictionary.

    Returns:
        (cleaned_task, list_of_modifications)
    """
    cfg = config or {}
    unicode_norm = cfg.get("unicode_normalization", "NFKC")
    strip_zw = cfg.get("strip_zero_width_chars", True)
    canon_answers = cfg.get("canonicalize_answers", True)
    enforce_qm = cfg.get("enforce_question_mark", True)

    mods: list[str] = []

    # Clean question
    orig_q = task.question
    cleaned_q = format_question(
        normalize_text(orig_q, unicode_norm=unicode_norm, strip_zero_width=strip_zw),
        enforce_question_mark=enforce_qm,
    )
    if cleaned_q != orig_q:
        mods.append("question_normalized")

    # Clean context if present
    orig_ctx = task.context
    cleaned_ctx: str | None = None
    if orig_ctx is not None:
        cleaned_ctx = normalize_text(orig_ctx, unicode_norm=unicode_norm, strip_zero_width=strip_zw)
        if cleaned_ctx != orig_ctx:
            mods.append("context_normalized")

    # Clean ground_truth
    orig_gt = task.ground_truth or ""
    if canon_answers:
        cleaned_gt = canonicalize_answer(orig_gt)
    else:
        cleaned_gt = normalize_text(orig_gt, unicode_norm=unicode_norm, strip_zero_width=strip_zw)

    if cleaned_gt != orig_gt:
        mods.append("ground_truth_canonicalized")

    # Construct new Task (Task is frozen/immutable)
    cleaned_task = Task(
        task_id=task.task_id,
        question=cleaned_q,
        context=cleaned_ctx,
        depth_score=task.depth_score,
        parallel_score=task.parallel_score,
        ground_truth=cleaned_gt,
        source_dataset=task.source_dataset,
    )

    return cleaned_task, mods


def clean_dataset_records(
    tasks: list[Task],
    *,
    config: dict[str, Any] | None = None,
) -> tuple[list[Task], dict[str, Any]]:
    """Clean all tasks in a dataset and compile an audit summary.

    Args:
        tasks: List of Task objects to clean.
        config: Optional cleaning configuration dictionary.

    Returns:
        (cleaned_tasks, cleaning_summary_dict)
    """
    cleaned_tasks: list[Task] = []
    mod_counts: dict[str, int] = {}
    task_audit_log: list[dict[str, Any]] = []

    for task in tasks:
        cleaned, mods = clean_task(task, config=config)
        cleaned_tasks.append(cleaned)

        if mods:
            for mod in mods:
                mod_counts[mod] = mod_counts.get(mod, 0) + 1
            task_audit_log.append(
                {
                    "task_id": task.task_id,
                    "modifications": mods,
                    "before": {
                        "question": task.question,
                        "ground_truth": task.ground_truth,
                    },
                    "after": {
                        "question": cleaned.question,
                        "ground_truth": cleaned.ground_truth,
                    },
                }
            )

    summary = {
        "total_input_tasks": len(tasks),
        "total_output_tasks": len(cleaned_tasks),
        "tasks_modified_count": len(task_audit_log),
        "modifications_by_type": mod_counts,
        "task_audit_log": task_audit_log,
    }

    return cleaned_tasks, summary

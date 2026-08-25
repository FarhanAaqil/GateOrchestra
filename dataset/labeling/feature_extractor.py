"""
dataset/labeling/feature_extractor.py
=======================================
Person 1's labeling feature extractor.

Extracts lightweight, interpretable structural signals from a Task dict
(or Task object) to feed into the depth and parallel labelers.

This is SEPARATE from gate/feature_extractor.py (Person 3), which extracts
features at inference time. This module runs at dataset construction time.

Signals extracted:
  - question_word_count   : number of words in question
  - entity_count          : named entity / capitalized phrase count (regex)
  - clause_count          : subordinate clause count (regex)
  - hop_count             : hint from task pool, or heuristic fallback
  - sub_question_count    : number of sub-questions (count of '?', 'and ... what', etc.)
  - conjunction_count     : list-like conjunctions ('and', 'or', 'both', 'also')
  - list_count            : comma-separated list detection
  - has_context           : whether a non-empty context string is present
  - context_word_count    : word count of context if present

All signals are stored alongside final labels for full transparency.

Person 1 owns this file.
"""

from __future__ import annotations

import re


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def extract_labeling_features(
    question: str,
    context: str | None = None,
    hop_hint: int = 1,
) -> dict:
    """Extract structural features for depth and parallel labeling.

    Args:
        question:   The task question text.
        context:    Optional supporting passage.
        hop_hint:   Number of reasoning hops from task pool metadata.

    Returns:
        dict of feature name → value. All values are primitive types
        (int, float, bool) for easy serialization.
    """
    q = question.strip()
    ctx = (context or "").strip()

    entity_count = _count_entities(q)
    clause_count = _count_clauses(q)
    sub_question_count = _count_sub_questions(q)
    conjunction_count = _count_conjunctions(q)
    list_count = _count_list_items(q)
    question_word_count = len(q.split())
    has_context = bool(ctx)
    context_word_count = len(ctx.split()) if ctx else 0

    return {
        "question_word_count": question_word_count,
        "entity_count": entity_count,
        "clause_count": clause_count,
        "hop_count": hop_hint,
        "sub_question_count": sub_question_count,
        "conjunction_count": conjunction_count,
        "list_count": list_count,
        "has_context": has_context,
        "context_word_count": context_word_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _count_entities(text: str) -> int:
    """Count capitalized noun phrase spans as proxy for named entities.

    Uses regex: two or more consecutive Title Case words (avoids
    sentence-initial false positives by requiring multi-word spans).
    Single capitalized words at non-sentence-start are also counted.
    """
    # Multi-word capitalized spans (e.g., "New York", "Marie Curie")
    multi = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text)
    # Single capitalized words not at start of sentence (rough proxy)
    single_mid = re.findall(r"(?<=[.!?]\s)([A-Z][a-z]+)\b|[a-z]\s+([A-Z][a-z]+)\b", text)
    single_count = sum(1 for m in single_mid if any(g for g in m))
    return len(multi) + single_count


def _count_clauses(text: str) -> int:
    """Count subordinate clause indicators."""
    clause_keywords = (
        r"\b(that|which|when|where|who|whom|whose|because|although|"
        r"since|if|while|whereas|though|unless|until|after|before|"
        r"as long as|so that)\b"
    )
    return len(re.findall(clause_keywords, text, flags=re.IGNORECASE))


def _count_sub_questions(text: str) -> int:
    """Count sub-questions using question marks and question-introducing phrases."""
    # Direct question marks (beyond the final one)
    q_marks = len(re.findall(r"\?", text))
    # Embedded question phrases: "what X", "which X", "who X" in mid-sentence
    embedded = re.findall(
        r",\s*(what|which|who|where|when|how|why)\b",
        text,
        flags=re.IGNORECASE,
    )
    return max(q_marks - 1, 0) + len(embedded)


def _count_conjunctions(text: str) -> int:
    """Count list-forming conjunctions."""
    pattern = r"\b(and|or|both|also|as well as|along with|together with)\b"
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def _count_list_items(text: str) -> int:
    """Detect comma-separated lists as proxy for parallel structure.

    A list is detected when there are 2+ commas in a sequence without
    a period separating them, or explicit enumeration words.
    """
    # Count comma clusters (3+ items in a list = 2+ commas)
    comma_count = text.count(",")
    # Enumeration markers
    enum_words = re.findall(r"\b(first|second|third|finally|lastly|respectively)\b",
                             text, flags=re.IGNORECASE)
    return comma_count + len(enum_words)

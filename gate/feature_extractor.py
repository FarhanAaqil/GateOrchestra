"""
gate/feature_extractor.py
=========================
Extracts GateFeatures from a (Task, ProbeResult) pair.

Design principles:
  - ZERO LLM calls — all features computed locally
  - Fast: target < 50ms per task
  - Fallback: regex mode if spaCy is unavailable

Person 3 owns this file.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from shared.config import SPACY_MODEL, USE_SPACY
from shared.schemas import GateFeatures, ProbeResult, Task

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# spaCy lazy loader
# ─────────────────────────────────────────────────────────────────────────────

_nlp = None  # Lazy-loaded to avoid slow import at module level


def _get_nlp():
    """Load the spaCy model once and cache it."""
    global _nlp
    if _nlp is None:
        try:
            import spacy  # noqa: PLC0415

            _nlp = spacy.load(SPACY_MODEL)
            logger.info(f"Loaded spaCy model: {SPACY_MODEL}")
        except (ImportError, OSError) as e:
            logger.warning(
                f"spaCy unavailable ({e}). Falling back to regex-based feature extraction."
            )
            _nlp = "unavailable"
    return _nlp if _nlp != "unavailable" else None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def extract_features(
    task: Task,
    probe: ProbeResult,
    use_spacy: Optional[bool] = None,
) -> GateFeatures:
    """Extract GateFeatures from a Task and its ProbeResult.

    Args:
        task:      The task being evaluated.
        probe:     The CoT-SC probe result for this task.
        use_spacy: Override config.USE_SPACY if provided.

    Returns:
        GateFeatures ready to be fed into any gate classifier.
    """
    if task.task_id != probe.task_id:
        raise ValueError(
            f"task.task_id ({task.task_id!r}) != probe.task_id ({probe.task_id!r})"
        )

    _use_spacy = USE_SPACY if use_spacy is None else use_spacy

    text = task.question
    context_text = task.context or ""
    full_text = f"{text} {context_text}".strip()

    if _use_spacy:
        nlp = _get_nlp()
        if nlp is not None:
            entity_count, clause_count = _spacy_features(full_text, nlp)
        else:
            entity_count, clause_count = _regex_features(text)
    else:
        entity_count, clause_count = _regex_features(text)

    question_word_count = len(text.split())
    has_context = bool(task.context and task.context.strip())

    estimated_depth = _estimate_depth(entity_count, clause_count, question_word_count)
    estimated_parallel = _estimate_parallel(text)

    return GateFeatures(
        task_id=task.task_id,
        consistency_score=probe.consistency_score,
        probe_tokens=probe.tokens_used,
        question_word_count=question_word_count,
        entity_count=entity_count,
        clause_count=clause_count,
        has_context=has_context,
        estimated_depth=estimated_depth,
        estimated_parallel=estimated_parallel,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _spacy_features(text: str, nlp) -> tuple[int, int]:
    """Extract entity and clause counts using spaCy."""
    doc = nlp(text)
    entity_count = len(doc.ents)
    # Subordinate clauses via dependency parse (advcl, relcl, ccomp, acl)
    clause_deps = {"advcl", "relcl", "ccomp", "acl", "xcomp"}
    clause_count = sum(1 for token in doc if token.dep_ in clause_deps)
    return entity_count, clause_count


def _regex_features(text: str) -> tuple[int, int]:
    """Regex fallback for entity and clause extraction (no spaCy).

    Approximations:
      - entity_count: capitalized multi-word spans (Title Case phrases)
      - clause_count: count of clause-introducing tokens (that, which, when, etc.)
    """
    # Capitalized noun phrases (rough proxy for named entities)
    entity_matches = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
    entity_count = len(entity_matches)

    # Subordinate clause keywords
    clause_keywords = r"\b(that|which|when|where|who|whom|whose|because|although|since|if|while)\b"
    clause_count = len(re.findall(clause_keywords, text, flags=re.IGNORECASE))

    return entity_count, clause_count


def _estimate_depth(entity_count: int, clause_count: int, word_count: int) -> float:
    """Heuristic depth estimate normalized to roughly [0, 5] range.

    Formula: weighted combination of clause count and entity density.
    This is a proxy — ground-truth depth labels come from Person 1.
    """
    entity_density = entity_count / max(word_count, 1)
    return round(min(clause_count * 0.8 + entity_density * 10, 5.0), 2)


def _estimate_parallel(text: str) -> float:
    """Heuristic parallelism estimate (proxy for Person 1's parallel_score).

    Counts sub-questions or 'and'-joined clauses as a proxy for parallelism.
    """
    # Count question-like sub-clauses
    sub_questions = re.findall(r"\?", text)
    # Count list-like conjunctions
    conjunctions = re.findall(r"\b(and|or|also|both|as well as)\b", text, re.IGNORECASE)
    score = len(sub_questions) * 1.5 + len(conjunctions) * 0.5
    return round(min(score, 4.0), 2)

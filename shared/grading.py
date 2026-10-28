"""
shared/grading.py
=================
The ONE correctness function used everywhere in GateOrchestra.

Every grader in the repo (pipeline, baselines, evaluation metrics, analysis
tools) delegates to :func:`is_correct`, and the probe's majority vote uses
:func:`answers_equivalent`, so "correct" and "agrees with" always mean the
same thing.

Rules
-----
1. The final answer is extracted from the raw response ("Final Answer: ...",
   "The answer is ...", ``\\boxed{}``, ``####``, "Therefore ... is X").
2. Each acceptable answer (the ground truth plus any ``answer_aliases``) is
   tried in turn:

   * Numeric candidates ("$5.00", "30 square centimeters", "4.5 × 10^-4",
     "1/4", "25%") are compared by value.  The predicted number is rounded to
     the candidate's precision, so "5" matches "$5.00" but "500" does not, and
     "15" never matches "5".
   * Text candidates match when the normalised prediction equals the
     candidate, or *starts with* it as whole words and does not continue
     with a negation ("Paris, France" matches "Paris"; "Unknown" does not
     match "no"; "Canada is near the United States" does not match
     "United States").
   * A candidate written as ``"part one && part two"`` requires every part to
     appear as a whole phrase in the prediction (for two-part questions).
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable, Sequence

_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

# Number token: optional sign/"minus", digits with thousands separators,
# optional decimals, optional fraction or scientific-notation suffix.
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z\d.])"  # not part of a unit or identifier ("cm2", "H2O")
    r"(?P<sign>-|−|\bminus\s+)?"
    r"(?<![A-Za-z])(?P<int>\d{1,3}(?:,\d{3})+|\d+)"
    r"(?P<dec>\.\d+)?"
    r"(?:\s*/\s*(?P<den>\d+(?:\.\d+)?))?"
    r"(?:\s*(?:[x×*]|times)\s*10\s*\^\s*(?P<exp>[-−]?\d+)|e(?P<eexp>[-−]?\d+))?",
    flags=re.IGNORECASE,
)

_NEGATION_RE = re.compile(
    r"^(?:is|was|are|were|means|equals)?\s*(?:not|wrong|incorrect|false)\b",
)

ALL_OF_SEPARATOR = "&&"


# Answer extraction


def extract_final_answer(text: str) -> str:
    """Extract an explicitly stated final answer from a raw model response."""
    if not text or not text.strip():
        return ""

    cleaned = re.sub(r"<think>.*?</think>", "", text.strip(), flags=re.DOTALL).strip()
    if not cleaned:
        return ""

    marker_patterns = (
        r"(?:final\s+answer|the\s+answer\s+is|answer)\s*[:=]?\s*([^\n\r]+)",
        r"\\boxed\{([^}]+)\}",
        r"####\s*([^\n\r]+)",
    )
    for pattern in marker_patterns:
        matches = list(re.finditer(pattern, cleaned, flags=re.IGNORECASE))
        if matches:
            return matches[-1].group(1).strip().strip("`*_\"' ")

    conclusion = re.search(
        r"(?:therefore|thus|so),?\s+.*?\b(?:is|are|equals)\s+([^!?\n]+?)(?:\.\s|\.$|$)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if conclusion:
        return conclusion.group(1).strip()

    return cleaned


# Normalisation


def normalize_text(text: str) -> str:
    """Lower-case, strip accents and punctuation, drop a leading article."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower().strip()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = " ".join(normalized.split())
    return re.sub(r"^(?:the|a|an)\s+", "", normalized)


def _replace_number_words(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return str(_NUMBER_WORDS[match.group(0).lower()])

    pattern = r"\b(?:" + "|".join(_NUMBER_WORDS) + r")\b"
    return re.sub(pattern, repl, text, flags=re.IGNORECASE)


def parse_numbers(text: str) -> list[tuple[float, int]]:
    """Return ``(value, decimals)`` for every number in ``text``, in order.

    ``decimals`` is the precision the number was written with, used to
    round a prediction before comparing it.
    """
    if not text:
        return []
    prepared = _replace_number_words(unicodedata.normalize("NFKC", text))
    numbers: list[tuple[float, int]] = []
    for match in _NUMBER_RE.finditer(prepared):
        integer = match.group("int").replace(",", "")
        decimals_str = match.group("dec") or ""
        value = float(integer + decimals_str)
        decimals = max(len(decimals_str) - 1, 0)

        if match.group("den"):
            denominator = float(match.group("den"))
            if denominator == 0:
                continue
            value /= denominator
            decimals = 6
        exponent = match.group("exp") or match.group("eexp")
        if exponent:
            exp = int(exponent.replace("−", "-"))
            value *= 10.0**exp
            decimals = max(decimals - exp, 0)
        if match.group("sign"):
            value = -value
        numbers.append((value, decimals))
    return numbers


# Matching


def _numbers_equal(predicted: float, candidate: float, decimals: int) -> bool:
    return math.isclose(round(predicted, decimals), round(candidate, decimals), abs_tol=1e-9)


def _numeric_match(predicted: str, candidate: str) -> bool:
    candidate_numbers = parse_numbers(candidate)
    predicted_numbers = parse_numbers(predicted)
    if not predicted_numbers:
        return False

    if len(candidate_numbers) == 1:
        value, decimals = candidate_numbers[0]
        # The answer is the first number stated, or the result after the last "=".
        options = [predicted_numbers[0][0]]
        if "=" in predicted:
            tail_numbers = parse_numbers(predicted.rsplit("=", 1)[1])
            if tail_numbers:
                options.append(tail_numbers[0][0])
        return any(_numbers_equal(option, value, decimals) for option in options)

    # Multi-number candidates ("1 hour 43 minutes") need every number, in order.
    position = 0
    for value, decimals in candidate_numbers:
        while position < len(predicted_numbers) and not _numbers_equal(
            predicted_numbers[position][0], value, decimals
        ):
            position += 1
        if position == len(predicted_numbers):
            return False
        position += 1
    return True


def _contains_phrase(haystack: str, phrase: str) -> bool:
    return bool(phrase) and re.search(rf"(?:^|\s){re.escape(phrase)}(?:\s|$)", haystack) is not None


def _text_match(predicted: str, candidate: str) -> bool:
    norm_pred = normalize_text(predicted)
    norm_cand = normalize_text(candidate)
    if not norm_pred or not norm_cand:
        return False
    if norm_pred == norm_cand:
        return True
    if not norm_pred.startswith(norm_cand + " "):
        return False
    remainder = norm_pred[len(norm_cand) :].strip()
    return _NEGATION_RE.match(remainder) is None


def _all_of_match(predicted: str, candidate: str) -> bool:
    norm_pred = normalize_text(predicted)
    parts = [normalize_text(part) for part in candidate.split(ALL_OF_SEPARATOR)]
    return all(_contains_phrase(norm_pred, part) for part in parts if part)


def matches_candidate(predicted_answer: str, candidate: str) -> bool:
    """Return True if an already-extracted answer matches one acceptable answer."""
    if not predicted_answer or not candidate or not candidate.strip():
        return False
    if ALL_OF_SEPARATOR in candidate:
        return _all_of_match(predicted_answer, candidate)
    if parse_numbers(candidate):
        return _numeric_match(predicted_answer, candidate)
    return _text_match(predicted_answer, candidate)


def is_correct(
    predicted: str,
    ground_truth: str,
    aliases: Iterable[str] = (),
) -> bool:
    """Canonical correctness check used by every evaluation path.

    Args:
        predicted:    Raw model output or extracted answer.
        ground_truth: The task's gold answer.
        aliases:      Other acceptable answers (``Task.answer_aliases``).
    """
    if not predicted or not ground_truth:
        return False
    answer = extract_final_answer(predicted)
    if not answer:
        return False
    return any(matches_candidate(answer, candidate) for candidate in (ground_truth, *aliases))


def grade_task(predicted: str, task: object) -> bool | None:
    """Grade against a Task's ground truth and aliases; None if it has no ground truth."""
    ground_truth = getattr(task, "ground_truth", None)
    if ground_truth is None:
        return None
    aliases: Sequence[str] = getattr(task, "answer_aliases", ()) or ()
    return is_correct(predicted, ground_truth, aliases)


def answers_equivalent(first: str, second: str) -> bool:
    """Whether two extracted answers say the same thing (used for majority voting)."""
    if not first or not second:
        return False
    if normalize_text(first) == normalize_text(second):
        return True
    first_numbers = parse_numbers(first)
    second_numbers = parse_numbers(second)
    if first_numbers or second_numbers:
        if not first_numbers or not second_numbers:
            return False
        return _numeric_match(first, second) and _numeric_match(second, first)
    return _text_match(first, second) or _text_match(second, first)

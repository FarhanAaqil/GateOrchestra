"""
tests/test_grading.py
=====================
The canonical grader (shared/grading.py), including the cases the project
review found graded wrongly by the old exact-match and substring graders.
"""

from __future__ import annotations

import pytest

from evaluation.metrics import evaluate_answer
from shared.data_loader import exact_match, load_all
from shared.grading import answers_equivalent, grade_task, is_correct, parse_numbers
from shared.schemas import Task


@pytest.mark.parametrize(
    ("predicted", "ground_truth", "expected"),
    [
        # Review: punctuation stripping graded "$500" correct for "$5.00".
        ("$500", "$5.00", False),
        ("$5", "$5.00", True),
        ("Final Answer: $5.00", "$5.00", True),
        # Review: substring matching graded these correct.
        ("Unknown", "no", False),
        ("15", "5", False),
        ("Final Answer: 523.599 cubic cm", "approximately 523.6 cubic centimeters", True),
        ("$14,500", "$14,450", False),
        ("Final Answer: 4.5 x 10^-4", "4.5 × 10^-4", True),
        ("0.25", "1/4", True),
        ("-268.9 °C", "-268.9 degrees Celsius", True),
        ("268.9", "-268.9 degrees Celsius", False),
        ("6 × 10 / 2 = 30", "30 square centimeters", True),
        ("1 hour", "1 hour 43 minutes", False),
        ("Luiz Inacio Lula da Silva", "Luiz Inácio Lula da Silva", True),
        ("Paris, France", "Paris", True),
        ("Canada is near the United States", "United States", False),
        ("United States is not correct; Canada is", "United States", False),
    ],
)
def test_is_correct(predicted: str, ground_truth: str, expected: bool) -> None:
    assert is_correct(predicted, ground_truth) is expected


def test_all_of_alias_requires_every_part() -> None:
    aliases = ["Bush && Republican"]
    assert is_correct(
        "George H.W. Bush, a Republican", "George H.W. Bush, Republican Party", aliases
    )
    assert not is_correct("George H.W. Bush", "George H.W. Bush, Republican Party", aliases)


def test_every_grader_entry_point_agrees() -> None:
    cases = [("$500", "$5.00"), ("15", "5"), ("Unknown", "no"), ("Paris.", "Paris")]
    for predicted, ground_truth in cases:
        expected = is_correct(predicted, ground_truth)
        assert exact_match(predicted, ground_truth) is expected
        assert evaluate_answer(predicted, ground_truth) is expected


def test_grade_task_uses_aliases_and_handles_missing_ground_truth() -> None:
    task = Task(
        task_id="t1",
        question="Which is closer to the Sun: Mars or Venus?",
        ground_truth="Venus is closer to the Sun than Mars",
        answer_aliases=("Venus",),
    )
    assert grade_task("Final Answer: Venus", task) is True
    assert grade_task("Final Answer: Mars", task) is False
    assert grade_task("Venus", Task(task_id="t2", question="?")) is None


def test_units_with_digits_are_not_numbers() -> None:
    assert parse_numbers("30 cm2") == [(30.0, 0)]
    assert parse_numbers("H2O") == []


def test_equivalence_keeps_different_numbers_apart() -> None:
    assert not answers_equivalent("30 square centimeters", "60 square centimeters")
    assert answers_equivalent("30", "30 cm2")
    assert answers_equivalent("Paris", "Paris, France")


def test_every_dataset_answer_and_alias_grades_itself_correct() -> None:
    tasks = [task for split in load_all().values() for task in split]
    for task in tasks:
        assert task.ground_truth is not None
        for candidate in (task.ground_truth, *task.answer_aliases):
            assert is_correct(candidate, task.ground_truth, task.answer_aliases), task.task_id

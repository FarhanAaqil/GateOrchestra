"""The chat evaluation: its question set, grading and analysis (no model calls)."""

from __future__ import annotations

import contextlib
import io
import re

import pytest

from scripts import chat_eval


def test_the_question_set_is_complete_and_its_code_answers_are_right() -> None:
    questions = chat_eval.load_questions()
    assert len(questions) == 80 and len({q["id"] for q in questions}) == 80
    assert {q["category"] for q in questions} == {"factual", "arithmetic", "multistep", "code"}
    assert all(q["question"].strip() and q["answer"].strip() for q in questions)
    for q in (q for q in questions if q["category"] == "code"):
        code = re.search(r"```python\n(.*?)```", q["question"], re.S).group(1)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(code, {})  # our own fixed snippets
        assert out.getvalue().strip() == q["answer"], q["id"]


@pytest.mark.parametrize(
    ("answer", "right"),
    [
        ("Canberra is the capital.\n\nAnswer: Canberra", True),
        ("**Answer:** Canberra", True),
        ("It is not Sydney.\nAnswer: Sydney", False),
        ("No final line at all, but Canberra is mentioned", False),
    ],
)
def test_grading_reads_the_final_answer_line(answer: str, right: bool) -> None:
    assert chat_eval.grade(answer, {"answer": "Canberra", "aliases": []}) is right


def row(
    qid: str, mode: str, correct: bool, tokens: int, learned: str = "STOP", consistency: float = 1.0
) -> dict:
    return {
        "id": qid,
        "category": "factual",
        "mode": mode,
        "correct": correct,
        "total_tokens": tokens,
        "latency_ms": 1000,
        "consistency": consistency,
        "verdicts": {"learned": learned, "rule_based": "STOP"},
        "features": dict.fromkeys(chat_eval.FEATURES, 0),
    }


def test_analysis_scores_every_route_from_two_runs() -> None:
    rows = []
    for i in range(20):
        rows.append(
            row(
                f"q{i}",
                "always_stop",
                correct=i > 1,
                tokens=500,
                learned="ESCALATE" if i <= 1 else "STOP",
                consistency=2 / 3 if i <= 1 else 1.0,
            )
        )
        rows.append(row(f"q{i}", "always_escalate", correct=True, tokens=2500))
    rows.append(
        {"id": "q99", "mode": "always_stop", "category": "factual", "error": {"code": "needs_key"}}
    )
    summary = chat_eval.analyze(rows)
    m = summary["methods"]
    assert summary["paired"] == 20 and summary["failed_turns"] == 1
    assert (
        m["Always answer directly"]["accuracy"] == 90.0
        and m["Always research"]["accuracy"] == 100.0
    )
    assert m["Learned gate (benchmark-trained)"]["accuracy"] == 100.0
    assert m["Learned gate (benchmark-trained)"]["avg_tokens"] == pytest.approx(700)
    assert m["Threshold t=1 (any disagreement)"]["escalation_rate"] == 10.0  # 2 of 20
    assert m["Oracle (research only when it fixes the answer)"]["accuracy"] == 100.0
    # The learned gate matches the best accuracy at the lowest cost, so it is the recommendation.
    assert summary["recommended_default"] == "Learned gate (benchmark-trained)"
    assert summary["research_vs_direct"]["research_only_right"] == 2
    assert summary["chat_gate"]["available"] is True and summary["chat_gate"]["positives"] == 2
    report = chat_eval.write_report(summary)
    assert "| Always research | 100.0% |" in report and "Recommended default" in report


def test_only_completed_turns_count_as_done(tmp_path) -> None:
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        '{"id": "a", "mode": "always_stop", "correct": true}\n{"id": "b", "mode": "always_stop", "error": {"code": "x"}}\n',
        encoding="utf-8",
    )
    assert chat_eval.done_keys(raw) == {("a", "always_stop")}

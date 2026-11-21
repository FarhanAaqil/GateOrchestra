"""
Second human review of the 64 tasks added to the test split
===========================================================

The 64 tasks added on 2026-11-10 were written and fact-checked by the AI
assistant. No independent person has read them yet. This script gives a
teammate everything they need and records what they found:

    python scripts/second_review.py sheet
        writes reports/dataset_quality/second_review_sheet.csv: one row per task with the
        question, context, gold answer and accepted aliases, plus empty
        columns: verdict (ok / wrong / ambiguous), correct_answer, note.

    python scripts/second_review.py import --reviewer "Name" [--file PATH]
        checks that every row has a verdict and writes a "second_review" block
        into dataset/masbench_mini/REVIEW.json (who, when, counts, and every
        task marked wrong or ambiguous with the reviewer's note).

Recording a review never edits a task. If the reviewer marks gold answers
wrong, the recorded predictions can be re-graded against their corrections
before anything is changed, and the report must say so.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.data_loader import load_answer_aliases, load_split  # noqa: E402

REVIEW = ROOT / "dataset" / "masbench_mini" / "REVIEW.json"
SHEET = ROOT / "reports" / "dataset_quality" / "second_review_sheet.csv"
VERDICTS = {"ok", "wrong", "ambiguous"}
COLUMNS = [
    "task_id",
    "question",
    "context",
    "gold_answer",
    "accepted_aliases",
    "verdict",
    "correct_answer",
    "note",
]


def tasks_to_review(review_path: Path = REVIEW) -> list[str]:
    record = json.loads(review_path.read_text(encoding="utf-8"))
    return list(record["assistant_reviewed"]["task_ids"])


def write_sheet(out: Path = SHEET, review_path: Path = REVIEW) -> int:
    wanted = set(tasks_to_review(review_path))
    aliases = load_answer_aliases()
    tasks = [t for t in load_split("test") if t.task_id in wanted]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for task in sorted(tasks, key=lambda t: t.task_id):
            writer.writerow(
                {
                    "task_id": task.task_id,
                    "question": task.question,
                    "context": task.context or "",
                    "gold_answer": task.ground_truth,
                    "accepted_aliases": " | ".join(aliases.get(task.task_id, [])),
                    "verdict": "",
                    "correct_answer": "",
                    "note": "",
                }
            )
    return len(tasks)


def read_verdicts(path: Path, expected: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {r["task_id"].strip(): r for r in rows}
    missing = [t for t in expected if t not in by_id]
    if missing:
        raise ValueError(f"The sheet is missing {len(missing)} task(s): {', '.join(missing[:5])}")
    problems = []
    for task_id in expected:
        verdict = by_id[task_id].get("verdict", "").strip().lower()
        if verdict not in VERDICTS:
            problems.append(f"{task_id}: verdict must be ok, wrong or ambiguous (got {verdict!r})")
        elif verdict == "wrong" and not by_id[task_id].get("correct_answer", "").strip():
            problems.append(f"{task_id}: marked wrong, so correct_answer is needed")
    if problems:
        raise ValueError("The sheet is not finished:\n  " + "\n  ".join(problems))
    return [by_id[t] for t in expected]


def record_review(
    reviewer: str,
    rows: list[dict[str, str]],
    review_path: Path = REVIEW,
    today: str | None = None,
) -> dict[str, Any]:
    if not reviewer.strip():
        raise ValueError("Give the reviewer's name.")
    record = json.loads(review_path.read_text(encoding="utf-8"))
    if reviewer.strip().lower().startswith("claude"):
        raise ValueError("The second review must be by a person, not the assistant.")
    flagged = [
        {
            "task_id": r["task_id"].strip(),
            "verdict": r["verdict"].strip().lower(),
            "correct_answer": r.get("correct_answer", "").strip(),
            "note": r.get("note", "").strip(),
        }
        for r in rows
        if r["verdict"].strip().lower() != "ok"
    ]
    record["second_review"] = {
        "reviewed_by": reviewer.strip(),
        "reviewed_on": today or dt.date.today().isoformat(),
        "task_ids": [r["task_id"].strip() for r in rows],
        "counts": {
            v: sum(r["verdict"].strip().lower() == v for r in rows) for v in sorted(VERDICTS)
        },
        "flagged": flagged,
    }
    review_path.write_text(
        json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return record["second_review"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="step", required=True)
    sub.add_parser("sheet")
    imp = sub.add_parser("import")
    imp.add_argument("--reviewer", required=True)
    imp.add_argument("--file", type=Path, default=SHEET)
    args = parser.parse_args(argv)
    if args.step == "sheet":
        n = write_sheet()
        print(
            f"wrote {n} tasks to {SHEET.relative_to(ROOT)}; fill in verdict, correct_answer and note"
        )
        return 0
    try:
        rows = read_verdicts(args.file, tasks_to_review())
        summary = record_review(args.reviewer, rows)
    except ValueError as err:
        print(err, file=sys.stderr)
        return 1
    print(f"recorded {len(rows)} verdicts by {summary['reviewed_by']}: {summary['counts']}")
    for item in summary["flagged"]:
        print(
            f"  {item['task_id']}: {item['verdict']} -> {item['correct_answer'] or '-'} ({item['note']})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
scripts/expand_test_split.py
============================
Append the 64 extra test tasks (dataset/generation/test_expansion.py) to every
dataset artifact, taking the test split from 32 to 96 tasks.

The tasks are only appended to the test split. The splitter is NOT rerun,
because that would reshuffle train, validation and test and invalidate the gate
and router calibrated on the existing train/validation tasks.

Patched, idempotently:
  dataset/raw/tasks_raw.jsonl
  dataset/processed/tasks_cleaned.jsonl
  dataset/masbench_mini/test/test.jsonl
  exports/dataset/test.csv
  dataset/masbench_mini/answer_aliases.json

Refuses to run if a new task repeats an existing question, or if any gold
answer or alias does not grade as correct under shared/grading.py.

    python scripts/expand_test_split.py
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset.generation.test_expansion import build_aliases, build_new_tasks  # noqa: E402
from shared.grading import is_correct  # noqa: E402

SPLITS = ("train", "val", "test")
MASBENCH = ROOT / "dataset" / "masbench_mini"
CSV_FIELDS = (
    "task_id",
    "question",
    "context",
    "ground_truth",
    "depth_score",
    "parallel_score",
    "source_dataset",
)
NEAR_DUPLICATE = 0.8


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(a: str, b: str) -> float:
    x, y = _words(a), _words(b)
    return len(x & y) / len(x | y) if x | y else 0.0


_NUMBER_WORDS = {
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
}


def _supported_by_context(gold: str, context: str) -> bool:
    """The gold is stated in the context, or is a number that the context's numbers explain."""
    text = context.lower()
    if gold.lower() in text:
        return True
    numbers = [int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", context)]
    wanted = re.fullmatch(r"(\d[\d,]*)(?: years)?", gold.strip())
    if wanted:
        value = int(wanted.group(1).replace(",", ""))
        words = _NUMBER_WORDS.get(str(value))
        differences = {abs(a - b) for a in numbers for b in numbers}
        return (
            value in numbers
            or value in differences
            or bool(words and re.search(rf"\b{words}\b", text))
        )
    # Comparison golds are sentences: the winner's name must appear in the context.
    key = re.findall(r"[A-Za-z0-9]{4,}", gold)
    return any(k.lower() in text for k in key)


def check_tasks(new: list[dict], aliases: dict[str, list[str]]) -> list[str]:
    """Problems that must stop the expansion (empty list = fine)."""
    problems: list[str] = []
    existing = {
        row["task_id"]: row
        for split in SPLITS
        for row in _read_jsonl(MASBENCH / split / f"{split}.jsonl")
        if row["task_id"] not in {t["task_id"] for t in new}
    }
    for task in new:
        tid = task["task_id"]
        arithmetic = task["source_dataset"] == "template_arithmetic"
        for other_id, other in existing.items():
            same = task["question"].strip().lower() == other["question"].strip().lower()
            if same or (
                not arithmetic and _jaccard(task["question"], other["question"]) >= NEAR_DUPLICATE
            ):
                problems.append(f"{tid} repeats {other_id}: {task['question']!r}")
        accepted = [task["ground_truth"], *aliases.get(tid, [])]
        for answer in accepted:
            if not is_correct(answer, task["ground_truth"], aliases.get(tid, [])):
                problems.append(f"{tid}: {answer!r} does not grade as correct")
        if not arithmetic and not task["context"]:
            problems.append(f"{tid}: a factual task needs a context")
        if task["context"] and not _supported_by_context(task["ground_truth"], task["context"]):
            problems.append(f"{tid}: the answer is not supported by the context")
    return problems


def _upsert_jsonl(path: Path, new: list[dict]) -> int:
    rows = _read_jsonl(path)
    by_id = {r["task_id"]: i for i, r in enumerate(rows)}
    added = 0
    for task in new:
        row = {
            k: task[k]
            for k in (
                "task_id",
                "question",
                "context",
                "depth_score",
                "parallel_score",
                "ground_truth",
                "source_dataset",
            )
        }
        if task["task_id"] in by_id:
            rows[by_id[task["task_id"]]] = row
        else:
            rows.append(row)
            added += 1
    _write_jsonl(path, rows)
    return added


def _upsert_csv(path: Path, new: list[dict]) -> int:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or list(CSV_FIELDS)
        rows = list(reader)
    have = {r["task_id"] for r in rows}
    added = 0
    for task in new:
        if task["task_id"] in have:
            continue
        rows.append({k: ("" if task[k] is None else task[k]) for k in fieldnames})
        added += 1
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8")
    return added


def main() -> int:
    new = build_new_tasks()
    aliases = build_aliases()
    problems = check_tasks(new, aliases)
    if problems:
        print("[FAIL] The new tasks have problems:")
        for p in problems:
            print("  -", p)
        return 1

    report = {
        "dataset/raw/tasks_raw.jsonl": _upsert_jsonl(
            ROOT / "dataset" / "raw" / "tasks_raw.jsonl", new
        ),
        "dataset/processed/tasks_cleaned.jsonl": _upsert_jsonl(
            ROOT / "dataset" / "processed" / "tasks_cleaned.jsonl", new
        ),
        "dataset/masbench_mini/test/test.jsonl": _upsert_jsonl(
            MASBENCH / "test" / "test.jsonl", new
        ),
        "exports/dataset/test.csv": _upsert_csv(ROOT / "exports" / "dataset" / "test.csv", new),
    }
    alias_path = MASBENCH / "answer_aliases.json"
    stored = json.loads(alias_path.read_text(encoding="utf-8"))
    before = len(stored)
    stored.update(aliases)
    alias_path.write_text(json.dumps(stored, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report["dataset/masbench_mini/answer_aliases.json"] = len(stored) - before

    for path, added in report.items():
        print(f"  {path:50} +{added}")
    total = len(_read_jsonl(MASBENCH / "test" / "test.jsonl"))
    print(f"[OK] {len(new)} test tasks defined; the test split now has {total} tasks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

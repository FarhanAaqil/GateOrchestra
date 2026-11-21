"""
The concise-answer step on the main test split (gpt-oss-120b reduced run)
=========================================================================

The benchmark recorded every Always-MAS answer. This restates each one with
the concise-answer step (GATE_MAS_CONCISE, one short call per task) and grades
both versions with the current grader, so the step's effect on the main split
is measured without re-running the multi-agent system.

    python scripts/concise_main_split.py extract   # checkpoint -> committed table (local checkpoint needed)
    python scripts/concise_main_split.py run       # one short call per task; resumable
    python scripts/concise_main_split.py analyze   # -> results/real/analysis/concise_main_split.json

The recorded benchmark results are not changed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CHECKPOINT = (
    ROOT
    / "logs"
    / "results"
    / "checkpoints"
    / "groq-gpt-oss-120b"
    / "final_eval_seed_42_n_all.json"
)
OUT_DIR = ROOT / "results" / "real" / "analysis"
TABLE = OUT_DIR / "concise_main_split.jsonl"
SUMMARY = OUT_DIR / "concise_main_split.json"
RUN_ENV = {
    "GATE_LLM_PROVIDER": "groq",
    "GROQ_MODEL_NAME": "openai/gpt-oss-120b",
    "GATE_RUN_TAG": "groq-gpt-oss-120b",
}


def extract() -> int:
    data = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    rows = []
    for key, entry in data["results"].items():
        if key.startswith("test:Always-MAS:"):
            r = entry.get("result", entry)
            rows.append(
                {
                    "task_id": r["task_id"],
                    "mas_answer": r["predicted_answer"],
                    "mas_tokens": r["tokens_spent"],
                    "recorded_correct": r["is_correct"],
                }
            )
    rows.sort(key=lambda r: r["task_id"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    print(f"wrote {len(rows)} rows to {TABLE.relative_to(ROOT)}")
    return 0


def run() -> int:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    for name, value in RUN_ENV.items():
        os.environ[name] = value
    from agents.orchestrator.orchestrator import request_orchestrator
    from shared.data_loader import load_split

    tasks = {t.task_id: t for t in load_split("test")}
    rows = [
        json.loads(line) for line in TABLE.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    for index, row in enumerate(rows, 1):
        if "concise_answer" in row:
            continue
        mas = request_orchestrator()
        short, extra = mas._concise(mas.react_agent, tasks[row["task_id"]], row["mas_answer"])
        row["concise_answer"], row["concise_extra_tokens"] = short, extra
        TABLE.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
        print(f"[{index}/{len(rows)}] {row['task_id']}: {short[:60]!r}", flush=True)
    return 0


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from scripts.threshold_baseline import mcnemar_exact, wilson
    from shared.data_loader import load_split
    from shared.grading import grade_task

    tasks = {t.task_id: t for t in load_split("test")}
    rows = [r for r in rows if "concise_answer" in r and r["task_id"] in tasks]
    plain = [bool(grade_task(r["mas_answer"], tasks[r["task_id"]])) for r in rows]
    short = [bool(grade_task(r["concise_answer"], tasks[r["task_id"]])) for r in rows]
    n = len(rows)
    fixed = sum(s and not p for p, s in zip(plain, short, strict=True))
    broke = sum(p and not s for p, s in zip(plain, short, strict=True))
    return {
        "model": RUN_ENV["GROQ_MODEL_NAME"],
        "n": n,
        "recorded_always_mas_correct": sum(r["recorded_correct"] for r in rows),
        "always_mas": {
            "accuracy": round(100 * sum(plain) / n, 1),
            "accuracy_ci95": wilson(sum(plain), n),
            "avg_tokens": round(sum(r["mas_tokens"] for r in rows) / n, 1),
        },
        "always_mas_concise": {
            "accuracy": round(100 * sum(short) / n, 1),
            "accuracy_ci95": wilson(sum(short), n),
            "avg_tokens": round(
                sum(r["mas_tokens"] + r["concise_extra_tokens"] for r in rows) / n, 1
            ),
        },
        "fixed": fixed,
        "broke": broke,
        "p_value": mcnemar_exact(fixed, broke),
        "changed": [
            {
                "task_id": r["task_id"],
                "plain": r["mas_answer"][:120],
                "concise": r["concise_answer"],
                "now_correct": s,
            }
            for r, p, s in zip(rows, plain, short, strict=True)
            if p != s
        ],
    }


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(errors="backslashreplace")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=["extract", "run", "analyze"])
    args = parser.parse_args(argv)
    if args.step == "extract":
        return extract()
    if args.step == "run":
        return run()
    rows = [
        json.loads(line) for line in TABLE.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    summary = analyze(rows)
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "changed"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

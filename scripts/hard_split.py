"""
Harder tasks, and a concise-answer step for the multi-agent system
===================================================================

The main test split is near its ceiling (CoT-SC scores 97-99%), so it cannot
show a gate earning its keep. This experiment adds 48 harder multi-step tasks
(dataset/masbench_hard/hard.jsonl: long arithmetic chains, date and logic
puzzles, three-step factual questions) and runs them on gpt-oss-120b with the
calibrated gate, router and probe budget of that model's reduced benchmark run
(GATE_RUN_TAG=groq-gpt-oss-120b, GATE_PROBE_TOKEN_BUDGET=2400).

For every task it records the CoT-SC probe, the multi-agent system's answer,
and that same answer restated by the concise-answer step (GATE_MAS_CONCISE),
then scores CoT-SC, Always-MAS (plain and concise), the learned gate, the
threshold rules and an oracle from those outcomes, as in
scripts/threshold_baseline.py.

    python scripts/hard_split.py run        # real model calls; resumable
    python scripts/hard_split.py reconcise  # redo only the concise restatement of each stored MAS answer
    python scripts/hard_split.py analyze    # -> results/real/analysis/hard_split.json, reports/hard_split.md

``analyze`` grades every stored answer again with the current grader, so a grading
fix applies to recorded runs without calling a model.

The main benchmark and its recorded results are not touched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TASKS = ROOT / "dataset" / "masbench_hard" / "hard.jsonl"
OUT_DIR = ROOT / "results" / "real" / "analysis"
RAW = OUT_DIR / "hard_split_raw.jsonl"
SUMMARY = OUT_DIR / "hard_split.json"
REPORT = ROOT / "reports" / "hard_split.md"
RUN_ENV = {
    "GATE_LLM_PROVIDER": "groq",
    "GROQ_MODEL_NAME": "openai/gpt-oss-120b",
    "GATE_PROBE_TOKEN_BUDGET": "2400",
    "GATE_RUN_TAG": "groq-gpt-oss-120b",
}
GATE_PATH = ROOT / "configs" / "models" / "groq-gpt-oss-120b" / "best_gate_seed_42.pkl"


def load_tasks(path: Path = TASKS) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def finished(path: Path = RAW) -> set[str]:
    if not path.exists():
        return set()
    return {
        json.loads(line)["task_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and "error" not in json.loads(line)
    }


def run(limit: int | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    for name, value in RUN_ENV.items():  # before the benchmark modules read their settings
        os.environ[name] = value

    from agents.orchestrator.orchestrator import request_orchestrator
    from agents.probe_agent import probe_agent
    from gate.classifier import load_provenanced_gate
    from gate.feature_extractor import extract_features
    from shared.config import K_DEFAULT, PROBE_TOKEN_BUDGET
    from shared.grading import grade_task
    from shared.schemas import Task

    gate = load_provenanced_gate(GATE_PATH)
    if gate is None:
        print(f"No usable gate at {GATE_PATH}", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = finished()
    todo = [t for t in load_tasks() if t["task_id"] not in done][: limit or None]
    print(
        f"{len(done)} tasks done, {len(todo)} to run (k={K_DEFAULT}, probe budget {PROBE_TOKEN_BUDGET})",
        flush=True,
    )
    budget = K_DEFAULT * PROBE_TOKEN_BUDGET
    for index, spec in enumerate(todo, 1):
        task = Task(**{k: v for k, v in spec.items() if k in Task.model_fields})
        record: dict[str, Any] = {"task_id": task.task_id, "source": spec["source_dataset"]}
        for attempt in range(6):
            try:
                probe = probe_agent(task)
                features = extract_features(task, probe)
                decision = gate.predict(
                    features, k=K_DEFAULT, probe_tokens=probe.tokens_used
                ).decision
                plain = request_orchestrator()
                plain.concise = False
                mas_answer, mas_tokens = plain(task, budget)
                # The concise step applied to this same answer, so the comparison isolates it.
                short_answer, extra = plain._concise(plain.react_agent, task, mas_answer)
                short_tokens = mas_tokens + extra
                record.update(
                    consistency=probe.consistency_score,
                    cot_answer=probe.answer,
                    cot_correct=bool(grade_task(probe.answer, task)),
                    cot_tokens=probe.tokens_used,
                    gate_decision=decision,
                    mas_answer=mas_answer,
                    mas_correct=bool(grade_task(mas_answer, task)),
                    mas_tokens=mas_tokens,
                    mas_concise_answer=short_answer,
                    mas_concise_correct=bool(grade_task(short_answer, task)),
                    mas_concise_tokens=short_tokens,
                    mas_strategy=plain._last_strategy,
                )
                record.pop("error", None)
                break
            except Exception as err:  # noqa: BLE001  rate limits and outages: wait and retry
                record["error"] = f"{type(err).__name__}: {str(err)[:160]}"
                wait = 60 * (attempt + 1)
                print(f"  {task.task_id}: {record['error']}; retrying in {wait}s", flush=True)
                time.sleep(wait)
        with RAW.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if "error" in record:
            print(f"[{index}/{len(todo)}] {task.task_id} failed", flush=True)
        else:
            print(
                f"[{index}/{len(todo)}] {task.task_id:<18} cot {'ok' if record['cot_correct'] else 'X '} "
                f"mas {'ok' if record['mas_correct'] else 'X '} concise {'ok' if record['mas_concise_correct'] else 'X '} "
                f"gate {decision}",
                flush=True,
            )
    return 0


def regrade(rows: list[dict[str, Any]]) -> int:
    """Grade stored answers with the current grader; returns how many verdicts changed."""
    from shared.grading import grade_task
    from shared.schemas import Task

    tasks = {
        t["task_id"]: Task(**{k: v for k, v in t.items() if k in Task.model_fields})
        for t in load_tasks()
    }
    changed = 0
    for row in rows:
        for key in ("cot", "mas", "mas_concise"):
            answer = row.get(f"{key}_answer")
            if answer is None or row["task_id"] not in tasks:
                continue
            verdict = bool(grade_task(answer, tasks[row["task_id"]]))
            changed += verdict != row.get(f"{key}_correct")
            row[f"{key}_correct"] = verdict
    return changed


def reconcise() -> int:
    """Redo the concise restatement of every stored MAS answer (short model calls only)."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    for name, value in RUN_ENV.items():
        os.environ[name] = value
    from agents.orchestrator.orchestrator import request_orchestrator
    from shared.schemas import Task

    tasks = {t["task_id"]: t for t in load_tasks()}
    rows = [
        json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    for index, row in enumerate(rows, 1):
        if "error" in row:
            continue
        spec = tasks[row["task_id"]]
        task = Task(**{k: v for k, v in spec.items() if k in Task.model_fields})
        mas = request_orchestrator()
        short, extra = mas._concise(mas.react_agent, task, row["mas_answer"])
        row["mas_concise_answer"], row["mas_concise_tokens"] = short, row["mas_tokens"] + extra
        print(f"[{index}/{len(rows)}] {row['task_id']}: {short[:60]!r}", flush=True)
    RAW.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return 0


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from scripts.threshold_baseline import mcnemar_exact, wilson

    rows = sorted(
        {r["task_id"]: r for r in rows if "error" not in r}.values(), key=lambda r: r["task_id"]
    )
    regraded = regrade(rows) if all("cot_answer" in r for r in rows) else 0

    def score(pick) -> dict[str, Any]:
        outcomes = [pick(r) for r in rows]
        n = len(outcomes)
        right = sum(c for c, _, _ in outcomes)
        return {
            "n": n,
            "accuracy": round(100 * right / n, 1) if n else 0.0,
            "accuracy_ci95": wilson(right, n),
            "avg_tokens": round(sum(t for _, t, _ in outcomes) / n, 1) if n else 0.0,
            "escalation_rate": round(100 * sum(e for _, _, e in outcomes) / n, 1) if n else 0.0,
            "_correct": [c for c, _, _ in outcomes],
        }

    def gated(escalate, concise: bool = False):
        def pick(r):
            if escalate(r):
                key = "mas_concise" if concise else "mas"
                return r[f"{key}_correct"], r["cot_tokens"] + r[f"{key}_tokens"], True
            return r["cot_correct"], r["cot_tokens"], False

        return pick

    methods = {
        "CoT-SC only": score(lambda r: (r["cot_correct"], r["cot_tokens"], False)),
        "Always-MAS": score(lambda r: (r["mas_correct"], r["mas_tokens"], True)),
        "Always-MAS + concise answer": score(
            lambda r: (r["mas_concise_correct"], r["mas_concise_tokens"], True)
        ),
        "GateOrchestra (learned gate)": score(gated(lambda r: r["gate_decision"] == "ESCALATE")),
        "GateOrchestra + concise answer": score(
            gated(lambda r: r["gate_decision"] == "ESCALATE", concise=True)
        ),
        "Threshold t=2/3": score(gated(lambda r: r["consistency"] < 2 / 3 - 1e-9)),
        "Threshold t=1": score(gated(lambda r: r["consistency"] < 1 - 1e-9)),
        "Oracle": score(gated(lambda r: (not r["cot_correct"]) and r["mas_correct"])),
    }

    def paired(a: str, b: str) -> dict[str, Any]:
        ca, cb = methods[a]["_correct"], methods[b]["_correct"]
        a_only = sum(x and not y for x, y in zip(ca, cb, strict=True))
        b_only = sum(y and not x for x, y in zip(ca, cb, strict=True))
        return {
            "a": a,
            "b": b,
            "a_only_correct": a_only,
            "b_only_correct": b_only,
            "p_value": mcnemar_exact(a_only, b_only),
        }

    tests = [
        paired("Always-MAS", "CoT-SC only"),
        paired("Always-MAS + concise answer", "Always-MAS"),
        paired("GateOrchestra (learned gate)", "CoT-SC only"),
        paired("GateOrchestra (learned gate)", "Threshold t=1"),
    ]
    by_source: dict[str, dict[str, Any]] = {}
    for source in sorted({r["source"] for r in rows}):
        sub = [r for r in rows if r["source"] == source]
        by_source[source] = {
            "n": len(sub),
            "cot": round(100 * sum(r["cot_correct"] for r in sub) / len(sub), 1),
            "mas": round(100 * sum(r["mas_correct"] for r in sub) / len(sub), 1),
            "mas_concise": round(100 * sum(r["mas_concise_correct"] for r in sub) / len(sub), 1),
            "mas_helps": sum((not r["cot_correct"]) and r["mas_correct"] for r in sub),
            "mas_hurts": sum(r["cot_correct"] and not r["mas_correct"] for r in sub),
        }
    for m in methods.values():
        m.pop("_correct")
    return {
        "model": RUN_ENV["GROQ_MODEL_NAME"],
        "n_tasks": len(load_tasks()),
        "n_scored": len(rows),
        "regraded_verdicts": regraded,
        "methods": methods,
        "paired_tests": tests,
        "by_source": by_source,
    }


def write_report(s: dict[str, Any]) -> str:
    lines = [
        "# Harder tasks and a concise-answer step",
        "",
        "Generated by `python scripts/hard_split.py analyze` from `results/real/analysis/hard_split_raw.jsonl`.",
        "",
        f"{s['n_scored']} of {s['n_tasks']} harder multi-step tasks (`dataset/masbench_hard/hard.jsonl`) on {s['model']}, with that",
        "model's calibrated gate, router and 2,400-token probe budget. One run per task, so there is no between-run spread.",
        "Gated rows pay for the probe and, when they escalate, for the multi-agent system too.",
        "",
        "| Method | Accuracy | 95% CI | Avg tokens | Escalation |",
        "| :--- | --: | :--- | --: | --: |",
    ]
    for name, m in s["methods"].items():
        lines.append(
            f"| {name} | {m['accuracy']:.1f}% | [{m['accuracy_ci95'][0]}, {m['accuracy_ci95'][1]}] | {m['avg_tokens']:,.0f} | {m['escalation_rate']:.1f}% |"
        )
    lines += [
        "",
        "## Paired tests (exact McNemar)",
        "",
        "| A | B | only A right | only B right | p |",
        "| :--- | :--- | --: | --: | --: |",
    ]
    for t in s["paired_tests"]:
        lines.append(
            f"| {t['a']} | {t['b']} | {t['a_only_correct']} | {t['b_only_correct']} | {t['p_value']} |"
        )
    lines += [
        "",
        "## By kind of task",
        "",
        "| Kind | n | CoT-SC | MAS | MAS + concise | MAS fixed | MAS broke |",
        "| :--- | --: | --: | --: | --: | --: | --: |",
    ]
    for src, b in s["by_source"].items():
        lines.append(
            f"| {src} | {b['n']} | {b['cot']:.0f}% | {b['mas']:.0f}% | {b['mas_concise']:.0f}% | {b['mas_helps']} | {b['mas_hurts']} |"
        )
    m = s["methods"]
    mas_vs_cot = s["paired_tests"][0]
    concise = s["paired_tests"][1]
    lines += [
        "",
        "## What it means",
        "",
        f"On these harder tasks the multi-agent system was right where CoT-SC was wrong {mas_vs_cot['a_only_correct']} time(s) and wrong where "
        f"CoT-SC was right {mas_vs_cot['b_only_correct']} time(s) (p = {mas_vs_cot['p_value']}). The concise-answer step changed Always-MAS from "
        f"{m['Always-MAS']['accuracy']:.1f}% to {m['Always-MAS + concise answer']['accuracy']:.1f}% (p = {concise['p_value']}). "
        f"An oracle gate would reach {m['Oracle']['accuracy']:.1f}% by escalating {m['Oracle']['escalation_rate']:.1f}% of tasks; the learned gate "
        f"reached {m['GateOrchestra (learned gate)']['accuracy']:.1f}% escalating {m['GateOrchestra (learned gate)']['escalation_rate']:.1f}%.",
        "",
        "48 tasks and one run are small: differences of a few tasks are within noise, and these tasks were written and checked by the "
        "assistant, so they need the same human review as the main split before being cited.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    # Model output can hold any character; a Windows console must not crash on one.
    sys.stdout.reconfigure(errors="backslashreplace")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=["run", "reconcise", "analyze"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.step == "run":
        return run(args.limit)
    if args.step == "reconcise":
        return reconcise()
    rows = [
        json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    summary = analyze(rows)
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(write_report(summary), encoding="utf-8")
    print(f"wrote {SUMMARY.relative_to(ROOT)} and {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

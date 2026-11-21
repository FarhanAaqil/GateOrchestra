"""
Is the learned gate better than a one-line rule?
================================================

The learned gate leans almost entirely on one feature: how much the probe's
three samples agree. This script asks whether the simplest possible gate on
that feature, "escalate when agreement is below t", does as well.

With three samples, agreement is 1/3, 2/3 or 1, so there are only three
threshold gates worth testing:

    never      escalate nothing (this is CoT-SC-only)
    t = 2/3    escalate only when all three samples disagree
    t = 1      escalate whenever any sample disagrees

Every gate is scored the same way, from outcomes the final benchmark recorded:
a STOP task keeps the probe's answer and tokens; an ESCALATE task takes the
recorded Always-MAS answer and pays the probe's tokens plus Always-MAS's.
The learned gate's own live result is reported next to its simulated one.

The threshold is chosen on the validation runs with the learned gate's own
rule (the cheapest option within epsilon = 1/32 of the best validation
accuracy), then scored once on the test split. All three are reported anyway.

Two steps, so the analysis reproduces from the repository alone:

    python scripts/threshold_baseline.py extract   # checkpoints -> per_task_outcomes.json
    python scripts/threshold_baseline.py analyze   # per_task_outcomes.json -> report

The checkpoints (logs/results/checkpoints/) are not in git; the extracted
per-task table is. Nothing under results/real/final is read or changed by
``analyze``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = ROOT / "logs" / "results" / "checkpoints"
OUT_DIR = ROOT / "results" / "real" / "analysis"
OUTCOMES = OUT_DIR / "per_task_outcomes.json"
SUMMARY = OUT_DIR / "threshold_baseline.json"
REPORT = ROOT / "reports" / "threshold_baseline.md"
SEEDS = (42, 123)
EPSILON = 1 / 32
THRESHOLDS = {"never": 0.0, "t=2/3": 2 / 3, "t=1": 1.0}


# Extraction (needs the local checkpoints)


def _result(entry: dict[str, Any]) -> dict[str, Any]:
    return entry.get("result", entry)


def extract(checkpoints: Path = CHECKPOINTS) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        path = checkpoints / f"final_eval_seed_{seed}_n_all.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        probes, results = data["probes"], data["results"]
        for key, probe in probes.items():
            split, _, task_id, repeat = key.split(":")
            cot = results.get(f"{split}:CoT-SC-only:{task_id}:{repeat}")
            mas = results.get(f"{split}:Always-MAS:{task_id}:{repeat}")
            if cot is None or mas is None:
                continue
            cot, mas = _result(cot), _result(mas)
            row: dict[str, Any] = {
                "seed": seed,
                "split": split,
                "task_id": task_id,
                "repeat": int(repeat),
                "consistency": round(float(probe["consistency_score"]), 4),
                "cot_correct": bool(cot["is_correct"]),
                "cot_tokens": int(cot["tokens_spent"]),
                "mas_correct": bool(mas["is_correct"]),
                "mas_tokens": int(mas["tokens_spent"]),
            }
            gate = results.get(f"{split}:GateOrchestra:{task_id}:{repeat}")
            if gate is not None:
                gate = _result(gate)
                row["gate_decision"] = gate["gate_decision"]["decision"]
                row["gate_correct"] = bool(gate["is_correct"])
                row["gate_tokens"] = int(gate["tokens_spent"])
            rows.append(row)
    rows.sort(key=lambda r: (r["seed"], r["split"], r["task_id"], r["repeat"]))
    return rows


# Analysis (repository only)


def outcome(row: dict[str, Any], escalate: bool) -> tuple[bool, int]:
    """(correct, tokens) for one task under a STOP or ESCALATE decision."""
    if escalate:
        return row["mas_correct"], row["cot_tokens"] + row["mas_tokens"]
    return row["cot_correct"], row["cot_tokens"]


def wilson(correct: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(100 * (centre - half), 1), round(100 * (centre + half), 1)]


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return round(min(1.0, 2 * tail), 4)


def score(rows: list[dict[str, Any]], decide) -> dict[str, Any]:
    picks = [(r, decide(r)) for r in rows]
    results = [outcome(r, esc) for r, esc in picks]
    n = len(rows)
    correct = sum(c for c, _ in results)
    return {
        "n": n,
        "accuracy": round(100 * correct / n, 1) if n else 0.0,
        "accuracy_ci95": wilson(correct, n),
        "avg_tokens": round(sum(t for _, t in results) / n, 1) if n else 0.0,
        "escalation_rate": round(100 * sum(esc for _, esc in picks) / n, 1) if n else 0.0,
        "_correct": [c for c, _ in results],
    }


def threshold_rule(t: float):
    return lambda r: r["consistency"] < t - 1e-9


def choose_threshold(val_rows: list[dict[str, Any]]) -> tuple[str, dict[str, dict[str, Any]]]:
    scores = {name: score(val_rows, threshold_rule(t)) for name, t in THRESHOLDS.items()}
    best = max(s["accuracy"] for s in scores.values())
    eligible = [n for n, s in scores.items() if s["accuracy"] >= best - 100 * EPSILON]
    chosen = min(eligible, key=lambda n: (scores[n]["avg_tokens"], n))
    return chosen, scores


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    val = [r for r in rows if r["split"] == "val"]
    test = [r for r in rows if r["split"] == "test"]
    chosen, val_scores = choose_threshold(val)

    methods: dict[str, dict[str, Any]] = {}
    methods["CoT-SC-only (never escalate)"] = score(test, lambda r: False)
    methods["Threshold t=2/3"] = score(test, threshold_rule(2 / 3))
    methods["Threshold t=1"] = score(test, threshold_rule(1.0))
    methods["Always-MAS (simulated: probe + MAS)"] = score(test, lambda r: True)
    methods["GateOrchestra (simulated)"] = score(
        test, lambda r: r.get("gate_decision") == "ESCALATE"
    )
    methods["Oracle (escalate only when it helps)"] = score(
        test, lambda r: (not r["cot_correct"]) and r["mas_correct"]
    )
    live_correct = [r["gate_correct"] for r in test]
    methods["GateOrchestra (live, as recorded)"] = {
        "n": len(test),
        "accuracy": round(100 * sum(live_correct) / len(test), 1),
        "accuracy_ci95": wilson(sum(live_correct), len(test)),
        "avg_tokens": round(sum(r["gate_tokens"] for r in test) / len(test), 1),
        "escalation_rate": round(
            100 * sum(r["gate_decision"] == "ESCALATE" for r in test) / len(test), 1
        ),
        "_correct": live_correct,
    }
    recorded_mas_tokens = sum(r["mas_tokens"] for r in test) / len(test)
    for m in methods.values():
        m["tokens_vs_always_mas_pct"] = round(
            100 * (m["avg_tokens"] - recorded_mas_tokens) / recorded_mas_tokens, 1
        )

    chosen_name = {
        "never": "CoT-SC-only (never escalate)",
        "t=2/3": "Threshold t=2/3",
        "t=1": "Threshold t=1",
    }[chosen]

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

    pairs = [
        ("GateOrchestra (live, as recorded)", chosen_name),
        ("GateOrchestra (live, as recorded)", "CoT-SC-only (never escalate)"),
        ("GateOrchestra (simulated)", "Threshold t=2/3"),
        ("GateOrchestra (simulated)", "Threshold t=1"),
    ]
    comparisons = [paired(x, y) for i, (x, y) in enumerate(pairs) if (x, y) not in pairs[:i]]
    agreement = defaultdict(int)
    for r in test:
        agreement[str(r["consistency"])] += 1

    for m in methods.values():
        m.pop("_correct")
    for s in val_scores.values():
        s.pop("_correct")
    return {
        "seeds": list(SEEDS),
        "n_test_runs": len(test),
        "n_val_runs": len(val),
        "epsilon": EPSILON,
        "chosen_threshold": chosen,
        "chosen_method": chosen_name,
        "validation": val_scores,
        "test": methods,
        "recorded_always_mas_avg_tokens": round(recorded_mas_tokens, 1),
        "paired_tests": comparisons,
        "test_agreement_counts": dict(sorted(agreement.items())),
        "note": (
            "Pooled over two seeds that share the same 96 test tasks, so p-values are optimistic. "
            "Simulated rows score an escalation with the recorded Always-MAS outcome plus the probe's tokens."
        ),
    }


def write_report(summary: dict[str, Any]) -> str:
    t = summary["test"]
    lines = [
        "# Is the learned gate better than a one-line rule?",
        "",
        "Generated by `python scripts/threshold_baseline.py analyze` from",
        "`results/real/analysis/per_task_outcomes.json` (extracted from the final benchmark's checkpoints).",
        "",
        "The learned gate depends almost entirely on probe agreement. With three samples agreement is",
        "1/3, 2/3 or 1, so only three threshold gates exist. Each is scored on the recorded outcomes of the",
        f"same {summary['n_test_runs']} test runs (96 tasks x seeds {', '.join(map(str, summary['seeds']))}).",
        "",
        f"**Threshold chosen on validation:** `{summary['chosen_threshold']}`, with the learned gate's own rule",
        f"(the cheapest option within {summary['epsilon']:.4f} of the best validation accuracy).",
        "",
        "| Method (test) | Accuracy | 95% CI | Avg tokens | vs Always-MAS | Escalation |",
        "| :--- | --: | :--- | --: | --: | --: |",
    ]
    for name, m in t.items():
        lines.append(
            f"| {name} | {m['accuracy']:.1f}% | [{m['accuracy_ci95'][0]}, {m['accuracy_ci95'][1]}] | "
            f"{m['avg_tokens']:,.0f} | {m['tokens_vs_always_mas_pct']:+.1f}% | {m['escalation_rate']:.1f}% |"
        )
    lines += [
        "",
        f"'vs Always-MAS' compares with the recorded Always-MAS average of {summary['recorded_always_mas_avg_tokens']:,.0f} tokens.",
        "Simulated rows score an escalation as the recorded Always-MAS answer plus the probe's tokens, so they",
        "pay for both, as a gated method does.",
        "",
        "## Paired tests (exact McNemar)",
        "",
        "| A | B | only A correct | only B correct | p |",
        "| :--- | :--- | --: | --: | --: |",
    ]
    for c in summary["paired_tests"]:
        lines.append(
            f"| {c['a']} | {c['b']} | {c['a_only_correct']} | {c['b_only_correct']} | {c['p_value']} |"
        )
    lines += [
        "",
        "Pooled over two seeds that share the same tasks, so these p-values are optimistic.",
        "",
        "## Validation (where the threshold was chosen)",
        "",
        "| Threshold | Accuracy | Avg tokens | Escalation |",
        "| :--- | --: | --: | --: |",
    ]
    for name, s in summary["validation"].items():
        lines.append(
            f"| {name} | {s['accuracy']:.1f}% | {s['avg_tokens']:,.0f} | {s['escalation_rate']:.1f}% |"
        )
    lines += [
        "",
        "## What it means",
        "",
        conclusion(summary),
        "",
    ]
    return "\n".join(lines)


def conclusion(summary: dict[str, Any]) -> str:
    t = summary["test"]
    live = t["GateOrchestra (live, as recorded)"]
    chosen = t[summary["chosen_method"]]
    oracle = t["Oracle (escalate only when it helps)"]
    cot = t["CoT-SC-only (never escalate)"]
    p = summary["paired_tests"][0]["p_value"]
    same = abs(live["accuracy"] - chosen["accuracy"]) <= 1.0
    verdict = (
        "On this benchmark the learned gate is not measurably better than the one-line rule"
        if same or p >= 0.05
        else "On this benchmark the learned gate is measurably better than the one-line rule"
    )
    return (
        f"{verdict}: {live['accuracy']:.1f}% at {live['avg_tokens']:,.0f} tokens against "
        f"{chosen['accuracy']:.1f}% at {chosen['avg_tokens']:,.0f} for `{summary['chosen_threshold']}` (p = {p}). "
        f"The ceiling is low for any gate: an oracle that escalates only when Always-MAS fixes a wrong probe answer "
        f"reaches {oracle['accuracy']:.1f}% by escalating {oracle['escalation_rate']:.1f}% of runs, against "
        f"{cot['accuracy']:.1f}% for never escalating. Escalating only when all three samples disagree (t=2/3) "
        f"reaches {t['Threshold t=2/3']['accuracy']:.1f}% at {t['Threshold t=2/3']['avg_tokens']:,.0f} tokens. "
        f"The simulated gate ({t['GateOrchestra (simulated)']['avg_tokens']:,.0f} tokens) tracks the live one "
        f"({live['avg_tokens']:,.0f}), which supports the scoring. The gate's measured value here is that it learned not to "
        "escalate; a benchmark where the multi-agent system wins more often is needed to show a learned gate earning "
        "its keep over a threshold."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=["extract", "analyze"])
    args = parser.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.step == "extract":
        rows = extract()
        OUTCOMES.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {len(rows)} rows to {OUTCOMES.relative_to(ROOT)}")
        return 0
    rows = json.loads(OUTCOMES.read_text(encoding="utf-8"))
    summary = analyze(rows)
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(write_report(summary), encoding="utf-8")
    print(f"wrote {SUMMARY.relative_to(ROOT)} and {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

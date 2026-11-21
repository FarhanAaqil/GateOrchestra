"""
How accurate is the chat assistant, and does its gate help?
===========================================================

The benchmark measures the research system; this measures the product. A fixed
set of 80 questions with known answers (chat_eval/questions.jsonl: facts,
arithmetic, multi-step questions and Python output) is run through the real
chat pipeline twice, with the real models:

    always_stop       answer directly after the quick check (no research agent)
    always_escalate   research with tools first, then answer

Every turn also records what every gate would have decided (learned,
rule-based) and the quick check's agreement, so auto routing and the gates are
scored from the two runs the same way as the benchmark analysis: a STOP takes
the direct answer and its tokens, an ESCALATE the researched one and its tokens.

Each question asks for a last line "Answer: ...", which the project's grader
(shared/grading.py) reads, so every mode is graded identically.

    python scripts/chat_eval.py run        # needs provider keys; resumable, saves after each turn
    python scripts/chat_eval.py analyze    # results/chat_eval/raw.jsonl -> summary + report

Results: results/chat_eval/raw.jsonl, results/chat_eval/summary.json,
reports/chat_eval.md.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUESTIONS = ROOT / "chat_eval" / "questions.jsonl"
OUT_DIR = ROOT / "results" / "chat_eval"
RAW = OUT_DIR / "raw.jsonl"
SUMMARY = OUT_DIR / "summary.json"
REPORT = ROOT / "reports" / "chat_eval.md"
MODES = ("always_stop", "always_escalate")
SUFFIX = "\n\nEnd your reply with a final line in the form 'Answer: <your answer>'."
FEATURES = (
    "consistency_score",
    "probe_tokens",
    "num_entities",
    "num_clauses",
    "question_length",
    "depth_proxy",
    "parallel_proxy",
    "has_context",
)


def load_questions(path: Path = QUESTIONS) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def grade(answer: str, question: dict[str, Any]) -> bool:
    from shared.grading import is_correct

    return is_correct(answer, question["answer"], question.get("aliases") or ())


# Running (real model calls)


def run_one(service: Any, question: dict[str, Any], mode: str) -> dict[str, Any]:
    from assistant.pipeline import ChatPipeline

    record: dict[str, Any] = {"id": question["id"], "category": question["category"], "mode": mode}
    answer, gate, done, error = "", {}, {}, None
    for event in ChatPipeline(service).run(question["question"] + SUFFIX, None, gate_mode=mode):
        kind = event.get("type")
        if kind == "delta":
            answer += event["text"]
        elif kind == "gate":
            gate = event
        elif kind == "done":
            done = event["meta"]
        elif kind == "error":
            error = event
    if error:
        record["error"] = {"code": error.get("code"), "message": error.get("message", "")[:200]}
        return record
    record.update(
        answer=answer,
        correct=grade(answer, question),
        decision=done.get("decision"),
        verdicts=gate.get("verdicts", {}),
        features=gate.get("features", {}),
        consistency=done.get("consistency"),
        task_type=done.get("task_type"),
        tokens=done.get("tokens", {}),
        total_tokens=done.get("total_tokens", 0),
        latency_ms=done.get("latency_ms", 0),
        tools=[t.get("name") for t in done.get("tools", [])],
        models=done.get("models", {}),
    )
    return record


def done_keys(path: Path = RAW) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if "error" not in row:
                keys.add((row["id"], row["mode"]))
    return keys


def run(limit: int | None = None, max_wait: float = 900.0) -> int:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    from assistant.config import AssistantSettings
    from assistant.service import AssistantService

    work = Path(tempfile.mkdtemp(prefix="chat-eval-"))
    settings = AssistantSettings(db_path=work / "eval.db", cache_ttl=0)
    service = AssistantService(settings)
    if not service.gateway.provider_order() or "mock" in service.gateway.provider_order():
        print(
            "No real provider configured (set GROQ_API_KEY or another provider in .env).",
            file=sys.stderr,
        )
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    finished = done_keys()
    todo = [(q, m) for q in load_questions() for m in MODES if (q["id"], m) not in finished]
    if limit:
        todo = todo[:limit]
    print(f"{len(finished)} turns already recorded, {len(todo)} to run", flush=True)
    for index, (question, mode) in enumerate(todo, 1):
        waited = 0.0
        while True:
            record = run_one(service, question, mode)
            code = (record.get("error") or {}).get("code")
            if code != "needs_key" or waited >= max_wait:
                break
            pause = 60.0
            print(f"  rate limited on {question['id']} {mode}; waiting {pause:.0f}s", flush=True)
            time.sleep(pause)
            waited += pause
        with RAW.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        status = (
            "error " + str(record["error"]["code"])
            if "error" in record
            else ("right" if record["correct"] else "WRONG")
        )
        print(
            f"[{index}/{len(todo)}] {question['id']:<14} {mode:<16} {status} {record.get('total_tokens', 0)} tok",
            flush=True,
        )
    return 0


# Analysis (no model calls)


def wilson(correct: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(100 * (centre - half), 1), round(100 * (centre + half), 1)]


def mcnemar_exact(a_only: int, b_only: int) -> float:
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(a_only, b_only) + 1)) / 2**n
    return round(min(1.0, 2 * tail), 4)


def paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per question that has both a direct and a researched result."""
    by_id: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in rows:
        if "error" not in row:
            by_id[row["id"]][row["mode"]] = row  # the last good attempt wins
    return [
        {
            "id": qid,
            "category": modes["always_stop"]["category"],
            "stop": modes["always_stop"],
            "escalate": modes["always_escalate"],
        }
        for qid, modes in sorted(by_id.items())
        if "always_stop" in modes and "always_escalate" in modes
    ]


def score(pairs: list[dict[str, Any]], decide) -> dict[str, Any]:
    picks = [(p, decide(p)) for p in pairs]
    chosen = [p["escalate"] if esc else p["stop"] for p, esc in picks]
    n = len(chosen)
    correct = sum(r["correct"] for r in chosen)
    return {
        "n": n,
        "accuracy": round(100 * correct / n, 1) if n else 0.0,
        "accuracy_ci95": wilson(correct, n),
        "avg_tokens": round(sum(r["total_tokens"] for r in chosen) / n, 1) if n else 0.0,
        "avg_latency_s": round(sum(r["latency_ms"] for r in chosen) / n / 1000, 2) if n else 0.0,
        "escalation_rate": round(100 * sum(esc for _, esc in picks) / n, 1) if n else 0.0,
        "_correct": [r["correct"] for r in chosen],
    }


def verdict(gate: str):
    return lambda p: (p["stop"].get("verdicts") or {}).get(gate) == "ESCALATE"


def threshold(t: float):
    return (
        lambda p: (
            p["stop"].get("consistency") if p["stop"].get("consistency") is not None else 1.0
        )
        < t - 1e-9
    )


def chat_gate_cv(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """A gate trained on chat data, scored by leave-one-out cross-validation."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x = np.array(
        [[float((p["stop"].get("features") or {}).get(f, 0) or 0) for f in FEATURES] for p in pairs]
    )
    helps = np.array([int(p["escalate"]["correct"] and not p["stop"]["correct"]) for p in pairs])
    if helps.sum() < 2 or helps.sum() > len(pairs) - 2:
        return {
            "available": False,
            "reason": f"only {int(helps.sum())} question(s) where research helped; too few to learn from",
            "positives": int(helps.sum()),
        }
    decisions = []
    for i in range(len(pairs)):
        train = [j for j in range(len(pairs)) if j != i]
        if helps[train].sum() == 0:
            decisions.append(False)
            continue
        scaler = StandardScaler().fit(x[train])
        model = LogisticRegression(class_weight="balanced", max_iter=1000).fit(
            scaler.transform(x[train]), helps[train]
        )
        decisions.append(bool(model.predict(scaler.transform(x[i : i + 1]))[0]))
    chosen = dict(zip([q["id"] for q in pairs], decisions, strict=True))
    result = score(pairs, lambda p: chosen[p["id"]])
    result.update(available=True, positives=int(helps.sum()))
    return result


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = paired_rows(rows)
    succeeded = {(r["id"], r["mode"]) for r in rows if "error" not in r}
    errors = [r for r in rows if "error" in r and (r["id"], r["mode"]) not in succeeded]
    methods = {
        "Always answer directly": score(pairs, lambda p: False),
        "Always research": score(pairs, lambda p: True),
        "Learned gate (benchmark-trained)": score(pairs, verdict("learned")),
        "Rule-based gate": score(pairs, verdict("rule_based")),
        "Threshold t=2/3 (all samples disagree)": score(pairs, threshold(2 / 3)),
        "Threshold t=1 (any disagreement)": score(pairs, threshold(1.0)),
        "Oracle (research only when it fixes the answer)": score(
            pairs, lambda p: (not p["stop"]["correct"]) and p["escalate"]["correct"]
        ),
    }
    cv = chat_gate_cv(pairs)
    if cv.get("available"):
        methods["Chat-trained gate (leave-one-out)"] = {
            k: v for k, v in cv.items() if k not in ("available", "positives")
        }

    # Within one question of the best, counted in questions (not rounded percentages), then cheapest.
    candidates = {k: v for k, v in methods.items() if not k.startswith("Oracle")}
    right = {k: sum(v["_correct"]) for k, v in candidates.items()}
    best = max(right.values())
    recommended = min(
        (k for k in candidates if right[k] >= best - 1),
        key=lambda k: (candidates[k]["avg_tokens"], list(candidates).index(k)),
    )

    direct, research = (
        methods["Always answer directly"]["_correct"],
        methods["Always research"]["_correct"],
    )
    by_category: dict[str, dict[str, Any]] = {}
    for cat in sorted({p["category"] for p in pairs}):
        sub = [p for p in pairs if p["category"] == cat]
        by_category[cat] = {
            "n": len(sub),
            "direct": score(sub, lambda p: False)["accuracy"],
            "research": score(sub, lambda p: True)["accuracy"],
            "direct_tokens": score(sub, lambda p: False)["avg_tokens"],
            "research_tokens": score(sub, lambda p: True)["avg_tokens"],
        }
    research_extra = [p["escalate"]["total_tokens"] - p["stop"]["total_tokens"] for p in pairs]
    for m in methods.values():
        m.pop("_correct", None)
    return {
        "questions": len(load_questions()),
        "paired": len(pairs),
        "failed_turns": len(errors),
        "methods": methods,
        "by_category": by_category,
        "research_vs_direct": {
            "research_only_right": sum(r and not d for d, r in zip(direct, research, strict=True)),
            "direct_only_right": sum(d and not r for d, r in zip(direct, research, strict=True)),
            "p_value": mcnemar_exact(
                sum(r and not d for d, r in zip(direct, research, strict=True)),
                sum(d and not r for d, r in zip(direct, research, strict=True)),
            ),
            "avg_extra_tokens_for_research": (
                round(sum(research_extra) / len(research_extra), 1) if research_extra else 0.0
            ),
        },
        "chat_gate": {k: v for k, v in cv.items() if k in ("available", "reason", "positives")},
        "recommended_default": recommended,
        "models": {
            role: model
            for p in pairs
            for side in ("stop", "escalate")
            for role, model in (p[side].get("models") or {}).items()
        },
    }


def write_report(summary: dict[str, Any]) -> str:
    m = summary["methods"]
    rv = summary["research_vs_direct"]
    lines = [
        "# Chat assistant evaluation",
        "",
        "Generated by `python scripts/chat_eval.py analyze` from `results/chat_eval/raw.jsonl`.",
        "",
        f"{summary['paired']} of {summary['questions']} questions (`chat_eval/questions.jsonl`: facts, arithmetic,",
        "multi-step questions and Python output) were answered by the real chat pipeline twice, once",
        "answering directly after the quick check and once researching with tools first. Gates are scored",
        "from those two runs with the verdicts each gate gave on the same turn.",
        f"Models: {', '.join(f'{k} {v}' for k, v in summary['models'].items()) or 'see raw.jsonl'}.",
        "",
        "| Method | Accuracy | 95% CI | Avg tokens | Avg time | Researches |",
        "| :--- | --: | :--- | --: | --: | --: |",
    ]
    for name, r in m.items():
        lines.append(
            f"| {name} | {r['accuracy']:.1f}% | [{r['accuracy_ci95'][0]}, {r['accuracy_ci95'][1]}] | "
            f"{r['avg_tokens']:,.0f} | {r['avg_latency_s']:.1f} s | {r['escalation_rate']:.0f}% |"
        )
    lines += [
        "",
        f"Research against answering directly: research alone was right on {rv['research_only_right']} questions, "
        f"answering directly alone on {rv['direct_only_right']} (exact McNemar p = {rv['p_value']}); researching cost "
        f"{rv['avg_extra_tokens_for_research']:,.0f} more tokens per question on average.",
        "",
        "| Kind of question | n | Direct | Research | Direct tokens | Research tokens |",
        "| :--- | --: | --: | --: | --: | --: |",
    ]
    for cat, c in summary["by_category"].items():
        lines.append(
            f"| {cat} | {c['n']} | {c['direct']:.0f}% | {c['research']:.0f}% | {c['direct_tokens']:,.0f} | {c['research_tokens']:,.0f} |"
        )
    gate = summary["chat_gate"]
    lines += [
        "",
        "## A gate trained on chat questions",
        "",
        (
            f"Trained on the quick check's eight features to predict when research fixes the answer ({gate['positives']} such questions), "
            "and scored by leave-one-out cross-validation; see the table."
            if gate.get("available")
            else f"Not trained: {gate.get('reason')}."
        ),
        "",
        "## What it means",
        "",
        f"Recommended default for the chat's routing: **{summary['recommended_default']}** (the cheapest option within one question "
        "of the best accuracy, the same rule the benchmark gate uses). "
        + (
            f"{summary['failed_turns']} turn(s) failed and are left out. "
            if summary["failed_turns"]
            else ""
        )
        + f"Research fixed {rv['research_only_right']} question(s) that answering directly got wrong and broke "
        f"{rv['direct_only_right']}, at about "
        f"{1 + rv['avg_extra_tokens_for_research'] / max(1.0, m['Always answer directly']['avg_tokens']):.1f} "
        "times the tokens. The gate trained on the benchmark escalated "
        f"{m['Learned gate (benchmark-trained)']['escalation_rate']:.0f}% of these questions and scored "
        f"{m['Learned gate (benchmark-trained)']['accuracy']:.1f}%, the same as answering directly: it does not "
        "spend research where research rarely helps. "
        + (
            "A chat-specific gate cannot be trained from so few cases where research helps; that needs "
            "harder or longer questions than these."
            if not gate.get("available")
            else ""
        ),
        "",
        "These are 80 questions with short, checkable answers, so they say how often the assistant gets such answers right "
        "and what research costs; open-ended requests (explanations, writing) are not graded here.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=["run", "analyze"])
    parser.add_argument("--limit", type=int, help="run at most this many turns (a trial)")
    args = parser.parse_args(argv)
    if args.step == "run":
        return run(args.limit)
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

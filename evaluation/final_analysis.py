"""
evaluation/final_analysis.py
============================
Analyses of the final benchmark that need no further LLM calls.

Everything here reads saved checkpoints (every probe and every graded answer,
including the train/validation repeats used to calibrate the gate) and the
per-seed result files:

  premise          Is MAS more accurate than CoT-SC, and at what cost? (train + val)
  oracle_gate      Best any STOP/ESCALATE gate could do with these two systems
  axis_cells       CoT-SC vs MAS accuracy by Depth x Parallel label and by source
  feature_ablation Retrain the gate with each feature neutralised (leave one out)
  error_analysis   How the methods fail on the test split, and what MAS fixes/breaks
  routing          Which sub-agents the learned router picked, and how they did
  budget           How often MAS spends more than its k x probe-token cap
  power            Smallest accuracy difference the test split can detect
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist, mean
from typing import Any

from evaluation.metrics import mcnemar_exact, wilson_confidence_interval
from gate.classifier import FEATURE_NAMES, make_classifier
from gate.feature_extractor import extract_features
from gate.train_gate import apply_repeated_label_rule, evaluate_validation_policy
from shared.config import K_DEFAULT, TAU_ACC
from shared.grading import normalize_text
from shared.schemas import EvalResult, GateFeatures, ProbeResult, Task

COT = "CoT-SC-only"
MAS = "Always-MAS"
NO_ANSWER = {"", "unknown", "none", "n/a", "[simulated: no llm answer]"}
LENIENT_MAX_WORDS = 25
_FINISH = re.compile(r"^\s*finish\s*\[(.*)\]\s*$", re.I)
_NEGATION = re.compile(r"\b(?:not|no|never|neither|nor)\b|n't", re.I)


def lenient_correct(result: EvalResult, task: Task) -> bool:
    """Strict grade, or the gold answer/an alias stated inside a short answer.

    A sensitivity check for wordy answers ("Bong Joon-ho was born in South
    Korea") and leaked action syntax ("finish[South China Sea]"). Answers over
    25 words or containing a negation never pass on this rule alone.
    """
    if result.is_correct:
        return True
    predicted = result.predicted_answer or ""
    if match := _FINISH.match(predicted):
        predicted = match.group(1)
    text = normalize_text(predicted)
    if not text or len(text.split()) > LENIENT_MAX_WORDS or _NEGATION.search(predicted):
        return False
    for candidate in (task.ground_truth or "", *(task.answer_aliases or ())):
        wanted = normalize_text(candidate)
        if len(wanted) >= 2 and re.search(rf"\b{re.escape(wanted)}\b", text):
            return True
    return False


def _regraded(results: list[EvalResult], task: Task) -> list[EvalResult]:
    return [r.model_copy(update={"is_correct": lenient_correct(r, task)}) for r in results]


# Loading


def repeated_results(state: dict[str, Any], phase: str, method: str) -> dict[str, list[EvalResult]]:
    """task_id -> results for every repeat of ``method`` in a train/val ``phase``."""
    out: dict[str, list[tuple[int, EvalResult]]] = defaultdict(list)
    prefix = f"{phase}:{method}:"
    for key, value in state["results"].items():
        if not key.startswith(prefix):
            continue
        task_id, repeat = key[len(prefix) :].rsplit(":", 1)
        payload = value["result"] if "result" in value else value
        out[task_id].append((int(repeat), EvalResult.model_validate(payload)))
    return {tid: [r for _, r in sorted(runs, key=lambda x: x[0])] for tid, runs in out.items()}


def single_results(state: dict[str, Any], phase: str, method: str) -> dict[str, EvalResult]:
    """task_id -> the one result of ``method`` in a single-run ``phase`` (e.g. test)."""
    return {tid: runs[0] for tid, runs in repeated_results(state, phase, method).items()}


def probes(state: dict[str, Any], phase: str, repeat: int = 0) -> dict[str, ProbeResult]:
    out = {}
    for key, value in state["probes"].items():
        p, _, task_id, r = key.split(":")
        if p == phase and int(r) == repeat:
            out[task_id] = ProbeResult.model_validate(value)
    return out


# Small statistics helpers


def _rate(correct: int, total: int) -> dict[str, Any]:
    low, high = wilson_confidence_interval(correct, total) if total else (0.0, 0.0)
    return {
        "correct": correct,
        "n": total,
        "accuracy": correct / total if total else 0.0,
        "ci95": [low, high],
    }


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test (ties already removed)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def mcnemar_mde(n_pairs: int, discordant_rate: float, alpha=0.05, power=0.8) -> float | None:
    """Smallest accuracy difference a paired McNemar test detects with this power.

    Uses the normal approximation n = (z_a*sqrt(p_d) + z_b*sqrt(p_d - d^2))^2 / d^2,
    solved for d by bisection. ``None`` when even d = p_d is not detectable.
    """
    if n_pairs <= 0 or discordant_rate <= 0:
        return None
    z_a = NormalDist().inv_cdf(1 - alpha / 2)
    z_b = NormalDist().inv_cdf(power)
    p_d = discordant_rate

    def needed(d: float) -> float:
        return (z_a * math.sqrt(p_d) + z_b * math.sqrt(max(p_d - d * d, 0.0))) ** 2 / (d * d)

    if needed(p_d) > n_pairs:
        return None
    low, high = 1e-6, p_d
    for _ in range(60):
        mid = (low + high) / 2
        low, high = (mid, high) if needed(mid) > n_pairs else (low, mid)
    return high


# Analyses


def premise(cot: dict[str, list[EvalResult]], mas: dict[str, list[EvalResult]]) -> dict[str, Any]:
    """CoT-SC vs MAS on the same tasks, every repeat graded."""
    task_ids = sorted(set(cot) & set(mas))
    cot_runs = [r for t in task_ids for r in cot[t]]
    mas_runs = [r for t in task_ids for r in mas[t]]
    cot_acc = {t: mean(r.is_correct is True for r in cot[t]) for t in task_ids}
    mas_acc = {t: mean(r.is_correct is True for r in mas[t]) for t in task_ids}
    mas_wins = sum(mas_acc[t] > cot_acc[t] for t in task_ids)
    cot_wins = sum(cot_acc[t] > mas_acc[t] for t in task_ids)
    cot_tokens = mean(r.tokens_spent for r in cot_runs)
    mas_tokens = mean(r.tokens_spent for r in mas_runs)
    cot_correct = sum(r.is_correct is True for r in cot_runs)
    mas_correct = sum(r.is_correct is True for r in mas_runs)
    return {
        "n_tasks": len(task_ids),
        "n_runs": len(cot_runs),
        "cot_sc": {**_rate(cot_correct, len(cot_runs)), "avg_tokens": cot_tokens},
        "mas": {**_rate(mas_correct, len(mas_runs)), "avg_tokens": mas_tokens},
        "accuracy_gain_pts": 100 * (mas_correct - cot_correct) / max(1, len(cot_runs)),
        "token_ratio": mas_tokens / cot_tokens if cot_tokens else None,
        "tokens_per_correct": {
            "cot_sc": sum(r.tokens_spent for r in cot_runs) / max(1, cot_correct),
            "mas": sum(r.tokens_spent for r in mas_runs) / max(1, mas_correct),
        },
        "tasks_mas_better": mas_wins,
        "tasks_cot_better": cot_wins,
        "tasks_tied": len(task_ids) - mas_wins - cot_wins,
        "sign_test_p": sign_test(mas_wins, cot_wins),
    }


def oracle_gate(cot: dict[str, EvalResult], mas: dict[str, EvalResult]) -> dict[str, Any]:
    """Escalate exactly when MAS is right and CoT-SC is wrong: the ceiling for any gate.

    An escalated task pays for the probe and the MAS run, as in the pipeline.
    """
    task_ids = sorted(set(cot) & set(mas))
    correct = tokens = escalated = 0
    for t in task_ids:
        if mas[t].is_correct and not cot[t].is_correct:
            escalated += 1
            correct += 1
            tokens += cot[t].tokens_spent + (mas[t].mas_tokens or mas[t].tokens_spent)
        else:
            correct += cot[t].is_correct is True
            tokens += cot[t].tokens_spent
    mas_tokens = mean(mas[t].tokens_spent for t in task_ids)
    return {
        **_rate(correct, len(task_ids)),
        "escalate_rate": escalated / len(task_ids),
        "avg_tokens": tokens / len(task_ids),
        "token_savings_vs_mas_pct": 100 * (1 - (tokens / len(task_ids)) / mas_tokens),
    }


def axis_cells(
    cot: dict[str, list[EvalResult]],
    mas: dict[str, list[EvalResult]],
    tasks: dict[str, Task],
) -> dict[str, list[dict[str, Any]]]:
    """Accuracy of each system per Depth x Parallel cell and per source dataset."""

    def group(key_fn) -> list[dict[str, Any]]:
        buckets: dict[Any, list[str]] = defaultdict(list)
        for t in set(cot) & set(mas):
            buckets[key_fn(tasks[t])].append(t)
        rows = []
        for key, ids in sorted(buckets.items(), key=lambda kv: str(kv[0])):
            c = [r.is_correct is True for t in ids for r in cot[t]]
            m = [r.is_correct is True for t in ids for r in mas[t]]
            rows.append(
                {
                    "key": key,
                    "n_tasks": len(ids),
                    "cot_sc_accuracy": mean(c),
                    "mas_accuracy": mean(m),
                    "mas_gain_pts": 100 * (mean(m) - mean(c)),
                }
            )
        return rows

    return {
        "depth_parallel": group(lambda t: f"D{t.depth_score}xP{t.parallel_score}"),
        "depth": group(lambda t: f"D{t.depth_score}"),
        "parallel": group(lambda t: f"P{t.parallel_score}"),
        "source": group(lambda t: t.source_dataset or "unknown"),
    }


def _neutralise(features: list[GateFeatures], name: str, value: Any) -> list[GateFeatures]:
    return [f.model_copy(update={name: value}) for f in features]


def feature_ablation(
    train_features: list[GateFeatures],
    train_cot: dict[str, list[EvalResult]],
    train_mas: dict[str, list[EvalResult]],
    val_features: list[GateFeatures],
    val_cot: dict[str, EvalResult],
    val_mas: dict[str, EvalResult],
    classifiers: tuple[str, ...] = ("logreg", "gbt"),
    tau: float = TAU_ACC,
    k: int = K_DEFAULT,
) -> dict[str, Any]:
    """Validation accuracy/savings of gates retrained with one feature neutralised.

    Setting a feature to one constant in both train and validation removes the
    information it carries without changing the model's input shape.
    """
    n_repeats = len(next(iter(train_cot.values())))
    labels_by_task = apply_repeated_label_rule(train_cot, train_mas, n_repeats, tau)
    labels = [labels_by_task[f.task_id] for f in train_features]
    escalate_share = labels.count("ESCALATE") / len(labels)
    rows: list[dict[str, Any]] = []
    if escalate_share in (0.0, 1.0):
        return {"escalate_label_share": escalate_share, "rows": rows, "note": "one label only"}

    def score(name: str, train: list[GateFeatures], val: list[GateFeatures], dropped: str):
        gate = make_classifier(name)
        gate.train(train, labels)
        policy = evaluate_validation_policy(gate, val, val_cot, {k: val_mas}, k)
        rows.append({"classifier": name, "dropped": dropped, **policy})

    for name in classifiers:
        score(name, train_features, val_features, "none")
        for feature in FEATURE_NAMES:
            value = getattr(train_features[0], feature)
            score(
                name,
                _neutralise(train_features, feature, value),
                _neutralise(val_features, feature, value),
                feature,
            )
    return {"escalate_label_share": escalate_share, "tau_acc": tau, "k": k, "rows": rows}


def classify_error(predicted: str, gold: str) -> str:
    """Coarse failure type of a wrong answer."""
    p = (predicted or "").strip().lower()
    g = (gold or "").strip().lower()
    if p in NO_ANSWER or p.startswith("error"):
        return "no answer"
    p_nums = re.findall(r"-?\d+(?:\.\d+)?", p.replace(",", ""))
    g_nums = re.findall(r"-?\d+(?:\.\d+)?", g.replace(",", ""))
    if g_nums and p_nums:
        return "wrong number"
    p_words = set(re.findall(r"[a-z]{3,}", p))
    g_words = set(re.findall(r"[a-z]{3,}", g))
    if g_words and p_words & g_words:
        return "partly right"
    if len(p.split()) > 12:
        return "no clear answer"
    return "wrong entity"


def error_analysis(
    test_by_seed: dict[int, dict[str, dict[str, EvalResult]]], tasks: dict[str, Task]
) -> dict[str, Any]:
    """Failure types per method and how MAS changes CoT-SC's answers, over all seeds."""
    types: dict[str, Counter] = defaultdict(Counter)
    transitions = Counter()
    examples: list[dict[str, Any]] = []
    for seed, methods in sorted(test_by_seed.items()):
        for method, results in methods.items():
            for task_id, result in results.items():
                if not result.is_correct:
                    types[method][
                        classify_error(result.predicted_answer, tasks[task_id].ground_truth or "")
                    ] += 1
        cot, mas = methods.get(COT, {}), methods.get(MAS, {})
        for task_id in sorted(set(cot) & set(mas)):
            c, m = cot[task_id].is_correct, mas[task_id].is_correct
            label = (
                "both right"
                if c and m
                else "both wrong" if not c and not m else ("MAS fixes" if m else "MAS breaks")
            )
            transitions[label] += 1
            if label in ("MAS fixes", "MAS breaks") and len(examples) < 12:
                examples.append(
                    {
                        "seed": seed,
                        "task_id": task_id,
                        "change": label,
                        "question": tasks[task_id].question,
                        "gold": tasks[task_id].ground_truth,
                        "cot_sc": cot[task_id].predicted_answer,
                        "mas": mas[task_id].predicted_answer,
                        "mas_strategy": mas[task_id].mas_strategy,
                    }
                )
    return {
        "failure_types": {m: dict(c.most_common()) for m, c in sorted(types.items())},
        "cot_to_mas": dict(transitions),
        "examples": examples,
    }


def budget_overshoot(results: list[EvalResult], budgets: list[int]) -> dict[str, Any]:
    """How far MAS spends past its cap (the cap is enforced on a chars/4 prompt estimate)."""
    over = [
        (r.tokens_spent - b) / b
        for r, b in zip(results, budgets, strict=True)
        if b > 0 and r.tokens_spent > b
    ]
    return {
        "n": len(results),
        "over_budget": len(over),
        "over_budget_pct": 100 * len(over) / max(1, len(results)),
        "max_overshoot_pct": 100 * max(over, default=0.0),
        "mean_overshoot_pct_when_over": 100 * mean(over) if over else 0.0,
    }


def power(test_by_seed: dict[int, dict[str, dict[str, EvalResult]]], method: str) -> dict[str, Any]:
    """Minimum detectable difference vs Always-MAS, per seed and pooled."""
    a: list[bool] = []
    b: list[bool] = []
    for methods in test_by_seed.values():
        x, y = methods.get(method, {}), methods.get(MAS, {})
        ids = sorted(set(x) & set(y))
        a += [x[t].is_correct is True for t in ids]
        b += [y[t].is_correct is True for t in ids]
    if not a:
        return {"method": method, "n_pairs": 0}
    test = mcnemar_exact(a, b)
    discordant = (test["a_only"] + test["b_only"]) / len(a)
    per_seed = len(a) // max(1, len(test_by_seed))
    return {
        "method": method,
        "reference": MAS,
        "n_pairs_pooled": len(a),
        "discordant_rate": discordant,
        "mde_pts_per_seed": _pts(mcnemar_mde(per_seed, discordant)),
        "mde_pts_pooled": _pts(mcnemar_mde(len(a), discordant)),
        "observed_diff_pts": 100 * (sum(a) - sum(b)) / len(a),
    }


def _pts(value: float | None) -> float | None:
    return None if value is None else 100 * value


# Orchestration


def analyse(
    checkpoints_dir: Path,
    results_dir: Path,
    tasks: dict[str, Task],
    split_label: str = "all",
) -> dict[str, Any]:
    """Run every analysis on a finished (or partial) final benchmark."""
    seed_results = {
        int(p.stem.split("_")[3]): json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(results_dir.glob(f"final_eval_seed_*_n_{split_label}.json"))
    }
    states = {
        int(p.stem.split("_")[3]): json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(checkpoints_dir.glob(f"final_eval_seed_*_n_{split_label}.json"))
    }
    if not states:
        raise FileNotFoundError(f"No checkpoints in {checkpoints_dir}")
    calibration_seed = next(
        (r.get("calibration_seed") for r in seed_results.values() if r.get("calibration_seed")),
        min(states),
    )
    calib = states[calibration_seed]
    out: dict[str, Any] = {"calibration_seed": calibration_seed, "seeds": sorted(states)}

    train_cot = repeated_results(calib, "train", COT)
    train_mas = repeated_results(calib, "train", MAS)
    val_cot = repeated_results(calib, "val", COT)
    val_mas = repeated_results(calib, "val", MAS)
    complete_train = {
        t for t in train_cot if t in train_mas and len(train_mas[t]) == len(train_cot[t])
    }
    train_cot = {t: train_cot[t] for t in complete_train}
    train_mas = {t: train_mas[t] for t in complete_train}

    calib_cot = {**train_cot, **val_cot}
    calib_mas = {**train_mas, **{t: v for t, v in val_mas.items() if t in val_cot}}
    if calib_cot and calib_mas:
        out["premise"] = premise(calib_cot, calib_mas)
        out["oracle_gate_calibration"] = oracle_gate(
            {t: r[0] for t, r in calib_cot.items()}, {t: r[0] for t, r in calib_mas.items()}
        )
        out["axis"] = axis_cells(calib_cot, calib_mas, tasks)

    if train_cot and val_cot and val_mas:
        train_probes, val_probes = probes(calib, "train"), probes(calib, "val")
        train_features = [extract_features(tasks[t], train_probes[t]) for t in sorted(train_cot)]
        val_ids = sorted(set(val_cot) & set(val_mas))
        val_features = [extract_features(tasks[t], val_probes[t]) for t in val_ids]
        out["feature_ablation"] = feature_ablation(
            train_features,
            train_cot,
            train_mas,
            val_features,
            {t: val_cot[t][0] for t in val_ids},
            {t: val_mas[t][0] for t in val_ids},
        )

    test_by_seed = {
        seed: {
            m: single_results(state, "test", m)
            for m in (COT, MAS, "RandomGate", "RuleBasedGate", "GateOrchestra")
        }
        for seed, state in states.items()
    }
    test_by_seed = {s: m for s, m in test_by_seed.items() if m[COT] and m[MAS]}
    if calib_cot and calib_mas:
        lenient_cot = {t: _regraded(r, tasks[t]) for t, r in calib_cot.items()}
        lenient_mas = {t: _regraded(r, tasks[t]) for t, r in calib_mas.items()}
        out["lenient"] = {"premise": premise(lenient_cot, lenient_mas)}
    if test_by_seed:
        lenient_test: dict[str, dict[str, Any]] = {}
        for method in (COT, MAS, "RandomGate", "RuleBasedGate", "GateOrchestra"):
            strict = [r for seed in test_by_seed.values() for r in seed.get(method, {}).values()]
            if not strict:
                continue
            loose = sum(lenient_correct(r, tasks[r.task_id]) for r in strict)
            lenient_test[method] = {
                "strict": _rate(sum(r.is_correct is True for r in strict), len(strict)),
                "lenient": _rate(loose, len(strict)),
            }
        out.setdefault("lenient", {})["test"] = lenient_test
        out["error_analysis"] = error_analysis(test_by_seed, tasks)
        out["power"] = [power(test_by_seed, m) for m in ("GateOrchestra", COT, "RuleBasedGate")]
        first = test_by_seed[min(test_by_seed)]
        out["oracle_gate_test"] = oracle_gate(first[COT], first[MAS])

    mas_runs: list[EvalResult] = []
    budgets: list[int] = []
    for state in states.values():
        for phase in ("train", "val"):
            phase_probes = {(t, r): p for r in range(3) for t, p in probes(state, phase, r).items()}
            for t, runs in repeated_results(state, phase, MAS).items():
                for r, result in enumerate(runs):
                    if (t, r) in phase_probes:
                        mas_runs.append(result)
                        budgets.append(K_DEFAULT * phase_probes[(t, r)].tokens_used)
        break  # train/val are shared by every seed; count them once
    if mas_runs:
        out["budget"] = budget_overshoot(mas_runs, budgets)

    out["routing"] = {seed: r.get("mas_routing") for seed, r in seed_results.items()}
    out["calibration_candidates"] = next(
        (r.get("calibration") for r in seed_results.values() if r.get("calibration")), []
    )
    out["gate"] = next((r.get("gate") for r in seed_results.values() if r.get("gate")), None)
    return out

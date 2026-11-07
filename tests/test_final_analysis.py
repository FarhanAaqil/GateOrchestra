"""Tests for evaluation/final_analysis.py and its report script."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation import final_analysis as fa
from scripts.analyze_final_results import render_markdown
from shared.schemas import EvalResult, ProbeResult, Task


def _task(i: int, split: str) -> Task:
    return Task(
        task_id=f"{split}_{i}",
        question=f"What is {i} plus {i}? Name {'Paris and Rome' if i % 2 else 'Oslo'}.",
        ground_truth=str(2 * i),
        depth_score=1 + i % 3,
        parallel_score=1 + i % 2,
        source_dataset="template_arithmetic" if i % 2 else "hotpotqa_style",
    )


def _result(task: Task, method: str, correct: bool, tokens: int, strategy=None) -> dict:
    return EvalResult(
        task_id=task.task_id,
        method=method,
        predicted_answer=task.ground_truth if correct else "Unknown",
        is_correct=correct,
        tokens_spent=tokens,
        probe_tokens=tokens if method != "Always-MAS" else None,
        mas_tokens=tokens if method == "Always-MAS" else None,
        mas_strategy=strategy,
    ).model_dump(mode="json")


def _probe(task: Task, consistency: float, tokens: int = 600) -> dict:
    return ProbeResult(
        task_id=task.task_id,
        answer=task.ground_truth or "",
        consistency_score=consistency,
        raw_outputs=["Final Answer: x"] * 3,
        tokens_used=tokens,
        latency_ms=10.0,
    ).model_dump(mode="json")


@pytest.fixture()
def benchmark(tmp_path: Path) -> tuple[Path, Path, dict[str, Task]]:
    """A finished 1-seed benchmark: MAS fixes the low-consistency tasks CoT-SC misses."""
    tasks: dict[str, Task] = {}
    state: dict = {"probes": {}, "results": {}}
    for split, n, repeats in (("train", 12, 3), ("val", 6, 3), ("test", 6, 1)):
        for i in range(n):
            task = _task(i, split)
            tasks[task.task_id] = task
            hard = i % 3 == 0  # CoT-SC wrong, MAS right
            for r in range(repeats):
                state["probes"][f"{split}:probe:{task.task_id}:{r}"] = _probe(
                    task, 0.34 if hard else 1.0
                )
                cot = _result(task, "CoT-SC-only", not hard, 600)
                mas = _result(task, "Always-MAS", True, 1000 + (300 if i == 1 else 0), "react")
                if split == "test":
                    state["results"][f"test:CoT-SC-only:{task.task_id}:0"] = cot
                    state["results"][f"test:Always-MAS:{task.task_id}:0"] = mas
                    state["results"][f"test:GateOrchestra:{task.task_id}:0"] = cot
                else:
                    for method, res in (("CoT-SC-only", cot), ("Always-MAS", mas)):
                        state["results"][f"{split}:{method}:{task.task_id}:{r}"] = {
                            "repeat_index": r,
                            "result": res,
                        }
    checkpoints, results = tmp_path / "ck", tmp_path / "res"
    checkpoints.mkdir()
    results.mkdir()
    (checkpoints / "final_eval_seed_42_n_all.json").write_text(json.dumps(state))
    seed_result = {
        "calibration_seed": 42,
        "mas_routing": {"train": {"react": {"n": 36, "correct": 36, "accuracy": 1.0}}},
        "calibration": [
            {
                "classifier": "gbt",
                "tau_acc": 0.05,
                "k": 3,
                "accuracy": 1.0,
                "token_savings_pct": 30.0,
                "escalation_rate": 0.33,
            }
        ],
        "gate": {"classifier": "gbt"},
    }
    (results / "final_eval_seed_42_n_all.json").write_text(json.dumps(seed_result))
    return checkpoints, results, tasks


def test_analyse_finds_where_mas_helps(benchmark) -> None:
    checkpoints, results, tasks = benchmark
    out = fa.analyse(checkpoints, results, tasks)

    premise = out["premise"]
    assert premise["n_tasks"] == 18
    assert premise["mas"]["accuracy"] == 1.0
    assert premise["cot_sc"]["accuracy"] == pytest.approx(12 / 18)
    assert premise["tasks_mas_better"] == 6 and premise["tasks_cot_better"] == 0
    assert premise["sign_test_p"] == pytest.approx(2 / 64)

    oracle = out["oracle_gate_calibration"]
    assert oracle["accuracy"] == 1.0
    assert oracle["escalate_rate"] == pytest.approx(6 / 18)

    gains = {row["key"]: row["mas_gain_pts"] for row in out["axis"]["depth"]}
    assert gains["D1"] == pytest.approx(100.0)  # i % 3 == 0 -> depth 1, all "hard"

    rows = out["feature_ablation"]["rows"]
    baseline = next(r for r in rows if r["classifier"] == "gbt" and r["dropped"] == "none")
    no_consistency = next(
        r for r in rows if r["classifier"] == "gbt" and r["dropped"] == "consistency_score"
    )
    assert baseline["accuracy"] == 1.0
    assert no_consistency["accuracy"] <= baseline["accuracy"]

    errors = out["error_analysis"]
    assert errors["cot_to_mas"] == {"MAS fixes": 2, "both right": 4}
    assert errors["failure_types"]["CoT-SC-only"] == {"no answer": 2}
    assert out["budget"]["over_budget"] == 0
    assert out["routing"][42]["train"]["react"]["n"] == 36

    markdown = render_markdown(out)
    for heading in ("premise", "Depth x Parallel", "feature ablation", "Error analysis", "power"):
        assert heading.lower() in markdown.lower()


def test_classify_error_types() -> None:
    assert fa.classify_error("Unknown", "Paris") == "no answer"
    assert fa.classify_error("41", "42") == "wrong number"
    assert fa.classify_error("Paris, Texas", "Paris") == "partly right"
    assert fa.classify_error("Rome", "Paris") == "wrong entity"


def test_budget_overshoot_and_power_helpers() -> None:
    task = _task(1, "train")
    results = [EvalResult.model_validate(_result(task, "Always-MAS", True, t)) for t in (900, 1100)]
    out = fa.budget_overshoot(results, [1000, 1000])
    assert out["over_budget"] == 1 and out["max_overshoot_pct"] == pytest.approx(10.0)

    # 100 pairs with 40% discordant: a 20-point gap is detectable, 1 pair is not enough.
    assert fa.mcnemar_mde(100, 0.4) == pytest.approx(0.19, abs=0.03)
    assert fa.mcnemar_mde(1, 0.4) is None
    assert fa.mcnemar_mde(0, 0.4) is None
    assert fa.sign_test(0, 0) == 1.0


def test_lenient_grading_accepts_wordy_but_not_negated_answers() -> None:
    task = Task(
        task_id="t",
        question="Where was he born?",
        ground_truth="South Korea",
        answer_aliases=("Korea",),
    )

    def graded(answer: str) -> bool:
        result = EvalResult(
            task_id="t",
            method="Always-MAS",
            predicted_answer=answer,
            is_correct=False,
            tokens_spent=1,
        )
        return fa.lenient_correct(result, task)

    assert graded("Bong Joon-ho was born in South Korea")
    assert graded("finish[South Korea]")
    assert not graded("He was not born in South Korea")
    assert not graded("North America")
    assert not graded("South Korea " + "word " * 30)

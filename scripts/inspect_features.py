"""
scripts/inspect_features.py
============================
Day 3 -- Loads real tasks and shows extracted features for each.

Gives you a clear picture of what the gate classifier will actually see.
Uses a simulated probe (not real LLM yet) to generate consistency scores
that reflect realistic behavior: high for factoid, low for parallel/multihop.

Run:
    python scripts/inspect_features.py
    python scripts/inspect_features.py --split val --n 20
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gate.feature_extractor import extract_features
from shared.data_loader import load_split
from shared.schemas import GateFeatures, ProbeResult, Task

random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# Realistic probe simulator (Day 3 version — heuristic, not real LLM)
# ─────────────────────────────────────────────────────────────────────────────


def simulate_probe(task: Task) -> ProbeResult:
    """
    Simulate a CoT-SC probe result based on task difficulty heuristics.
    Simple tasks -> high consistency, low tokens.
    Complex tasks -> low consistency, high tokens.

    Will be replaced by real LLM calls in Week 3.
    """
    depth = task.depth_score or 2
    parallel = task.parallel_score or 1

    # Difficulty score: 0 (easy) to 1 (hard)
    difficulty = ((depth - 1) / 4 * 0.6) + ((parallel - 1) / 3 * 0.4)

    # Consistency: high for easy, noisy for hard
    base_consistency = 1.0 - difficulty * 0.7
    noise = random.gauss(0, 0.08)
    consistency = max(0.1, min(1.0, base_consistency + noise))

    # Token count: easy tasks use fewer tokens
    base_tokens = 80 + int(difficulty * 370)
    noise_tokens = random.randint(-30, 30)
    tokens = max(50, base_tokens + noise_tokens)

    # Generate fake CoT outputs (consistent or inconsistent)
    n_samples = 5
    answers = _generate_fake_answers(task, n_samples, consistency)

    return ProbeResult(
        task_id=task.task_id,
        answer=answers[0],
        consistency_score=round(consistency, 3),
        tokens_used=tokens,
        raw_outputs=answers,
        model_name="simulated_v1",
    )


def _generate_fake_answers(task: Task, n: int, consistency: float) -> list[str]:
    """Generate n CoT-SC sample outputs with controlled consistency."""
    gt = task.ground_truth or "Unknown"
    answers = []
    for _ in range(n):
        if random.random() < consistency:
            answers.append(gt)
        else:
            # Wrong answer
            wrong = gt + " (incorrect)"
            answers.append(wrong)
    return answers


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def run(split: str = "val", n: int = 15) -> None:
    print("=" * 90)
    print(f"  GateOrchestra -- Feature Inspector (Day 3)  |  split={split!r}  n={n}")
    print("=" * 90)

    tasks = load_split(split)  # type: ignore[arg-type]
    sample = random.sample(tasks, min(n, len(tasks)))

    # Header
    header = (
        f"  {'task_id':<22} {'D':>2} {'P':>2}  "
        f"{'consist':>7}  {'tokens':>6}  "
        f"{'entities':>8}  {'clauses':>7}  {'words':>5}  "
        f"{'est_depth':>9}  {'est_par':>7}  {'ctx':>4}"
    )
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))

    feature_rows: list[tuple[Task, GateFeatures]] = []

    for task in sample:
        probe = simulate_probe(task)
        features = extract_features(task, probe, use_spacy=False)
        feature_rows.append((task, features))

        d = task.depth_score or "?"
        p = task.parallel_score or "?"
        ctx = "Y" if features.has_context else "N"
        ed = f"{features.estimated_depth:.2f}" if features.estimated_depth is not None else "  -"
        ep = (
            f"{features.estimated_parallel:.2f}"
            if features.estimated_parallel is not None
            else "  -"
        )

        print(
            f"  {task.task_id:<22} {str(d):>2} {str(p):>2}  "
            f"{features.consistency_score:>7.3f}  {features.probe_tokens:>6}  "
            f"{features.entity_count:>8}  {features.clause_count:>7}  {features.question_word_count:>5}  "
            f"{ed:>9}  {ep:>7}  {ctx:>4}"
        )

    # Summary stats
    print(f"\n  {'--- Summary (' + split + ') ---':^86}")
    all_f = [f for _, f in feature_rows]
    avg_consistency = sum(f.consistency_score for f in all_f) / len(all_f)
    avg_tokens = sum(f.probe_tokens for f in all_f) / len(all_f)
    n_stop_rule = sum(1 for f in all_f if f.consistency_score >= 0.8 and f.entity_count < 3)

    print(f"  Avg consistency:       {avg_consistency:.3f}")
    print(f"  Avg probe tokens:      {avg_tokens:.0f}")
    print(
        f"  Rule-STOP candidates:  {n_stop_rule}/{len(all_f)} " f"(consistency>=0.8 AND entities<3)"
    )

    # Breakdown by task type
    print("\n  Consistency by task type:")
    type_groups: dict[str, list[float]] = {}
    for task, f in feature_rows:
        src = task.source_dataset or "unknown"
        type_groups.setdefault(src, []).append(f.consistency_score)

    for src, scores in sorted(type_groups.items()):
        avg = sum(scores) / len(scores)
        bar = "#" * int(avg * 20)
        print(f"    {src:>25}: avg={avg:.2f}  {bar}")

    print("\n[OK] Day 3 complete. Features visible on real data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect gate features on real tasks")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=15, help="Number of tasks to show")
    args = parser.parse_args()
    run(split=args.split, n=args.n)

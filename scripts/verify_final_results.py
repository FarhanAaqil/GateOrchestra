"""scripts/verify_final_results.py
===============================
Final-result integrity checker for GateOrchestra.

Checks results/real/final/master_results.json (written by
scripts/final_benchmark.py) and fails unless:

  1. The file exists, parses, and has execution_mode == "real".
  2. Every required seed (42, 123, 999) is present, with no null seeds, and
     each has exactly one per-seed result.
  3. Each per-seed result is a real run of the full test split on the
     current dataset (its dataset_sha256 matches the files on disk).
  4. All five methods are present in every seed, with per-task records.
  5. No result is simulated or mock-derived (execution mode, gate or model
     names mentioning mock/simulat*).
  6. The aggregated numbers equal what the per-seed results imply.

Use --allow-partial to verify a smoke run (fewer seeds or a test subset);
the report then says PARTIAL instead of PASS.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.data_loader import dataset_sha256, load_split  # noqa: E402

REAL_FINAL_DIR = ROOT / "results" / "real" / "final"
MASTER_RESULTS_PATH = REAL_FINAL_DIR / "master_results.json"

REQUIRED_SEEDS = {42, 123, 999}
REQUIRED_METHODS = {
    "CoT-SC-only",
    "Always-MAS",
    "RandomGate",
    "RuleBasedGate",
    "GateOrchestra",
}
REQUIRED_METRICS = {
    "accuracy_mean",
    "accuracy_pooled_ci95",
    "avg_tokens_mean",
    "token_savings_pct_mean",
}
MOCK_MARKERS = ("mock", "simulat")


def _contains_mock_marker(value: object) -> bool:
    return isinstance(value, str) and any(m in value.lower() for m in MOCK_MARKERS)


def verify_master_results(
    master_path: Path = MASTER_RESULTS_PATH,
    *,
    allow_partial: bool = False,
    expected_dataset_hash: str | None = None,
    expected_test_size: int | None = None,
) -> tuple[list[str], list[str]]:
    """Verify master_results.json.

    Returns:
        (errors, partial_notes). ``partial_notes`` lists the ways a run is a
        smoke run; they are errors unless ``allow_partial`` is set.
    """
    if not master_path.exists():
        return [f"Missing canonical results file: {master_path}"], []
    try:
        data = json.loads(master_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"Corrupted JSON in {master_path}: {exc}"], []

    errors: list[str] = []
    partial: list[str] = []
    expected_dataset_hash = expected_dataset_hash or dataset_sha256()
    if expected_test_size is None:
        expected_test_size = len(load_split("test"))

    if data.get("execution_mode") != "real":
        errors.append(f"execution_mode must be 'real', got {data.get('execution_mode')!r}")

    seeds = data.get("seeds") or []
    if any(not isinstance(seed, int) for seed in seeds):
        errors.append(f"Seeds must all be integers, got {seeds!r}")
    missing_seeds = REQUIRED_SEEDS - {s for s in seeds if isinstance(s, int)}
    if missing_seeds:
        partial.append(f"Missing required seeds: {sorted(missing_seeds)}")

    per_seed = data.get("per_seed") or []
    per_seed_seeds = [run.get("seed") for run in per_seed]
    if sorted(per_seed_seeds, key=str) != sorted(seeds, key=str):
        errors.append(f"per_seed seeds {per_seed_seeds!r} do not match seeds {seeds!r}")

    expected_by_method: dict[str, list[dict]] = {}
    for run in per_seed:
        label = f"seed {run.get('seed')!r}"
        if run.get("execution_mode") != "real" or run.get("dry_run"):
            errors.append(f"{label}: not a real run (execution_mode={run.get('execution_mode')!r})")
        if run.get("dataset_sha256") != expected_dataset_hash:
            errors.append(
                f"{label}: dataset hash {run.get('dataset_sha256')!r} does not match the "
                f"current dataset {expected_dataset_hash[:16]}..."
            )
        if run.get("n") is not None:
            partial.append(f"{label}: ran on a subset (n={run.get('n')})")
        test_ids = run.get("test_task_ids") or []
        if len(test_ids) != expected_test_size:
            partial.append(f"{label}: {len(test_ids)} test tasks, expected {expected_test_size}")
        gate = run.get("gate") or {}
        if _contains_mock_marker(gate.get("classifier")) or _contains_mock_marker(gate.get("note")):
            errors.append(f"{label}: gate looks simulated/mock-derived: {gate!r}")
        if gate.get("degenerate"):
            print(f"  [note] {label}: the selected gate is degenerate ({gate.get('note')})")

        rows = {row.get("method"): row for row in run.get("results", [])}
        missing = REQUIRED_METHODS - set(rows)
        if missing:
            errors.append(f"{label}: missing methods {sorted(missing)}")
        per_task = run.get("per_task") or {}
        for method in REQUIRED_METHODS & set(rows):
            records = per_task.get(method)
            if not records:
                errors.append(f"{label}: no per-task records for {method}")
                continue
            if {r["task_id"] for r in records} != set(test_ids):
                errors.append(f"{label}: {method} per-task ids differ from test_task_ids")
            n_correct = sum(1 for r in records if r.get("is_correct") is True)
            if rows[method].get("n_correct") != n_correct or rows[method].get("n") != len(records):
                errors.append(f"{label}: {method} summary does not match its per-task records")
            expected_by_method.setdefault(method, []).append(rows[method])

    aggregated = {row.get("method"): row for row in data.get("aggregated") or []}
    for method in REQUIRED_METHODS:
        row = aggregated.get(method)
        if row is None:
            errors.append(f"Aggregated results missing method {method!r}")
            continue
        for metric in REQUIRED_METRICS:
            if metric not in row:
                errors.append(f"Method {method!r} missing aggregated metric {metric!r}")
        if row.get("seeds") != [
            run.get("seed")
            for run in per_seed
            if method in {r.get("method") for r in run.get("results", [])}
        ]:
            errors.append(f"Method {method!r} aggregated seeds do not match per-seed results")
        seed_rows = expected_by_method.get(method, [])
        if seed_rows:
            mean_acc = round(sum(r["accuracy"] for r in seed_rows) / len(seed_rows) * 100, 2)
            if abs(float(row.get("accuracy_mean", -1)) - mean_acc) > 0.01:
                errors.append(
                    f"Method {method!r} accuracy_mean {row.get('accuracy_mean')} does not match "
                    f"per-seed results ({mean_acc})"
                )

    if not allow_partial:
        errors.extend(partial)
    return errors, partial


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify final GateOrchestra results")
    parser.add_argument("--path", type=Path, default=MASTER_RESULTS_PATH)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Accept smoke runs (fewer seeds or a test subset); reported as PARTIAL.",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  GateOrchestra -- Final Result Integrity Checker")
    print("=" * 65)
    current_hash = dataset_sha256()
    print(f"  Dataset SHA-256: {current_hash}")

    errors, partial = verify_master_results(args.path, allow_partial=args.allow_partial)
    if errors:
        print("\n[FAIL] Integrity checks failed:")
        for e in errors:
            print(f"  - {e}")
        return 1
    if partial:
        print("\n[PARTIAL] Results are internally consistent but are a smoke run:")
        for note in partial:
            print(f"  - {note}")
        return 0

    print("\n[PASS] All final results verified successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

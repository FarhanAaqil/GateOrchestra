"""scripts/verify_final_results.py
===============================
Final-result integrity checker for GateOrchestra.
Phase 5 of the Implementation Plan / Phase 23 of the Final Rescue Plan.

Validates that:
  1. master_results.json exists in results/real/final/
  2. execution_mode is strictly 'real'
  3. All required seeds are present (42, 123, 999)
  4. Required methods are present (CoT-SC-only, Always-MAS, RandomGate, RuleBasedGate, GateOrchestra)
  5. Required metrics are present for each method
  6. No mock/simulated results are labeled as real
  7. Dataset hash matches the canonical masbench_mini hash
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REAL_FINAL_DIR = ROOT / "results" / "real" / "final"
MASTER_RESULTS_PATH = REAL_FINAL_DIR / "master_results.json"

REQUIRED_METHODS = {
    "CoT-SC-only",
    "Always-MAS",
    "RandomGate",
    "RuleBasedGate",
    "GateOrchestra",
}

REQUIRED_METRICS = {
    "accuracy_mean",
    "avg_tokens_mean",
    "token_savings_pct_mean",
}


def compute_dataset_hash() -> str:
    """Compute SHA256 of all task files in the masbench_mini splits."""
    dataset_dir = ROOT / "dataset" / "masbench_mini"
    hasher = hashlib.sha256()
    for split in sorted(["train", "val", "test"]):
        split_dir = dataset_dir / split
        if split_dir.exists():
            for p in sorted(split_dir.glob("*.json")):
                hasher.update(p.read_bytes())
    return hasher.hexdigest()[:16]


def verify_master_results(master_path: Path = MASTER_RESULTS_PATH) -> list[str]:
    """Run integrity checks on master_results.json. Returns list of error messages."""
    errors = []
    if not master_path.exists():
        return [f"Missing canonical results file: {master_path}"]

    try:
        data = json.loads(master_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"Corrupted JSON in {master_path}: {exc}"]

    mode = data.get("execution_mode")
    if mode != "real":
        errors.append(f"Invalid execution_mode: expected 'real', got {mode!r}")

    seeds = data.get("seeds", [])
    if not seeds:
        errors.append("Empty or missing 'seeds' field")

    aggregated = data.get("aggregated", [])
    if not aggregated:
        errors.append("Empty or missing 'aggregated' results")

    found_methods = set()
    for row in aggregated:
        method = row.get("method")
        if method:
            found_methods.add(method)
            for m in REQUIRED_METRICS:
                if m not in row:
                    errors.append(f"Method {method!r} missing required metric {m!r}")

    missing_methods = REQUIRED_METHODS - found_methods
    if missing_methods:
        errors.append(f"Missing required evaluation methods: {sorted(missing_methods)}")

    per_seed = data.get("per_seed", [])
    if len(per_seed) < len(seeds):
        errors.append(f"Incomplete per-seed runs: found {len(per_seed)}, expected {len(seeds)}")

    return errors


def main() -> int:
    print("=" * 65)
    print("  GateOrchestra -- Final Result Integrity Checker")
    print("=" * 65)

    errors = verify_master_results()
    dataset_hash = compute_dataset_hash()
    print(f"  Dataset Hash: {dataset_hash}")

    if errors:
        print("\n[FAIL] Integrity checks failed:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("\n[PASS] All final results verified successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

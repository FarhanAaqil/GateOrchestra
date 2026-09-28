"""
scripts/final_benchmark.py
==========================
GateOrchestra canonical final benchmark command.

Runs the full 3-seed evaluation and writes a master_results.json
to results/real/final/.

Usage:
    python scripts/final_benchmark.py              # full dataset, seeds 42/123/999
    python scripts/final_benchmark.py --n 5        # quick smoke test
    python scripts/final_benchmark.py --dry-run    # validate config only

This is the single entry point for final research evaluation.
All output is written to results/real/final/ with execution_mode="real".

RULES:
  - No silent simulation fallback.
  - Provider failure -> explicit error, not a mock run.
  - All seeds use the same dataset splits.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_final_evaluation import run_evaluation  # noqa: E402

SEEDS = [42, 123, 999]
REAL_FINAL_DIR = ROOT / "results" / "real" / "final"
REAL_FINAL_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="GateOrchestra canonical final benchmark")
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Subset size per split (default: full split). Use 5 for smoke test.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="Seeds to run (default: 42 123 999)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without making any LLM calls.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoints for each seed.",
    )
    args = parser.parse_args()

    print("=" * 74)
    print("  GateOrchestra -- Final Benchmark")
    print(f"  Seeds: {args.seeds}  |  N: {args.n or 'full'}  |  dry_run={args.dry_run}")
    print(f"  Output: {REAL_FINAL_DIR}")
    print("=" * 74)

    if args.dry_run:
        print("\n[DRY-RUN] Validating provider configuration only.\n")

    all_seed_results: list[dict] = []
    total_start = time.perf_counter()

    for seed in args.seeds:
        print(f"\n{'=' * 40}")
        print(f"  SEED {seed}")
        print(f"{'=' * 40}")
        run_evaluation(
            split_limit=args.n,
            seed=seed,
            dry_run=args.dry_run,
            resume=args.resume,
        )
        # Collect per-seed result file
        result_path = REAL_FINAL_DIR / f"final_eval_seed_{seed}_n_{args.n or 'all'}.json"
        if result_path.exists() and not args.dry_run:
            with open(result_path, encoding="utf-8") as f:
                all_seed_results.append(json.load(f))

    if not args.dry_run and all_seed_results:
        _write_master_results(all_seed_results, args)

    elapsed = time.perf_counter() - total_start
    print(f"\n[OK] Final benchmark completed in {elapsed:.1f}s")
    if not args.dry_run:
        print(f"[OK] Results in {REAL_FINAL_DIR}")


def _write_master_results(seed_results: list[dict], args: argparse.Namespace) -> None:
    """Aggregate seed results into a single master_results.json."""
    from collections import defaultdict

    methods: dict[str, list[dict]] = defaultdict(list)
    for seed_data in seed_results:
        for method_row in seed_data.get("results", []):
            methods[method_row["method"]].append(method_row)

    aggregated = []
    for method_name, rows in methods.items():
        if not rows:
            continue
        acc_values = [r.get("accuracy", 0.0) for r in rows]
        token_values = [r.get("avg_tokens", 0.0) for r in rows]
        savings_values = [r.get("token_savings_pct", 0.0) for r in rows]
        n = len(acc_values)

        mean_acc = sum(acc_values) / n
        std_acc = (sum((x - mean_acc) ** 2 for x in acc_values) / max(n - 1, 1)) ** 0.5

        aggregated.append(
            {
                "method": method_name,
                "seeds": [r["seed"] if "seed" in r else None for r in rows],
                "accuracy_mean": round(mean_acc * 100, 2),
                "accuracy_std": round(std_acc * 100, 2),
                "avg_tokens_mean": round(sum(token_values) / n, 1),
                "token_savings_pct_mean": round(sum(savings_values) / n, 2),
                "n_seeds": n,
            }
        )

    master = {
        "execution_mode": "real",
        "seeds": args.seeds,
        "n_per_split": args.n,
        "aggregated": aggregated,
        "per_seed": seed_results,
    }

    master_path = REAL_FINAL_DIR / "master_results.json"
    tmp_path = master_path.with_name(".master_results.json.tmp")
    tmp_path.write_text(json.dumps(master, indent=2), encoding="utf-8")
    tmp_path.replace(master_path)
    print(f"[OK] Master results written to {master_path}")


if __name__ == "__main__":
    main()

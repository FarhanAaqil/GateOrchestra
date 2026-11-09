"""
scripts/rebase_calibration.py
=============================
Reuse a finished calibration after the dataset changed, instead of redoing it.

Use it when the test split grew and a few train/validation tasks were edited:

  * train and validation results of every UNCHANGED task are kept (they cost
    hours of model time and are still valid);
  * the units of the tasks named with --changed are dropped, so the next
    benchmark run repeats them on the new text;
  * all test-phase results are dropped and the checkpoint is pointed at the
    current test split;
  * the saved gate is cleared, so the next run retrains it on the updated
    validation results.

The original checkpoint is copied to logs/results/archive/ first. Then:

    python scripts/rebase_calibration.py --seed 42 --changed comp_008 cmp_002 cmp_016
    python scripts/final_benchmark.py --resume

It refuses to run if the checkpoint's train or validation task ids differ from
the dataset (that would make the kept results invalid).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_final_evaluation import CHECKPOINTS_DIR, _atomic_write_json  # noqa: E402
from shared.data_loader import load_split  # noqa: E402


def rebase(path: Path, changed: set[str]) -> dict[str, int]:
    state = json.loads(path.read_text(encoding="utf-8"))
    ids = {split: [t.task_id for t in load_split(split)] for split in ("train", "val", "test")}  # type: ignore[arg-type]
    saved = state["metadata"]["task_ids"]
    for split in ("train", "val"):
        if saved[split] != ids[split]:
            raise SystemExit(
                f"The checkpoint's {split} tasks differ from the dataset; cannot reuse it."
            )
    unknown = changed - set(ids["train"]) - set(ids["val"])
    if unknown:
        raise SystemExit(f"--changed names tasks that are not in train/val: {sorted(unknown)}")

    dropped = {"test": 0, "changed": 0}
    state["metadata"]["task_ids"]["test"] = ids["test"]
    for section in ("probes", "results"):
        for key in list(state[section]):
            phase, parts = key.split(":", 1)[0], key.split(":")
            task_id = parts[-2] if len(parts) >= 3 else ""
            if phase == "test":
                del state[section][key]
                dropped["test"] += 1
            elif task_id in changed:
                del state[section][key]
                dropped["changed"] += 1
    state["pending_random_decisions"] = {}
    state["gate_training"] = None
    _atomic_write_json(state, path)
    return {**dropped, "kept_results": len(state["results"]), "kept_probes": len(state["probes"])}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=42, help="the calibration seed's checkpoint")
    parser.add_argument(
        "--changed", nargs="*", default=[], help="train/val task ids whose text changed"
    )
    args = parser.parse_args()
    path = CHECKPOINTS_DIR / f"final_eval_seed_{args.seed}_n_all.json"
    if not path.exists():
        raise SystemExit(f"No checkpoint at {path}")
    archive = path.parent.parent / "archive"
    archive.mkdir(exist_ok=True)
    backup = archive / f"{path.stem}_before_rebase.json"
    if not backup.exists():
        shutil.copy2(path, backup)
    print(f"[backup] {backup}")
    for name, value in rebase(path, set(args.changed)).items():
        print(f"  {name}: {value}")
    print("[OK] checkpoint rebased; run: python scripts/final_benchmark.py --resume")


if __name__ == "__main__":
    main()

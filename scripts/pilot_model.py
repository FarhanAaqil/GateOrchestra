"""
scripts/pilot_model.py
======================
A quick look at whether a model leaves room for a gate, before paying for the full benchmark.

Runs CoT-SC-only and Always-MAS once on a few tasks (the validation split plus a handful of
training tasks, never the test split) and reports accuracy, tokens and the oracle ceiling:
the share of tasks where MAS is right and CoT-SC is wrong, which is all a perfect gate could
win. The model comes from the usual environment variables, so run one process per model:

    $env:GATE_LLM_PROVIDER = "groq"; $env:GROQ_MODEL_NAME = "openai/gpt-oss-20b"
    python scripts/pilot_model.py --n 40

Results go to logs/pilot/<model>.json only; nothing in results/ or configs/ is touched.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.baselines.always_mas_baseline import run_always_mas_baseline  # noqa: E402
from agents.baselines.cot_sc_baseline import run_cot_sc_baseline  # noqa: E402
from agents.orchestrator.orchestrator import MASOrchestrator  # noqa: E402
from shared.config import (  # noqa: E402
    ACTIVE_MODEL_NAME,
    LLM_PROVIDER,
    LOGS_DIR,
    MAS_ROUTER,
    PROBE_TOKEN_BUDGET,
)
from shared.data_loader import load_split  # noqa: E402
from shared.schemas import EvalResult, Task  # noqa: E402

ATTEMPTS = 3


def _run_one(task: Task) -> dict:
    """CoT-SC and MAS on one task; a provider failure is retried, then reported as an error."""
    row: dict = {"task_id": task.task_id}
    for method, runner in (
        ("cot", lambda: run_cot_sc_baseline(task)),
        (
            "mas",
            lambda: run_always_mas_baseline(
                task,
                orchestrator_fn=MASOrchestrator(default_strategy=MAS_ROUTER),
                token_budget=3 * PROBE_TOKEN_BUDGET,
            ),
        ),
    ):
        for attempt in range(1, ATTEMPTS + 1):
            try:
                result: EvalResult = runner()
                row[method] = {
                    "correct": bool(result.is_correct),
                    "tokens": int(result.tokens_spent),
                }
                break
            except Exception as exc:  # noqa: BLE001 - report and move on
                if attempt == ATTEMPTS:
                    row[method] = {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Pilot a model before the full benchmark")
    parser.add_argument("--n", type=int, default=40, help="tasks to run (val first, then train)")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    tasks = (load_split("val") + load_split("train"))[: args.n]
    print(f"Pilot: {ACTIVE_MODEL_NAME} via {LLM_PROVIDER}, {len(tasks)} tasks", flush=True)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_run_one, tasks))
    elapsed = time.perf_counter() - start

    ok = [r for r in rows if "correct" in r.get("cot", {}) and "correct" in r.get("mas", {})]
    errors = len(rows) - len(ok)
    n = len(ok)
    cot_ok = sum(r["cot"]["correct"] for r in ok)
    mas_ok = sum(r["mas"]["correct"] for r in ok)
    helps = sum(r["mas"]["correct"] and not r["cot"]["correct"] for r in ok)
    hurts = sum(r["cot"]["correct"] and not r["mas"]["correct"] for r in ok)
    cot_tok = sum(r["cot"]["tokens"] for r in ok) / max(n, 1)
    mas_tok = sum(r["mas"]["tokens"] for r in ok) / max(n, 1)
    summary = {
        "model": ACTIVE_MODEL_NAME,
        "provider": LLM_PROVIDER,
        "tasks": n,
        "errors": errors,
        "cot_sc_accuracy": round(cot_ok / max(n, 1), 3),
        "mas_accuracy": round(mas_ok / max(n, 1), 3),
        "mas_fixes_cot_failures": helps,
        "mas_breaks_cot_successes": hurts,
        "oracle_escalation_share": round(helps / max(n, 1), 3),
        "avg_tokens_cot_sc": round(cot_tok),
        "avg_tokens_mas": round(mas_tok),
        "seconds": round(elapsed),
    }
    out_dir = LOGS_DIR / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", ACTIVE_MODEL_NAME)
    (out_dir / f"{name}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

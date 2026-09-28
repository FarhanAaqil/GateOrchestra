"""scripts/demo.py
================
Canonical GateOrchestra Demonstration Runner.
Phase 6 of Implementation Plan / Phase 25 of Rescue Plan.

Supports explicit live evaluation against Groq and deterministic offline simulation.

Usage:
    python scripts/demo.py --mode live
    python scripts/demo.py --mode simulation
    python scripts/demo.py --mode live --scenario easy
    python scripts/demo.py --mode live --scenario hard
    python scripts/demo.py --mode live --scenario recovery
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCENARIO_TASKS = {
    "easy": ("arith_005", "val"),       # Easy arithmetic -> STOP (Probe answers correctly, saves tokens)
    "hard": ("comp_038", "test"),      # Multi-hop complex -> ESCALATE (Routes to MAS)
    "recovery": ("arith_012", "val"),   # Probe wrong -> Gate escalates -> MAS solves correctly
}


def print_banner(mode: str, model: str | None = None) -> None:
    width = 72
    print("=" * width)
    if mode == "live":
        print("  GATEORCHESTRA CAPSTONE DEMONSTRATION -- LIVE MODE")
        print("  MODE     : LIVE")
        print("  PROVIDER : GROQ")
        print(f"  MODEL    : {model or 'qwen/qwen3.8-27b'}")
        print("  REAL LLM CALLS ACTIVE -- TOKEN ACCOUNTING LIVE")
    else:
        print("  GATEORCHESTRA CAPSTONE DEMONSTRATION -- SIMULATION MODE")
        print("  MODE     : SIMULATION")
        print("  NO EXTERNAL API CALLS -- 100% OFFLINE SAFE DEMO")
    print("=" * width)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Canonical GateOrchestra Demonstration")
    parser.add_argument(
        "--mode",
        choices=["live", "simulation"],
        default="live",
        help="Demonstration mode: 'live' (queries Groq API) or 'simulation' (offline fail-safe)",
    )
    parser.add_argument(
        "--scenario",
        choices=["all", "easy", "hard", "recovery"],
        default="all",
        help="Pre-configured demonstration scenario: easy (STOP), hard (ESCALATE), recovery (MAS corrects probe)",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Optional specific task ID to run",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="MAS budget multiplier (default: 3)",
    )
    args = parser.parse_args()

    model_name = os.getenv("GROQ_MODEL_NAME", "qwen/qwen3.8-27b")
    print_banner(args.mode, model_name)

    underlying_mode = "live" if args.mode == "live" else "mock"
    demo_script = ROOT / "scripts" / "week8_demo.py"

    cmd = [
        sys.executable,
        str(demo_script),
        "--mode",
        underlying_mode,
        "--trace",
        "--k",
        str(args.k),
    ]

    if args.task_id:
        cmd.extend(["--task-id", args.task_id])
    elif args.scenario in SCENARIO_TASKS:
        task_id, split = SCENARIO_TASKS[args.scenario]
        cmd.extend(["--task-id", task_id, "--split", split])
    else:
        cmd.extend(["--n", "3"])

    proc = subprocess.run(cmd, cwd=str(ROOT))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())

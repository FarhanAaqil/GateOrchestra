"""
GateOrchestra API
-----------------
Connects the dashboard and external clients to the GateOrchestra pipeline.

Run from the repository root:
    uvicorn api.main:app --reload --port 8000

Simulation mode (``force_simulation`` or ``GATE_LLM_PROVIDER=mock``) exercises
the routing logic without an LLM. Simulated components never see the ground
truth, so simulated runs are returned ungraded (``is_correct = null``) and
flagged with ``"simulated": true``.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Make sure the repository root is available for imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.baselines.always_mas_baseline import run_always_mas_baseline  # noqa: E402
from agents.baselines.cot_sc_baseline import run_cot_sc_baseline  # noqa: E402
from agents.baselines.simulated_probe import SimulatedProbe  # noqa: E402
from agents.orchestrator.orchestrator import (  # noqa: E402
    get_default_mas_orchestrator,
    orchestrator,
)
from agents.probe_agent import probe_agent  # noqa: E402
from gate.classifier import GateClassifier  # noqa: E402
from gate.feature_extractor import extract_features  # noqa: E402
from gate.random_gate import RandomGate  # noqa: E402
from gate.rule_based_gate import RuleBasedGate  # noqa: E402
from integration.pipeline import run_pipeline  # noqa: E402
from shared.config import (  # noqa: E402
    BEST_MODEL_PATH,
    GROQ_MODEL_NAME,
    K_DEFAULT,
    LLM_PROVIDER,
    MODEL_NAME,
    MODELS_DIR,
    PROBE_TOKEN_BUDGET,
)
from shared.data_loader import load_split  # noqa: E402
from shared.schemas import ProbeResult, Task  # noqa: E402
from shared.token_logger import TokenAccountant  # noqa: E402

logger = logging.getLogger(__name__)

MASTER_RESULTS_PATH = ROOT / "results" / "real" / "final" / "master_results.json"
RANDOM_GATE_ESCALATION_RATE = 0.5

# Application & State

app = FastAPI(
    title="GateOrchestra API",
    description="API bridge for the GateOrchestra multi-agent gating pipeline",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory history ring buffer (last 50 runs)
_HISTORY: deque[dict[str, Any]] = deque(maxlen=50)

# Cached learned gate instance to avoid repeated disk reads
_CACHED_LEARNED_GATE: GateClassifier | None = None

# One RandomGate for the server's lifetime, so successive requests draw new
# coin flips (a gate rebuilt per request with a fixed seed always draws the same).
_RANDOM_GATE = RandomGate(escalation_rate=RANDOM_GATE_ESCALATION_RATE, seed=42)
_RANDOM_GATE_LOCK = threading.Lock()


def _gate_model_candidates() -> list[Path]:
    """Trained gates produced by real calibration runs, preferred first."""
    seed_gates = sorted(
        MODELS_DIR.glob("best_gate_seed_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return [BEST_MODEL_PATH, *seed_gates]


def get_trained_gate() -> GateClassifier:
    """Load the gate trained by the evaluation pipeline.

    Raises:
        HTTPException(503): If no trained gate exists. The API never trains a
            stand-in gate on made-up data.
    """
    global _CACHED_LEARNED_GATE
    if _CACHED_LEARNED_GATE is not None:
        return _CACHED_LEARNED_GATE

    for candidate in _gate_model_candidates():
        if not candidate.exists():
            continue
        try:
            loaded = GateClassifier.load(candidate)
        except Exception as err:
            logger.warning(f"Could not load gate from {candidate}: {err}")
            continue
        if getattr(loaded, "_is_trained", False):
            _CACHED_LEARNED_GATE = loaded
            logger.info(f"Loaded trained gate from {candidate}")
            return loaded

    raise HTTPException(
        status_code=503,
        detail=(
            "No trained gate is available. Run `python scripts/final_benchmark.py` "
            f"to calibrate one (expected under {MODELS_DIR})."
        ),
    )


class _SharedRandomGate:
    """Thread-safe view of the server-wide RandomGate."""

    def predict(self, features: Any, k: int, probe_tokens: int) -> Any:
        with _RANDOM_GATE_LOCK:
            return _RANDOM_GATE.predict(features, k=k, probe_tokens=probe_tokens)


# Request / Response Models


class RunRequest(BaseModel):
    task_id: str = Field(..., description="Unique task identifier")
    question: str = Field(..., description="Question/task to solve")
    context: str | None = Field(default=None, description="Optional context")
    ground_truth: str | None = Field(
        default=None,
        description="Optional answer used for evaluation (ignored in simulation mode)",
    )
    method: str = Field(
        default="GateOrchestra",
        description="Method or Gate: GateOrchestra, RuleBasedGate, RandomGate, Always-MAS, CoT-SC",
    )
    k: int = Field(
        default=K_DEFAULT,
        ge=1,
        description="MAS token budget multiplier",
    )
    force_simulation: bool = Field(
        default=False,
        description="Use the offline simulator instead of a live LLM. Results are not graded.",
    )


# Health & Discovery Endpoints


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "GateOrchestra API",
        "status": "running",
        "message": "GateOrchestra multi-agent gating pipeline is online",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "pipeline": "available",
    }


@app.get("/models")
def list_models() -> dict[str, Any]:
    """Return available gate strategies, baselines, and provider status."""
    trained_available = any(path.exists() for path in _gate_model_candidates())

    methods = [
        {
            "id": "GateOrchestra",
            "name": "GateOrchestra (learned gate)",
            "type": "learned_gate",
            "description": "Gate selected by validation calibration in the final benchmark.",
            "available": trained_available,
        },
        {
            "id": "RuleBasedGate",
            "name": "Rule-Based Gate",
            "type": "heuristic_gate",
            "description": "Heuristic gate using consistency score and question complexity thresholds.",
            "available": True,
        },
        {
            "id": "RandomGate",
            "name": "Random Gate",
            "type": "baseline_gate",
            "description": (
                f"Coin-flip baseline escalating {RANDOM_GATE_ESCALATION_RATE:.0%} of tasks. "
                "In the benchmark its rate is matched to the learned gate's."
            ),
            "available": True,
        },
        {
            "id": "Always-MAS",
            "name": "Always-MAS",
            "type": "full_orchestrator",
            "description": "Bypasses gating; routes every task directly to the full multi-agent system.",
            "available": True,
        },
        {
            "id": "CoT-SC",
            "name": "CoT-SC Only",
            "type": "single_agent",
            "description": "Single-agent baseline using self-consistency majority voting.",
            "available": True,
        },
    ]

    return {
        "methods": methods,
        "providers": ["ollama", "groq", "mock"],
        "current_provider": LLM_PROVIDER,
        "default_model": GROQ_MODEL_NAME if LLM_PROVIDER == "groq" else MODEL_NAME,
        "trained_gate_available": trained_available,
    }


# Task Exploration Endpoints


@app.get("/tasks")
def list_tasks(
    split: str = Query(default="val", description="Split name: train, val, or test"),
    limit: int = Query(default=20, ge=1, le=100, description="Number of tasks to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    search: str | None = Query(default=None, description="Optional search term in questions"),
) -> dict[str, Any]:
    """Browse and filter MASBench-mini tasks."""
    if split not in ("train", "val", "test"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid split '{split}'. Must be one of: train, val, test",
        )

    try:
        tasks = load_split(split)  # type: ignore[arg-type]
    except Exception as err:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load split '{split}': {err}",
        ) from err

    if search:
        q = search.lower()
        tasks = [
            t for t in tasks if q in t.question.lower() or (t.context and q in t.context.lower())
        ]

    total = len(tasks)
    sliced = tasks[offset : offset + limit]

    return {
        "split": split,
        "total": total,
        "limit": limit,
        "offset": offset,
        "tasks": [t.model_dump(mode="json") for t in sliced],
    }


# Execution History Endpoints


@app.get("/history")
def get_history() -> dict[str, Any]:
    """Retrieve recent task execution results (up to 50 runs)."""
    return {
        "count": len(_HISTORY),
        "history": list(_HISTORY),
    }


@app.delete("/history")
def clear_history() -> dict[str, str]:
    """Clear in-memory execution history."""
    _HISTORY.clear()
    return {"status": "cleared"}


# Benchmark Results Endpoint


@app.get("/demo/summary")
def get_demo_summary() -> dict[str, Any]:
    """Return the recorded real benchmark results and whether they pass verification.

    Nothing here is hardcoded: the numbers come from
    results/real/final/master_results.json, checked by
    scripts/verify_final_results.py.
    """
    if not MASTER_RESULTS_PATH.exists():
        return {
            "status": "no_results",
            "message": "No real benchmark results yet. Run scripts/final_benchmark.py.",
            "aggregated": [],
        }

    from scripts.verify_final_results import verify_master_results  # noqa: PLC0415

    data = json.loads(MASTER_RESULTS_PATH.read_text(encoding="utf-8"))
    errors, partial = verify_master_results(MASTER_RESULTS_PATH, allow_partial=True)
    if errors:
        status = "unverified"
    elif partial:
        status = "partial"
    else:
        status = "verified"

    return {
        "status": status,
        "verification_issues": errors or partial,
        "seeds": data.get("seeds", []),
        "n_per_split": data.get("n_per_split"),
        "aggregated": data.get("aggregated", []),
        "pooled_paired_tests": data.get("pooled_paired_tests", []),
    }


# Run Pipeline / Execution


def _simulated_orchestrator_call(task: Task, budget: int) -> tuple[str, int]:
    """Offline stand-in for the MAS. It has no model, so it has no answer."""
    tokens = min(budget, max(120, len(task.question.split()) * 12))
    return "[simulated: no LLM answer]", tokens


def _respond(data: dict[str, Any], simulated: bool) -> dict[str, Any]:
    data["simulated"] = simulated
    _HISTORY.appendleft(data)
    return data


def _live_failure(stage: str, err: Exception) -> HTTPException:
    logger.error(f"Live {stage} failed: {type(err).__name__}: {err}")
    return HTTPException(
        status_code=503,
        detail=f"Live {stage} failed: {type(err).__name__}: {err}. "
        "Use force_simulation=true for offline demo mode.",
    )


@app.post("/run")
def run_gateorchestra(request: RunRequest) -> dict[str, Any]:
    """Run one task through the requested method / gate pipeline."""
    use_sim = request.force_simulation or (LLM_PROVIDER == "mock")
    task = Task(
        task_id=request.task_id,
        question=request.question,
        context=request.context,
        # Simulated components must never see the answer; simulated runs are ungraded.
        ground_truth=None if use_sim else request.ground_truth,
    )

    accountant = TokenAccountant()
    method_name = request.method.strip()

    probe_fn = SimulatedProbe(seed=42).run if use_sim else probe_agent
    orch_fn = _simulated_orchestrator_call if use_sim else orchestrator

    # 1. Baseline: CoT-SC only
    if method_name in ("CoT-SC", "CoT-SC-only"):
        try:
            result = run_cot_sc_baseline(task, probe_fn=probe_fn, accountant=accountant)
        except Exception as err:
            if use_sim:
                raise
            raise _live_failure("CoT-SC", err) from err
        return _respond(result.model_dump(mode="json"), use_sim)

    # 2. Baseline: Always-MAS
    if method_name == "Always-MAS":
        try:
            result = run_always_mas_baseline(
                task,
                orchestrator_fn=orch_fn,
                accountant=accountant,
                token_budget=request.k * PROBE_TOKEN_BUDGET,
            )
        except Exception as err:
            if use_sim:
                raise
            raise _live_failure("Always-MAS", err) from err
        return _respond(result.model_dump(mode="json"), use_sim)

    # 3. Gating Methods
    gate: Any
    if method_name == "RuleBasedGate":
        gate = RuleBasedGate()
    elif method_name == "RandomGate":
        gate = _SharedRandomGate()
    elif method_name == "GateOrchestra":
        gate = get_trained_gate()
    else:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown method '{method_name}'. Supported: "
                "GateOrchestra, RuleBasedGate, RandomGate, Always-MAS, CoT-SC"
            ),
        )

    probes: list[ProbeResult] = []

    def recording_probe(current: Task) -> ProbeResult:
        probe = probe_fn(current)
        probes.append(probe)
        return probe

    try:
        result = run_pipeline(
            task=task,
            gate=gate,
            probe_agent=recording_probe,
            orchestrator=orch_fn,
            accountant=accountant,
            k=request.k,
            method=method_name,
            mas_orchestrator=None if use_sim else get_default_mas_orchestrator(),
        )
    except Exception as err:
        if use_sim:
            raise
        raise _live_failure("pipeline", err) from err
    data = result.model_dump(mode="json")
    # The exact features the gate saw (extraction is deterministic).
    data["features"] = extract_features(task, probes[-1]).model_dump(mode="json")
    return _respond(data, use_sim)

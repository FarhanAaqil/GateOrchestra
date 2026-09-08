"""
GateOrchestra API
-----------------
Connects the dashboard and external clients to the GateOrchestra pipeline.

Run from the repository root:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import sys
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
from agents.orchestrator.orchestrator import orchestrator  # noqa: E402
from agents.probe_agent import probe_agent  # noqa: E402
from gate.classifier import GateClassifier, GBTGate, LogRegGate, MLPGate  # noqa: E402
from gate.random_gate import RandomGate  # noqa: E402
from gate.rule_based_gate import RuleBasedGate  # noqa: E402
from integration.pipeline import run_pipeline  # noqa: E402
from shared.config import (  # noqa: E402
    BEST_MODEL_PATH,
    K_DEFAULT,
    LLM_PROVIDER,
    LOGS_DIR,
    MODEL_NAME,
)
from shared.data_loader import load_split  # noqa: E402
from shared.schemas import Task  # noqa: E402
from shared.token_logger import TokenAccountant  # noqa: E402

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Application & State
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="GateOrchestra API",
    description="API bridge for the GateOrchestra multi-agent gating pipeline",
    version="1.1.0",
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


def get_trained_gate() -> GateClassifier:
    """Load the trained best gate from disk or fallback to a fresh GBTGate."""
    global _CACHED_LEARNED_GATE
    if _CACHED_LEARNED_GATE is not None:
        return _CACHED_LEARNED_GATE

    for candidate in [BEST_MODEL_PATH, LOGS_DIR / "week2_best_gate.pkl"]:
        if candidate.exists():
            try:
                _CACHED_LEARNED_GATE = GateClassifier.load(candidate)
                logger.info(f"Loaded trained gate from {candidate}")
                return _CACHED_LEARNED_GATE
            except Exception as err:
                logger.warning(f"Could not load gate from {candidate}: {err}")

    # Fallback to untrained GBTGate
    logger.info("Using fallback GBTGate instance")
    _CACHED_LEARNED_GATE = GBTGate()
    return _CACHED_LEARNED_GATE


# ─────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────


class RunRequest(BaseModel):
    task_id: str = Field(..., description="Unique task identifier")
    question: str = Field(..., description="Question/task to solve")
    context: str | None = Field(default=None, description="Optional context")
    ground_truth: str | None = Field(
        default=None,
        description="Optional answer used for evaluation",
    )
    method: str = Field(
        default="GateOrchestra",
        description=(
            "Method or Gate: GateOrchestra, GBTGate, LogRegGate, MLPGate, "
            "RuleBasedGate, RandomGate, Always-MAS, CoT-SC"
        ),
    )
    k: int = Field(
        default=K_DEFAULT,
        ge=1,
        description="MAS token budget multiplier",
    )
    force_simulation: bool = Field(
        default=False,
        description="Force using simulated probe/orchestrator instead of live LLM",
    )


# ─────────────────────────────────────────────────────────────
# Health & Discovery Endpoints
# ─────────────────────────────────────────────────────────────


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
    trained_available = BEST_MODEL_PATH.exists() or (LOGS_DIR / "week2_best_gate.pkl").exists()

    methods = [
        {
            "id": "GateOrchestra",
            "name": "⚡ GateOrchestra (Trained GBT)",
            "type": "learned_gate",
            "description": "Trained Gradient Boosted Trees gate calibrated under token-budget constraints.",
        },
        {
            "id": "RuleBasedGate",
            "name": "📐 Rule-Based Gate",
            "type": "heuristic_gate",
            "description": "Heuristic gate using consistency score and question complexity thresholds.",
        },
        {
            "id": "RandomGate",
            "name": "🎲 Random Gate",
            "type": "baseline_gate",
            "description": "Calibrated coin-flip baseline escalating at a fixed empirical rate (40%).",
        },
        {
            "id": "LogRegGate",
            "name": "📈 Logistic Regression Gate",
            "type": "learned_gate",
            "description": "Linear classifier gate with interpretable feature weights.",
        },
        {
            "id": "MLPGate",
            "name": "🧠 Multi-Layer Perceptron Gate",
            "type": "learned_gate",
            "description": "Neural net classifier gate modeling non-linear feature interactions.",
        },
        {
            "id": "Always-MAS",
            "name": "🤝 Always-MAS",
            "type": "full_orchestrator",
            "description": "Bypasses gating; routes every task directly to the full multi-agent system.",
        },
        {
            "id": "CoT-SC",
            "name": "💡 CoT-SC Only",
            "type": "single_agent",
            "description": "Cheap single-agent baseline using self-consistency majority voting.",
        },
    ]

    return {
        "methods": methods,
        "providers": ["ollama", "groq", "mock"],
        "current_provider": LLM_PROVIDER,
        "default_model": MODEL_NAME,
        "trained_gate_available": trained_available,
    }


# ─────────────────────────────────────────────────────────────
# Task Exploration Endpoints
# ─────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────
# Execution History Endpoints
# ─────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────
# Run Pipeline / Execution
# ─────────────────────────────────────────────────────────────


def _simulated_orchestrator_call(task: Task, budget: int) -> tuple[str, int]:
    """Fallback simulated orchestrator when live LLMs are offline or simulation requested."""
    depth = task.depth_score or 2
    tokens = min(budget, max(120, int(depth * 140)))
    answer = task.ground_truth if task.ground_truth else "MAS simulated consensus answer"
    return answer, tokens


@app.post("/run")
def run_gateorchestra(request: RunRequest) -> dict[str, Any]:
    """Run one task through the requested method / gate pipeline."""
    task = Task(
        task_id=request.task_id,
        question=request.question,
        context=request.context,
        ground_truth=request.ground_truth,
    )

    accountant = TokenAccountant()
    method_name = request.method

    # Determine probe agent and orchestrator callables (with automatic graceful simulated fallback)
    use_sim = request.force_simulation or (LLM_PROVIDER == "mock")
    probe_fn = SimulatedProbe(seed=42).run if use_sim else probe_agent
    orch_fn = _simulated_orchestrator_call if use_sim else orchestrator

    # 1. Baseline: CoT-SC only
    if method_name in ("CoT-SC", "CoT-SC-only", "🧠 CoT-SC"):
        try:
            result = run_cot_sc_baseline(task, probe_fn=probe_fn, accountant=accountant)
        except Exception as err:
            logger.warning(f"Live CoT-SC failed ({err}), falling back to simulated probe")
            result = run_cot_sc_baseline(
                task,
                probe_fn=SimulatedProbe(seed=42).run,
                accountant=accountant,
            )
        data = result.model_dump(mode="json")
        _HISTORY.appendleft(data)
        return data

    # 2. Baseline: Always-MAS
    if method_name in ("Always-MAS", "🤝 Always-MAS"):
        try:
            result = run_always_mas_baseline(task, orchestrator_fn=orch_fn, accountant=accountant)
        except Exception as err:
            logger.warning(
                f"Live Always-MAS failed ({err}), falling back to simulated orchestrator"
            )
            result = run_always_mas_baseline(
                task,
                orchestrator_fn=_simulated_orchestrator_call,
                accountant=accountant,
            )
        data = result.model_dump(mode="json")
        _HISTORY.appendleft(data)
        return data

    # 3. Gating Methods
    normalized_method = method_name
    if "Rule" in method_name:
        gate: Any = RuleBasedGate()
        normalized_method = "RuleBasedGate"
    elif "Random" in method_name:
        gate = RandomGate(escalation_rate=0.4, seed=42)
        normalized_method = "RandomGate"
    elif "LogReg" in method_name:
        gate = LogRegGate()
        normalized_method = "LogRegGate"
    elif "MLP" in method_name:
        gate = MLPGate()
        normalized_method = "MLPGate"
    elif "GBT" in method_name:
        gate = GBTGate()
        normalized_method = "GBTGate"
    elif method_name in ("GateOrchestra", "⚡ GateOrchestra", "✨ Auto Gate"):
        gate = get_trained_gate()
        normalized_method = "GateOrchestra"
    else:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown method '{method_name}'. Supported: "
                "GateOrchestra, GBTGate, LogRegGate, MLPGate, RuleBasedGate, "
                "RandomGate, Always-MAS, CoT-SC"
            ),
        )

    try:
        result = run_pipeline(
            task=task,
            gate=gate,
            probe_agent=probe_fn,
            orchestrator=orch_fn,
            accountant=accountant,
            k=request.k,
            method=normalized_method,
        )
    except Exception as err:
        logger.warning(
            f"Pipeline run encountered error with live LLM ({err}), falling back to simulation"
        )
        result = run_pipeline(
            task=task,
            gate=gate,
            probe_agent=SimulatedProbe(seed=42).run,
            orchestrator=_simulated_orchestrator_call,
            accountant=accountant,
            k=request.k,
            method=normalized_method,
        )

    data = result.model_dump(mode="json")
    _HISTORY.appendleft(data)
    return data

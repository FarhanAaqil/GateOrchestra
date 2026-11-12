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
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

# Make sure the repository root is available for imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.baselines.always_mas_baseline import run_always_mas_baseline  # noqa: E402
from agents.baselines.cot_sc_baseline import run_cot_sc_baseline  # noqa: E402
from agents.baselines.simulated_probe import SimulatedProbe  # noqa: E402
from agents.orchestrator.orchestrator import request_orchestrator  # noqa: E402
from agents.probe_agent import extract_answer, probe_agent  # noqa: E402
from api.chat_routes import router as chat_router  # noqa: E402
from api.history import RunHistory  # noqa: E402
from api.security import SecurityMiddleware  # noqa: E402
from evaluation.results_detail import build_results_detail  # noqa: E402
from gate.classifier import (  # noqa: E402
    FEATURE_NAMES,
    GateClassifier,
    GBTGate,
    LogRegGate,
    benchmark_gate_paths,
    gate_provenance,
    load_provenanced_gate,
)
from gate.feature_extractor import extract_features  # noqa: E402
from gate.random_gate import RandomGate  # noqa: E402
from gate.rule_based_gate import RuleBasedGate  # noqa: E402
from integration.pipeline import run_pipeline  # noqa: E402
from shared.config import (  # noqa: E402
    BEST_MODEL_PATH,
    GROQ_MODEL_NAME,
    K_DEFAULT,
    K_VALUES,
    LLM_PROVIDER,
    LOGS_DIR,
    MODEL_NAME,
    MODELS_DIR,
    PROBE_TOKEN_BUDGET,
)
from shared.data_loader import ANSWER_ALIASES_PATH, DATASET_DIR, load_split  # noqa: E402
from shared.schemas import ProbeResult, Task  # noqa: E402
from shared.token_logger import TokenAccountant  # noqa: E402

logger = logging.getLogger(__name__)

MASTER_RESULTS_PATH = ROOT / "results" / "real" / "final" / "master_results.json"
RESULTS_ROOT = ROOT / "results" / "real"
RUN_LABELS = {
    "final": "Qwen2.5-3B, main benchmark (2 seeds)",
    "groq-qwen3-27b": "Qwen3.8-27B, reduced run (1 seed)",
    "groq-gpt-oss-120b": "gpt-oss-120b, reduced run (1 seed)",
}
RANDOM_GATE_ESCALATION_RATE = 0.5

# Application & State


def _warm_up_assistant() -> None:
    """Start loading the chat's question analyser so the first message is not slow."""
    if os.getenv("ASSISTANT_WARMUP", "1") != "0":
        from assistant.service import get_service

        get_service().warm_up()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _warm_up_assistant()
    yield


app = FastAPI(
    lifespan=_lifespan,
    title="GateOrchestra API",
    description="API bridge for the GateOrchestra multi-agent gating pipeline",
    version="1.2.0",
)

# Middleware runs outermost-last: gzip and CORS wrap the security layer, so even
# a refused request carries CORS headers the browser can read.
app.add_middleware(SecurityMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("GATE_CORS_ORIGINS", "*").split(",") if o.strip()],
    allow_credentials=False,  # tokens travel in a header, not cookies
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(chat_router)

# The last 50 runs, persisted so they survive a restart.
_HISTORY = RunHistory(LOGS_DIR / "api_history.jsonl", limit=50)

# Parsed files keyed by (path, mtime): rebuilt only when the file changes.
_FILE_CACHE: dict[str, tuple[tuple[tuple[str, float], ...], Any]] = {}
_FILE_CACHE_LOCK = threading.Lock()


def _cached(name: str, paths: list[Path], build: Any) -> Any:
    """Return ``build()``, recomputed only when one of ``paths`` changes."""
    key = tuple((str(p), p.stat().st_mtime if p.exists() else -1.0) for p in paths)
    with _FILE_CACHE_LOCK:
        hit = _FILE_CACHE.get(name)
        if hit is not None and hit[0] == key:
            return hit[1]
    value = build()
    with _FILE_CACHE_LOCK:
        _FILE_CACHE[name] = (key, value)
    return value


def _split_paths(split: str) -> list[Path]:
    return [DATASET_DIR / split / f"{split}.jsonl", ANSWER_ALIASES_PATH]


def _load_split_cached(split: str) -> list[Task]:
    return _cached(f"split:{split}", _split_paths(split), lambda: load_split(split))  # type: ignore[arg-type]


def available_runs() -> list[str]:
    """Benchmark runs with results on disk: the main run first, then tagged runs."""
    runs = ["final"] if MASTER_RESULTS_PATH.exists() else []
    if RESULTS_ROOT.exists():
        for child in sorted(RESULTS_ROOT.iterdir()):
            if child.name != "final" and (child / "master_results.json").exists():
                runs.append(child.name)
    return runs


def _run_path(run: str | None) -> Path:
    """The master_results.json of ``run``; only runs found on disk are accepted."""
    if not run or run == "final":
        return MASTER_RESULTS_PATH
    if run not in available_runs():
        raise HTTPException(status_code=404, detail=f"No benchmark run called {run!r}.")
    return RESULTS_ROOT / run / "master_results.json"


def _verified_master(run: str | None = None) -> tuple[dict[str, Any], list[str], list[str]]:
    """A run's master_results.json with its verification outcome (cached until it changes)."""
    from scripts.verify_final_results import verify_master_results  # noqa: PLC0415

    path = _run_path(run)

    def build() -> tuple[dict[str, Any], list[str], list[str]]:
        data = json.loads(path.read_text(encoding="utf-8"))
        errors, partial = verify_master_results(path, allow_partial=True)
        return data, errors, partial

    return _cached(f"master:{path}", [path], build)


# Cached learned gate instance to avoid repeated disk reads
_CACHED_LEARNED_GATE: GateClassifier | None = None
_CACHED_GATE_INFO: dict[str, Any] | None = None
_CACHED_GATE_KEY: tuple[str, float] | None = None

# One RandomGate for the server's lifetime, so successive requests draw new
# coin flips (a gate rebuilt per request with a fixed seed always draws the same).
_RANDOM_GATE = RandomGate(escalation_rate=RANDOM_GATE_ESCALATION_RATE, seed=42)
_RANDOM_GATE_LOCK = threading.Lock()


def _gate_model_candidates() -> list[Path]:
    """Gates written by benchmark calibration, newest first."""
    return [*benchmark_gate_paths(MODELS_DIR), BEST_MODEL_PATH]


def _load_provenanced_gate() -> tuple[GateClassifier, dict[str, Any]] | None:
    """Newest trained gate with known provenance (cached until a newer file lands)."""
    global _CACHED_LEARNED_GATE, _CACHED_GATE_INFO, _CACHED_GATE_KEY
    for candidate in _gate_model_candidates():
        if not candidate.exists():
            continue
        key = (str(candidate), candidate.stat().st_mtime)
        if key == _CACHED_GATE_KEY and _CACHED_LEARNED_GATE is not None:
            return _CACHED_LEARNED_GATE, _CACHED_GATE_INFO or {}
        loaded = load_provenanced_gate(candidate)
        if loaded is None:
            continue
        info = gate_provenance(candidate, loaded)
        if candidate.is_relative_to(ROOT):
            info["path"] = str(candidate.relative_to(ROOT))
        _CACHED_LEARNED_GATE, _CACHED_GATE_INFO, _CACHED_GATE_KEY = loaded, info, key
        logger.info(f"Loaded trained gate from {candidate}")
        return loaded, info
    return None


def get_trained_gate() -> GateClassifier:
    """Load the newest gate trained by the benchmark's calibration.

    Raises:
        HTTPException(503): If no gate with known provenance exists. The API
            never trains a stand-in gate on made-up data.
    """
    if _CACHED_LEARNED_GATE is not None and _CACHED_GATE_KEY is None:
        return _CACHED_LEARNED_GATE  # injected directly (tests)
    found = _load_provenanced_gate()
    if found is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained gate is available. Run `python scripts/final_benchmark.py` "
                f"to calibrate one (expected under {MODELS_DIR})."
            ),
        )
    return found[0]


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
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional ESCALATE probability threshold for the learned gate "
        "(other gates ignore it). Omit to use the classifier's own decision.",
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
    found = _load_provenanced_gate()
    gate_info = found[1] if found else None
    trained_available = gate_info is not None

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
        "trained_gate": gate_info,
        "probe_token_budget": PROBE_TOKEN_BUDGET,
        "k_values": K_VALUES,
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
        tasks = _load_split_cached(split)
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
    runs = _HISTORY.snapshot()
    return {"count": len(runs), "history": runs}


@app.delete("/history")
def clear_history() -> dict[str, str]:
    """Clear the run history (memory and disk)."""
    _HISTORY.clear()
    return {"status": "cleared"}


# Benchmark Results Endpoint


@app.get("/results/runs")
def list_runs() -> dict[str, Any]:
    """The benchmark runs the Results page can show."""
    runs = []
    for run in available_runs():
        data = json.loads(_run_path(run).read_text(encoding="utf-8"))
        runs.append(
            {
                "id": run,
                "label": RUN_LABELS.get(run, run),
                "seeds": data.get("seeds", []),
                "main": run == "final",
            }
        )
    return {"runs": runs}


@app.get("/demo/summary")
def get_demo_summary(run: str | None = Query(default=None, max_length=80)) -> dict[str, Any]:
    """Return the recorded real benchmark results and whether they pass verification.

    Nothing here is hardcoded: the numbers come from
    results/real/final/master_results.json, checked by
    scripts/verify_final_results.py.
    """
    if not _run_path(run).exists():
        return {
            "status": "no_results",
            "message": "No real benchmark results yet. Run scripts/final_benchmark.py.",
            "aggregated": [],
        }

    data, errors, partial = _verified_master(run)
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
    _HISTORY.add(data)
    return data


def _live_failure(stage: str, err: Exception) -> HTTPException:
    logger.error(f"Live {stage} failed: {type(err).__name__}: {err}")
    return HTTPException(
        status_code=503,
        detail=f"Live {stage} failed: {type(err).__name__}: {err}. "
        "Use force_simulation=true for offline demo mode.",
    )


def _probe_details(probe: ProbeResult) -> dict[str, Any]:
    """What the CoT-SC probe saw: the vote, its agreement, and each sample's answer."""
    return {
        "answer": probe.answer,
        "consistency_score": probe.consistency_score,
        "tokens_used": probe.tokens_used,
        "latency_ms": probe.latency_ms,
        "samples": [
            output if output.startswith("Error:") else extract_answer(output)
            for output in probe.raw_outputs
        ],
    }


def _mas_trace(mas: Any, mas_tokens: int | None) -> list[dict[str, Any]]:
    """Steps the chosen sub-agent took in this request (empty when MAS did not run)."""
    if mas is None or not mas_tokens:
        return []
    return list(mas._last_trace)


@app.post("/run")
def run_gateorchestra(request: RunRequest) -> dict[str, Any]:
    """Run one task through the requested method / gate pipeline.

    Besides the EvalResult fields the response carries ``probe`` (the CoT-SC
    vote and per-sample answers, when a probe ran), ``features`` (gated
    methods), ``trace`` (each sub-agent step, when MAS ran live) and
    ``simulated``.
    """
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

    base_probe_fn = SimulatedProbe(seed=42).run if use_sim else probe_agent
    # A per-request MAS, so concurrent requests never read each other's trace.
    mas = None if use_sim else request_orchestrator()
    orch_fn = _simulated_orchestrator_call if mas is None else mas
    probes: list[ProbeResult] = []

    def probe_fn(current: Task) -> ProbeResult:
        probe = base_probe_fn(current)
        probes.append(probe)
        return probe

    # 1. Baseline: CoT-SC only
    if method_name in ("CoT-SC", "CoT-SC-only"):
        try:
            result = run_cot_sc_baseline(task, probe_fn=probe_fn, accountant=accountant)
        except Exception as err:
            if use_sim:
                raise
            raise _live_failure("CoT-SC", err) from err
        data = result.model_dump(mode="json")
        data["probe"] = _probe_details(probes[-1])
        data["trace"] = []
        return _respond(data, use_sim)

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
        data = result.model_dump(mode="json")
        if mas is not None:
            data["mas_strategy"] = mas._last_strategy
        data["probe"] = None
        data["trace"] = _mas_trace(mas, result.mas_tokens)
        return _respond(data, use_sim)

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

    try:
        result = run_pipeline(
            task=task,
            gate=gate,
            probe_agent=probe_fn,
            orchestrator=orch_fn,
            accountant=accountant,
            k=request.k,
            method=method_name,
            threshold=request.threshold,
            mas_orchestrator=mas,
        )
    except Exception as err:
        if use_sim:
            raise
        raise _live_failure("pipeline", err) from err
    data = result.model_dump(mode="json")
    # The exact features the gate saw (extraction is deterministic).
    data["features"] = extract_features(task, probes[-1]).model_dump(mode="json")
    data["probe"] = _probe_details(probes[-1])
    data["trace"] = _mas_trace(mas, result.mas_tokens)
    return _respond(data, use_sim)


# Gate & Results Detail Endpoints


@app.get("/gate/importance")
def gate_importance() -> dict[str, Any]:
    """Feature importances of the loaded learned gate, when its classifier has them.

    GBT gates report impurity-based importances; logistic-regression gates
    report normalised absolute coefficients (on standardised features).
    """
    found = _load_provenanced_gate()
    if found is None:
        return {
            "available": False,
            "reason": "No trained gate yet.",
            "gate": None,
            "importances": [],
        }
    gate, info = found
    values: list[float] | None = None
    if isinstance(gate, GBTGate) and gate._is_trained:
        values = [float(v) for v in gate._model.feature_importances_]
    elif isinstance(gate, LogRegGate) and gate._is_trained:
        coefs = [abs(float(c)) for c in gate._model.coef_[0]]
        total = sum(coefs) or 1.0
        values = [c / total for c in coefs]
    if values is None:
        return {
            "available": False,
            "reason": f"{info['classifier']} has no feature importances.",
            "gate": info,
            "importances": [],
        }
    return {
        "available": True,
        "gate": info,
        "importances": sorted(
            ({"feature": f, "importance": v} for f, v in zip(FEATURE_NAMES, values, strict=True)),
            key=lambda row: row["importance"],
            reverse=True,
        ),
    }


@app.get("/results/detail")
def results_detail(run: str | None = Query(default=None, max_length=80)) -> dict[str, Any]:
    """Per-task outcomes and derived views of the recorded benchmark results."""
    empty: dict[str, list[Any]] = {
        "per_task": [],
        "axis_cells": [],
        "escalation_quality": [],
        "k_sweep": [],
        "gates": [],
    }
    path = _run_path(run)
    if not path.exists():
        return {"status": "no_results", **empty}
    master, errors, partial = _verified_master(run)
    status = "unverified" if errors else "partial" if partial else "verified"

    def build() -> dict[str, Any]:
        labels = {
            t.task_id: t.model_dump(include={"depth_score", "parallel_score", "source_dataset"})
            for split in ("train", "val", "test")
            for t in _load_split_cached(split)
        }
        return build_results_detail(master, labels)

    paths = [path, *(p for s in ("train", "val", "test") for p in _split_paths(s))]
    return {"status": status, **_cached(f"results_detail:{path}", paths, build)}

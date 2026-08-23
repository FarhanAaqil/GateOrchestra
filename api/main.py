"""
GateOrchestra API
-----------------
Connects the dashboard to the existing GateOrchestra pipeline.

Run from the repository root:
    uvicorn api.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

import sys
from pathlib import Path

# Make sure the repository root is available for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from shared.config import K_DEFAULT
from shared.schemas import Task
from shared.token_logger import TokenAccountant

from gate.rule_based_gate import RuleBasedGate
from gate.random_gate import RandomGate
from integration.pipeline import run_pipeline

from tests.mocks.mock_probe_agent import mock_probe_agent
from tests.mocks.mock_orchestrator import mock_orchestrator


# ─────────────────────────────────────────────────────────────
# FastAPI application
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="GateOrchestra API",
    description="API bridge for the GateOrchestra multi-agent gating pipeline",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# Request model
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
        default="RuleBasedGate",
        description="Gate method: RuleBasedGate or RandomGate",
    )

    k: int = Field(
        default=K_DEFAULT,
        ge=1,
        description="MAS token budget multiplier",
    )


# ─────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "GateOrchestra API",
        "status": "running",
        "message": "Real pipeline API is online",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "pipeline": "available",
    }


# ─────────────────────────────────────────────────────────────
# Run GateOrchestra
# ─────────────────────────────────────────────────────────────

@app.post("/run")
def run_gateorchestra(request: RunRequest):
    """
    Run one task through the existing GateOrchestra pipeline.
    """

    # Create the Task using the shared contract
    task = Task(
        task_id=request.task_id,
        question=request.question,
        context=request.context,
        ground_truth=request.ground_truth,
    )

    # Select the gate
    if request.method == "RuleBasedGate":
        gate = RuleBasedGate()

    elif request.method == "RandomGate":
        gate = RandomGate(
            escalation_rate=0.4,
            seed=42,
        )

    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unknown method. "
                "Currently supported: RuleBasedGate, RandomGate"
            ),
        )

    # New accountant for this API request
    accountant = TokenAccountant()

    # Run the EXISTING GateOrchestra pipeline
    result = run_pipeline(
        task=task,
        gate=gate,
        probe_agent=mock_probe_agent,
        orchestrator=mock_orchestrator,
        accountant=accountant,
        k=request.k,
        method=request.method,
    )

    # Return the real EvalResult as JSON
    return result.model_dump(mode="json")
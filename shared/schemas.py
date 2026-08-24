"""
shared/schemas.py
=================
The FROZEN contract layer for GateOrchestra.

All 4 team members code against these types.  No one should change these
models without a team-wide PR review — a breaking change here breaks everyone.

Models (in pipeline order):
    Task          – a single evaluation task
    ProbeResult   – output from the CoT-SC Probe Agent (Person 2)
    GateFeatures  – extracted features fed into the Gate (Person 3)
    GateDecision  – the Gate's routing decision (Person 3)
    EvalResult    – final per-task evaluation record (Person 4)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ─────────────────────────────────────────────────────────────────────────────
# Task
# ─────────────────────────────────────────────────────────────────────────────


class Task(BaseModel):
    """A single evaluation task from MASBench-mini.

    Produced by: Person 1 (dataset/)
    Consumed by: Person 2 (probe_agent), Person 3 (feature_extractor),
                 Person 4 (evaluation harness)
    """

    task_id: str = Field(..., description="Unique identifier, e.g. 'hotpot_001'")
    question: str = Field(..., min_length=1, description="The question text")
    context: str | None = Field(
        default=None,
        description="Supporting passage(s), if any (multi-hop tasks often have these)",
    )
    depth_score: int | None = Field(
        default=None,
        ge=1,
        le=5,
        description="MASBench Depth axis label (1=shallow, 5=deep multi-hop). "
        "Set by Person 1's labeling heuristics.",
    )
    parallel_score: int | None = Field(
        default=None,
        ge=1,
        le=4,
        description="MASBench Parallelism axis label (1=sequential, 4=highly parallel). "
        "Set by Person 1's labeling heuristics.",
    )
    ground_truth: str | None = Field(
        default=None,
        description="Gold answer for evaluation. May be absent at inference time.",
    )
    source_dataset: str | None = Field(
        default=None,
        description="Origin dataset: 'hotpotqa', 'musique', 'template_arithmetic', etc.",
    )

    @field_validator("task_id")
    @classmethod
    def task_id_no_spaces(cls, v: str) -> str:
        if " " in v:
            raise ValueError("task_id must not contain spaces")
        return v

    model_config = {"frozen": True}  # Tasks are immutable once created


# ─────────────────────────────────────────────────────────────────────────────
# ProbeResult
# ─────────────────────────────────────────────────────────────────────────────


class ProbeResult(BaseModel):
    """Output from the cheap CoT-SC Probe Agent.

    Produced by: Person 2 (agents/probe_agent.py)
    Consumed by: Person 3 (feature_extractor, pipeline)

    The probe runs N CoT samples and picks the majority answer.
    consistency_score = fraction of samples that agree with the majority answer.
    """

    task_id: str = Field(..., description="Must match the Task.task_id this was run on")
    answer: str = Field(..., min_length=1, description="Majority-vote answer from CoT-SC")
    consistency_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Agreement rate among CoT-SC samples (e.g. 4/5 = 0.8). "
        "High consistency → gate leans STOP.",
    )
    tokens_used: int = Field(
        ...,
        ge=0,
        description="Total tokens consumed by all probe samples (prompt + completion)",
    )
    latency_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Wall-clock time for the probe run in milliseconds",
    )
    raw_outputs: list[str] = Field(
        default_factory=list,
        description="Individual CoT-SC sample outputs (length = n_samples). "
        "Used for consistency calculation and debugging.",
    )
    model_name: str | None = Field(
        default=None,
        description="LLM used for the probe (should match config.MODEL_NAME)",
    )

    @field_validator("raw_outputs")
    @classmethod
    def raw_outputs_not_empty_strings(cls, v: list[str]) -> list[str]:
        if any(s.strip() == "" for s in v):
            raise ValueError("raw_outputs must not contain empty strings")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# GateFeatures
# ─────────────────────────────────────────────────────────────────────────────


class GateFeatures(BaseModel):
    """Feature vector fed into the Gate classifier.

    Produced by: Person 3 (gate/feature_extractor.py)
    Consumed by: Person 3 (gate/classifier.py, gate/rule_based_gate.py)

    All features are extractable BEFORE MAS execution — this is the key
    property that makes the gate cheap.
    """

    task_id: str = Field(..., description="Matches Task.task_id")

    # ── Probe-derived features (free from ProbeResult) ────────────────────
    consistency_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="CoT-SC consistency score from ProbeResult",
    )
    probe_tokens: int = Field(
        ...,
        ge=0,
        description="Token count from ProbeResult — proxy for task complexity",
    )

    # ── Text-derived features (from Task.question + context) ───────────────
    question_word_count: int = Field(
        ..., ge=0, description="Number of words in the question"
    )
    entity_count: int = Field(
        ...,
        ge=0,
        description="Named entity count via spaCy NER (or regex fallback)",
    )
    clause_count: int = Field(
        ...,
        ge=0,
        description="Subordinate clause count via dep-parse (or comma/conjunction count)",
    )
    has_context: bool = Field(
        ..., description="Whether Task.context is non-None and non-empty"
    )

    # ── Axis proxy features (heuristic approximations before labeling) ─────
    estimated_depth: float | None = Field(
        default=None,
        ge=0.0,
        description="Heuristic depth estimate (e.g. entity_count + clause_count normalized)",
    )
    estimated_parallel: float | None = Field(
        default=None,
        ge=0.0,
        description="Heuristic parallelism estimate (e.g. count of conjunctions/sub-questions)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# GateDecision
# ─────────────────────────────────────────────────────────────────────────────


class GateDecision(BaseModel):
    """The Gate's routing decision for a single task.

    Produced by: Person 3 (gate/classifier.py, gate/rule_based_gate.py, gate/random_gate.py)
    Consumed by: Person 3 (integration/pipeline.py), Person 4 (evaluation)

    STOP     → return probe answer, spend 0 additional tokens
    ESCALATE → send task to MAS orchestrator with token_budget_cap
    """

    task_id: str = Field(..., description="Matches Task.task_id")
    decision: Literal["STOP", "ESCALATE"] = Field(
        ..., description="The gate's routing decision"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Gate's confidence in its decision (1.0 = certain). "
        "For rule-based and random gates, use 1.0 and 0.5 respectively.",
    )
    token_budget_cap: int | None = Field(
        default=None,
        ge=1,
        description="Maximum tokens the MAS orchestrator may spend (= k × probe_tokens). "
        "Must be set iff decision == 'ESCALATE'.",
    )
    gate_type: str = Field(
        default="learned",
        description="Which gate produced this decision: 'learned', 'rule_based', 'random'",
    )

    @model_validator(mode="after")
    def budget_cap_iff_escalate(self) -> GateDecision:
        if self.decision == "ESCALATE" and self.token_budget_cap is None:
            raise ValueError(
                "token_budget_cap must be set when decision is 'ESCALATE'"
            )
        if self.decision == "STOP" and self.token_budget_cap is not None:
            raise ValueError(
                "token_budget_cap must be None when decision is 'STOP'"
            )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# EvalResult
# ─────────────────────────────────────────────────────────────────────────────


class EvalResult(BaseModel):
    """Final evaluation record for a single (task, method) pair.

    Produced by: Person 3 (integration/pipeline.py)
    Consumed by: Person 4 (evaluation/metrics.py, pareto_plot.py)

    One EvalResult per task per method.  Running all 5 methods on 30 test
    tasks → 150 EvalResult objects total.
    """

    task_id: str = Field(..., description="Matches Task.task_id")
    method: Literal[
        "CoT-SC-only",
        "Always-MAS",
        "GateOrchestra",
        "RuleBasedGate",
        "RandomGate",
    ] = Field(..., description="Which method produced this result")
    predicted_answer: str = Field(..., min_length=1)
    is_correct: bool | None = Field(
        default=None,
        description="Whether predicted_answer matches ground_truth (exact or normalized). "
        "None if ground_truth was unavailable.",
    )
    tokens_spent: int = Field(
        ...,
        ge=0,
        description="Total tokens spent producing this answer (probe + MAS if escalated)",
    )
    probe_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Tokens spent on the probe stage only",
    )
    mas_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Tokens spent on MAS stage (None if STOP or method has no probe)",
    )
    gate_decision: GateDecision | None = Field(
        default=None,
        description="The gate decision that produced this result. "
        "None for CoT-SC-only and Always-MAS baselines.",
    )
    latency_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Total wall-clock time in milliseconds",
    )

    @model_validator(mode="after")
    def tokens_consistency(self) -> EvalResult:
        """Verify probe + mas tokens add up to total (when both are available)."""
        if self.probe_tokens is not None and self.mas_tokens is not None:
            expected = self.probe_tokens + self.mas_tokens
            if self.tokens_spent != expected:
                raise ValueError(
                    f"tokens_spent ({self.tokens_spent}) != "
                    f"probe_tokens ({self.probe_tokens}) + mas_tokens ({self.mas_tokens})"
                )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    task = Task(
        task_id="demo_001",
        question="What is the capital of the country where the Eiffel Tower is located?",
        ground_truth="Paris",
        depth_score=2,
        parallel_score=1,
        source_dataset="template_arithmetic",
    )

    probe = ProbeResult(
        task_id="demo_001",
        answer="Paris",
        consistency_score=1.0,
        tokens_used=120,
        raw_outputs=["Paris", "Paris", "Paris", "Paris", "Paris"],
    )

    features = GateFeatures(
        task_id="demo_001",
        consistency_score=probe.consistency_score,
        probe_tokens=probe.tokens_used,
        question_word_count=len(task.question.split()),
        entity_count=2,
        clause_count=1,
        has_context=task.context is not None,
    )

    decision = GateDecision(
        task_id="demo_001",
        decision="STOP",
        confidence=0.92,
        gate_type="learned",
    )

    result = EvalResult(
        task_id="demo_001",
        method="GateOrchestra",
        predicted_answer=probe.answer,
        is_correct=True,
        tokens_spent=probe.tokens_used,
        probe_tokens=probe.tokens_used,
        gate_decision=decision,
    )

    print("=== GateOrchestra Schema Smoke Test ===")
    for name, obj in [
        ("Task", task),
        ("ProbeResult", probe),
        ("GateFeatures", features),
        ("GateDecision", decision),
        ("EvalResult", result),
    ]:
        print(f"\n── {name} ──")
        print(json.dumps(obj.model_dump(), indent=2, default=str))

    print("\n✅ All schemas OK")

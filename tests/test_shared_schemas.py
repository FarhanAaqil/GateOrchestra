"""
tests/test_shared_schemas.py
============================
Contract tests for all 5 Pydantic schemas.

These tests MUST pass on every PR — they verify the shared contract layer.
"""

import pytest
from pydantic import ValidationError

from shared.schemas import (
    EvalResult,
    GateDecision,
    GateFeatures,
    ProbeResult,
    Task,
)

# ─────────────────────────────────────────────────────────────────────────────
# Task
# ─────────────────────────────────────────────────────────────────────────────


class TestTask:
    def test_minimal_task(self):
        t = Task(task_id="t001", question="What is 2+2?")
        assert t.task_id == "t001"
        assert t.question == "What is 2+2?"
        assert t.ground_truth is None

    def test_full_task(self):
        t = Task(
            task_id="t002",
            question="What happened?",
            context="Some context.",
            depth_score=3,
            parallel_score=2,
            ground_truth="Something",
            source_dataset="hotpotqa",
        )
        assert t.depth_score == 3

    def test_task_id_no_spaces(self):
        with pytest.raises(ValidationError):
            Task(task_id="bad id", question="Q?")

    def test_depth_score_out_of_range(self):
        with pytest.raises(ValidationError):
            Task(task_id="t003", question="Q?", depth_score=6)

    def test_parallel_score_out_of_range(self):
        with pytest.raises(ValidationError):
            Task(task_id="t004", question="Q?", parallel_score=0)

    def test_empty_question_rejected(self):
        with pytest.raises(ValidationError):
            Task(task_id="t005", question="")

    def test_task_is_frozen(self):
        t = Task(task_id="t006", question="Q?")
        with pytest.raises(ValidationError):
            t.question = "Modified"  # type: ignore

    def test_json_roundtrip(self):
        t = Task(task_id="t007", question="Q?", depth_score=2)
        restored = Task(**t.model_dump())
        assert restored == t


# ─────────────────────────────────────────────────────────────────────────────
# ProbeResult
# ─────────────────────────────────────────────────────────────────────────────


class TestProbeResult:
    def _base(self) -> dict:
        return {
            "task_id": "t001",
            "answer": "Paris",
            "consistency_score": 0.8,
            "tokens_used": 100,
        }

    def test_valid_probe_result(self):
        pr = ProbeResult(**self._base())
        assert pr.consistency_score == 0.8

    def test_consistency_score_out_of_range(self):
        d = self._base()
        d["consistency_score"] = 1.1
        with pytest.raises(ValidationError):
            ProbeResult(**d)

    def test_negative_tokens_rejected(self):
        d = self._base()
        d["tokens_used"] = -1
        with pytest.raises(ValidationError):
            ProbeResult(**d)

    def test_empty_raw_output_string_rejected(self):
        d = self._base()
        d["raw_outputs"] = ["Paris", ""]
        with pytest.raises(ValidationError):
            ProbeResult(**d)

    def test_json_roundtrip(self):
        pr = ProbeResult(**self._base(), raw_outputs=["Paris", "Paris", "London"])
        assert ProbeResult(**pr.model_dump()) == pr


# ─────────────────────────────────────────────────────────────────────────────
# GateFeatures
# ─────────────────────────────────────────────────────────────────────────────


class TestGateFeatures:
    def _base(self) -> dict:
        return {
            "task_id": "t001",
            "consistency_score": 0.8,
            "probe_tokens": 100,
            "question_word_count": 10,
            "entity_count": 2,
            "clause_count": 1,
            "has_context": False,
        }

    def test_valid_features(self):
        gf = GateFeatures(**self._base())
        assert gf.entity_count == 2

    def test_optional_depth_parallel(self):
        gf = GateFeatures(**self._base(), estimated_depth=2.5, estimated_parallel=1.0)
        assert gf.estimated_depth == 2.5

    def test_json_roundtrip(self):
        gf = GateFeatures(**self._base())
        assert GateFeatures(**gf.model_dump()) == gf


# ─────────────────────────────────────────────────────────────────────────────
# GateDecision
# ─────────────────────────────────────────────────────────────────────────────


class TestGateDecision:
    def test_stop_decision(self):
        gd = GateDecision(task_id="t001", decision="STOP", confidence=0.9)
        assert gd.decision == "STOP"
        assert gd.token_budget_cap is None

    def test_escalate_decision(self):
        gd = GateDecision(
            task_id="t001", decision="ESCALATE", confidence=0.75, token_budget_cap=500
        )
        assert gd.token_budget_cap == 500

    def test_escalate_without_budget_cap_fails(self):
        with pytest.raises(ValidationError):
            GateDecision(task_id="t001", decision="ESCALATE", confidence=0.75)

    def test_stop_with_budget_cap_fails(self):
        with pytest.raises(ValidationError):
            GateDecision(task_id="t001", decision="STOP", confidence=0.9, token_budget_cap=500)

    def test_confidence_out_of_range(self):
        with pytest.raises(ValidationError):
            GateDecision(task_id="t001", decision="STOP", confidence=1.5)

    def test_json_roundtrip(self):
        gd = GateDecision(task_id="t001", decision="STOP", confidence=0.9)
        assert GateDecision(**gd.model_dump()) == gd


# ─────────────────────────────────────────────────────────────────────────────
# EvalResult
# ─────────────────────────────────────────────────────────────────────────────


class TestEvalResult:
    def _base(self) -> dict:
        return {
            "task_id": "t001",
            "method": "GateOrchestra",
            "predicted_answer": "Paris",
            "tokens_spent": 120,
        }

    def test_minimal_eval_result(self):
        er = EvalResult(**self._base())
        assert er.is_correct is None
        assert er.gate_decision is None

    def test_full_eval_result_stop(self):
        decision = GateDecision(task_id="t001", decision="STOP", confidence=0.9)
        er = EvalResult(
            **self._base(),
            is_correct=True,
            probe_tokens=120,
            mas_tokens=0,
            gate_decision=decision,
        )
        assert er.is_correct is True

    def test_token_consistency_check(self):
        # probe(80) + mas(50) = 130 ≠ tokens_spent(120) → error
        with pytest.raises(ValidationError):
            EvalResult(
                task_id="t001",
                method="GateOrchestra",
                predicted_answer="Paris",
                tokens_spent=120,
                probe_tokens=80,
                mas_tokens=50,
            )

    def test_invalid_method_rejected(self):
        d = self._base()
        d["method"] = "InvalidMethod"
        with pytest.raises(ValidationError):
            EvalResult(**d)

    def test_json_roundtrip(self):
        er = EvalResult(**self._base(), is_correct=False)
        assert EvalResult(**er.model_dump()) == er


# ─────────────────────────────────────────────────────────────────────────────
# EvalResult.mas_strategy — Week 8 Priority 2
# ─────────────────────────────────────────────────────────────────────────────


class TestEvalResultMasStrategy:
    def _base(self) -> dict:
        return {
            "task_id": "t001",
            "method": "GateOrchestra",
            "predicted_answer": "Paris",
            "tokens_spent": 120,
        }

    def test_mas_strategy_defaults_to_none(self):
        """Existing callers that omit mas_strategy continue to work unchanged."""
        er = EvalResult(**self._base())
        assert er.mas_strategy is None

    def test_mas_strategy_none_for_stop(self):
        """STOP results must have mas_strategy=None explicitly."""
        decision = GateDecision(task_id="t001", decision="STOP", confidence=0.95)
        er = EvalResult(
            **self._base(),
            probe_tokens=120,
            mas_tokens=0,
            gate_decision=decision,
            mas_strategy=None,
        )
        assert er.mas_strategy is None

    def test_mas_strategy_react_accepted(self):
        decision = GateDecision(
            task_id="t001", decision="ESCALATE", confidence=0.8, token_budget_cap=400
        )
        er = EvalResult(
            **self._base(),
            probe_tokens=80,
            mas_tokens=40,
            gate_decision=decision,
            mas_strategy="react",
        )
        assert er.mas_strategy == "react"

    def test_mas_strategy_debate_accepted(self):
        decision = GateDecision(
            task_id="t001", decision="ESCALATE", confidence=0.7, token_budget_cap=400
        )
        er = EvalResult(
            task_id="t001",
            method="GateOrchestra",
            predicted_answer="Tokyo",
            tokens_spent=200,
            probe_tokens=100,
            mas_tokens=100,
            gate_decision=decision,
            mas_strategy="debate",
        )
        assert er.mas_strategy == "debate"

    def test_mas_strategy_reflexion_accepted(self):
        decision = GateDecision(
            task_id="t001", decision="ESCALATE", confidence=0.6, token_budget_cap=400
        )
        er = EvalResult(
            task_id="t001",
            method="GateOrchestra",
            predicted_answer="Rome",
            tokens_spent=300,
            probe_tokens=150,
            mas_tokens=150,
            gate_decision=decision,
            mas_strategy="reflexion",
        )
        assert er.mas_strategy == "reflexion"

    def test_invalid_mas_strategy_rejected(self):
        """An unknown strategy name must fail Pydantic validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            EvalResult(**self._base(), mas_strategy="unknown_strategy")

    def test_json_roundtrip_with_strategy(self):
        """model_dump / model reconstruct must preserve mas_strategy."""
        decision = GateDecision(
            task_id="t001", decision="ESCALATE", confidence=0.75, token_budget_cap=300
        )
        er = EvalResult(
            task_id="t001",
            method="GateOrchestra",
            predicted_answer="Berlin",
            tokens_spent=250,
            probe_tokens=100,
            mas_tokens=150,
            gate_decision=decision,
            mas_strategy="react",
        )
        restored = EvalResult(**er.model_dump())
        assert restored.mas_strategy == "react"
        assert restored == er


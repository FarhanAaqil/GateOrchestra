"""
tests/test_probe_agent.py
=========================
Unit tests for agents/probe_agent.py (Person 2).
Tests answer extraction, CoT prompting, self-consistency majority voting,
error handling, and schema integration without requiring live LLM servers.
"""

from typing import Any, cast

import pytest

from agents.probe_agent import (
    ProbeAgent,
    build_cot_prompt,
    extract_answer,
    normalize_answer,
    probe_agent,
)
from integration.pipeline import run_pipeline
from shared.config import COT_SC_N_SAMPLES
from shared.schemas import EvalResult, ProbeResult, Task
from shared.token_logger import TokenAccountant
from tests.mocks.mock_orchestrator import mock_orchestrator
from gate.rule_based_gate import RuleBasedGate


# ─────────────────────────────────────────────────────────────────────────────
# Test Answer Extraction & Normalization
# ─────────────────────────────────────────────────────────────────────────────


class TestAnswerExtraction:
    def test_final_answer_pattern(self):
        text = "First we calculate 2+2=4.\nFinal Answer: 4"
        assert extract_answer(text) == "4"

    def test_the_answer_is_pattern(self):
        text = "The country is France. Therefore, the answer is: Paris."
        assert extract_answer(text) == "Paris"

    def test_hash_pattern(self):
        text = "Step 1: x = 10\nStep 2: y = 20\n#### 30"
        assert extract_answer(text) == "30"

    def test_bold_answer_pattern(self):
        text = "Thinking through this...\n**Answer**: Marie Curie"
        assert extract_answer(text) == "Marie Curie"

    def test_fallback_last_line(self):
        text = "Reasoning step 1\nReasoning step 2\nLeonardo da Vinci"
        assert extract_answer(text) == "Leonardo da Vinci"

    def test_empty_or_whitespace_output(self):
        assert extract_answer("") == "Unknown"
        assert extract_answer("   \n\t ") == "Unknown"

    def test_normalize_answer(self):
        assert normalize_answer("  Paris.  ") == "paris"
        assert normalize_answer("**Paris**") == "paris"
        assert normalize_answer("'Paris'") == "paris"
        assert normalize_answer("The   United  States ") == "the united states"


# ─────────────────────────────────────────────────────────────────────────────
# Test Prompt Construction
# ─────────────────────────────────────────────────────────────────────────────


class TestPromptConstruction:
    def test_prompt_without_context(self):
        task = Task(task_id="t1", question="What is the speed of light?")
        prompt = build_cot_prompt(task)
        assert "What is the speed of light?" in prompt
        assert "Context:" not in prompt
        assert "Final Answer:" in prompt

    def test_prompt_with_context(self):
        task = Task(
            task_id="t2",
            question="Who won the Nobel prize?",
            context="Albert Einstein was awarded the 1921 Nobel Prize in Physics.",
        )
        prompt = build_cot_prompt(task)
        assert "Context:" in prompt
        assert "Albert Einstein was awarded" in prompt
        assert "Who won the Nobel prize?" in prompt


# ─────────────────────────────────────────────────────────────────────────────
# Test ProbeAgent Execution & Majority Voting
# ─────────────────────────────────────────────────────────────────────────────


class TestProbeAgentExecution:
    def test_majority_voting_unanimous(self):
        task = Task(task_id="t_unan", question="What is 1+1?")
        mock_responses = [
            ("Thinking... Final Answer: 2", 25),
            ("Step 1... Final Answer: 2", 20),
            ("1+1=2. Final Answer: 2", 30),
            ("Answer: 2", 15),
            ("2", 10),
        ]
        call_idx = 0

        def custom_caller(prompt: str, temp: float, max_tok: int) -> tuple[str, int]:
            nonlocal call_idx
            resp = mock_responses[call_idx % len(mock_responses)]
            call_idx += 1
            return resp

        agent = ProbeAgent(n_samples=5, llm_caller=custom_caller)
        result = agent.run(task)

        assert isinstance(result, ProbeResult)
        assert result.task_id == "t_unan"
        assert result.answer == "2"
        assert result.consistency_score == 1.0
        assert result.tokens_used == 100
        assert len(result.raw_outputs) == 5
        assert result.latency_ms is not None and result.latency_ms >= 0.0

    def test_majority_voting_split_agreement(self):
        task = Task(task_id="t_split", question="Where is the Louvre?")
        # 3 Paris, 2 Rome -> majority = Paris, consistency = 3/5 = 0.6
        mock_outputs = [
            ("Final Answer: Paris", 20),
            ("Final Answer: Rome", 20),
            ("Final Answer: Paris", 20),
            ("Final Answer: Paris", 20),
            ("Final Answer: Rome", 20),
        ]
        idx = 0

        def split_caller(prompt: str, temp: float, max_tok: int) -> tuple[str, int]:
            nonlocal idx
            res = mock_outputs[idx % len(mock_outputs)]
            idx += 1
            return res

        agent = ProbeAgent(n_samples=5, llm_caller=split_caller)
        result = agent.run(task)

        assert result.answer == "Paris"
        assert result.consistency_score == pytest.approx(0.6)
        assert result.tokens_used == 100

    def test_callable_instance_interface(self):
        task = Task(task_id="t_call", question="What is the capital of Japan?")

        def simple_caller(prompt: str, temp: float, max_tok: int) -> tuple[str, int]:
            return "Final Answer: Tokyo", 15

        agent = ProbeAgent(n_samples=3, llm_caller=simple_caller)
        result = agent(task)  # Call directly as a function

        assert isinstance(result, ProbeResult)
        assert result.answer == "Tokyo"
        assert result.consistency_score == 1.0
        assert result.tokens_used == 45

    def test_error_handling_resilience(self):
        task = Task(task_id="t_err", question="Test resilience")

        def failing_caller(prompt: str, temp: float, max_tok: int) -> tuple[str, int]:
            raise RuntimeError("Network timeout")

        agent = ProbeAgent(n_samples=3, llm_caller=failing_caller)
        result = agent.run(task)

        assert isinstance(result, ProbeResult)
        assert len(result.raw_outputs) == 3
        # Ensure schema validates non-empty strings
        assert all(len(s) > 0 for s in result.raw_outputs)

    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="n_samples must be ≥ 1"):
            ProbeAgent(n_samples=0)

        with pytest.raises(ValueError, match="temperature must be ≥ 0.0"):
            ProbeAgent(temperature=-0.5)

        with pytest.raises(ValueError, match="token_budget must be ≥ 1"):
            ProbeAgent(token_budget=0)


# ─────────────────────────────────────────────────────────────────────────────
# Test Pipeline Integration with ProbeAgent
# ─────────────────────────────────────────────────────────────────────────────


class TestPipelineIntegration:
    def test_probe_agent_in_pipeline(self):
        task = Task(
            task_id="pipe_probe_001",
            question="What is 5 * 5?",
            ground_truth="25",
        )

        def mock_caller(prompt: str, temp: float, max_tok: int) -> tuple[str, int]:
            return "Final Answer: 25", 20

        agent = ProbeAgent(n_samples=5, llm_caller=mock_caller)
        gate = RuleBasedGate()
        accountant = TokenAccountant()

        result = run_pipeline(
            task=task,
            gate=cast(Any, gate),
            probe_agent=agent,
            orchestrator=mock_orchestrator,
            accountant=accountant,
            k=3,
            method="GateOrchestra",
        )

        assert isinstance(result, EvalResult)
        assert result.task_id == "pipe_probe_001"
        assert result.predicted_answer == "25"
        assert result.is_correct is True
        assert result.probe_tokens == 100
        assert accountant.get_spend("pipe_probe_001")["probe"] == 100

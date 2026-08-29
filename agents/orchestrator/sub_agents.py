"""
agents/orchestrator/sub_agents.py
=================================
Sub-agent pool for the Multi-Agent System (MAS) Orchestrator (Person 2).

Includes:
  - ReActAgent: Thought-Action-Observation loop for multi-hop reasoning.
  - DebateAgent: Proposer-Critic multi-agent debate and consensus.
  - ReflexionAgent: Self-reflection and iterative answer refinement.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from agents.probe_agent import extract_answer
from agents.providers import default_llm_caller
from shared.config import (
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
    LLM_PROVIDER,
    MODEL_NAME,
)
from shared.schemas import Task

logger = logging.getLogger(__name__)

LLMCallerFn = Callable[[str, float, int], tuple[str, int]]


# ─────────────────────────────────────────────────────────────────────────────
# 1. ReAct Agent (Decomposition & Multi-Hop Reasoning)
# ─────────────────────────────────────────────────────────────────────────────


class ReActAgent:
    """ReAct (Reasoning + Acting) Agent for multi-hop decomposition."""

    def __init__(
        self,
        model_name: str | None = None,
        max_steps: int = 3,
        provider: str | None = None,
        api_key: str | None = None,
        llm_caller: LLMCallerFn | None = None,
    ) -> None:
        self.provider = provider or LLM_PROVIDER
        self.api_key = api_key or GROQ_API_KEY
        if self.provider == "groq":
            self.model_name = model_name or GROQ_MODEL_NAME
        else:
            self.model_name = model_name or MODEL_NAME

        self.max_steps = max_steps
        self.llm_caller = llm_caller

    def _call(self, prompt: str, budget: int) -> tuple[str, int]:
        if self.llm_caller is not None:
            return self.llm_caller(prompt, 0.2, budget)
        try:
            return default_llm_caller(
                prompt=prompt,
                temperature=0.2,
                max_tokens=budget,
                model_name=self.model_name,
                provider=self.provider,
                api_key=self.api_key,
            )
        except Exception as e:
            logger.warning(f"[ReActAgent] LLM call error: {e}")
            return f"Final Answer: {extract_answer(prompt)}", 10

    def run(self, task: Task, token_budget: int) -> tuple[str, int]:
        """Execute ReAct reasoning loop up to max_steps or token budget exhaustion."""
        total_tokens = 0
        context_str = f"Context: {task.context}\n" if task.context else ""

        history = [
            f"Solve the following question step by step using Thought and Action.\n"
            f"{context_str}Question: {task.question}\n"
        ]

        step_budget = max(20, token_budget // (self.max_steps + 1))

        for step in range(1, self.max_steps + 1):
            if total_tokens >= token_budget:
                break

            prompt = "\n".join(history) + f"\nStep {step}:\nThought:"
            current_budget = min(step_budget, token_budget - total_tokens)

            response, tokens = self._call(prompt, current_budget)
            total_tokens += tokens
            history.append(f"Step {step}:\nThought: {response}")

            # Check if final answer is reached
            if "final answer" in response.lower() or "the answer is" in response.lower():
                answer = extract_answer(response)
                return answer, total_tokens

        # Final synthesis if loop ended without explicit Final Answer
        synthesis_prompt = "\n".join(history) + "\nConclude with: Final Answer: <answer>"
        remaining_budget = max(10, token_budget - total_tokens)
        final_resp, fin_tokens = self._call(synthesis_prompt, remaining_budget)
        total_tokens += fin_tokens

        return extract_answer(final_resp), total_tokens


# ─────────────────────────────────────────────────────────────────────────────
# 2. Multi-Agent Debate & Consensus Agent
# ─────────────────────────────────────────────────────────────────────────────


class DebateAgent:
    """Proposer-Critic Multi-Agent Debate."""

    def __init__(
        self,
        model_name: str | None = None,
        num_rounds: int = 2,
        provider: str | None = None,
        api_key: str | None = None,
        llm_caller: LLMCallerFn | None = None,
    ) -> None:
        self.provider = provider or LLM_PROVIDER
        self.api_key = api_key or GROQ_API_KEY
        if self.provider == "groq":
            self.model_name = model_name or GROQ_MODEL_NAME
        else:
            self.model_name = model_name or MODEL_NAME

        self.num_rounds = num_rounds
        self.llm_caller = llm_caller

    def _call(self, prompt: str, budget: int) -> tuple[str, int]:
        if self.llm_caller is not None:
            return self.llm_caller(prompt, 0.4, budget)
        try:
            return default_llm_caller(
                prompt=prompt,
                temperature=0.4,
                max_tokens=budget,
                model_name=self.model_name,
                provider=self.provider,
                api_key=self.api_key,
            )
        except Exception as e:
            logger.warning(f"[DebateAgent] LLM call error: {e}")
            return f"Final Answer: {extract_answer(prompt)}", 10

    def run(self, task: Task, token_budget: int) -> tuple[str, int]:
        """Run multi-agent proposer and critic debate rounds."""
        total_tokens = 0
        context_str = f"Context: {task.context}\n" if task.context else ""
        round_budget = max(20, token_budget // (self.num_rounds * 2 + 1))

        # Round 1: Proposer initial solution
        prop_prompt = (
            f"You are Agent Proposer. Answer this question with complete reasoning.\n"
            f"{context_str}Question: {task.question}\n"
            f"Conclude with Final Answer: <answer>"
        )
        proposer_ans, tok1 = self._call(prop_prompt, round_budget)
        total_tokens += tok1

        current_solution = proposer_ans

        for _r in range(1, self.num_rounds + 1):
            if total_tokens >= token_budget:
                break

            # Critic step
            critic_prompt = (
                f"You are Agent Critic. Critique the following proposed solution for errors or unverified assumptions.\n"
                f"Question: {task.question}\n"
                f"Proposed Solution: {current_solution}\n"
                f"Point out any flaws or confirm if correct."
            )
            critic_feedback, tok_c = self._call(
                critic_prompt, min(round_budget, token_budget - total_tokens)
            )
            total_tokens += tok_c

            if total_tokens >= token_budget:
                break

            # Proposer revision step
            revision_prompt = (
                f"You are Agent Proposer. Review Critic's feedback and refine your answer.\n"
                f"Critic Feedback: {critic_feedback}\n"
                f"Question: {task.question}\n"
                f"Provide final refined conclusion with Final Answer: <answer>"
            )
            current_solution, tok_r = self._call(
                revision_prompt, min(round_budget, token_budget - total_tokens)
            )
            total_tokens += tok_r

        return extract_answer(current_solution), total_tokens


# ─────────────────────────────────────────────────────────────────────────────
# 3. Reflexion Agent (Self-Reflection & Refinement)
# ─────────────────────────────────────────────────────────────────────────────


class ReflexionAgent:
    """Self-Reflection agent that validates and refines draft solutions."""

    def __init__(
        self,
        model_name: str | None = None,
        provider: str | None = None,
        api_key: str | None = None,
        llm_caller: LLMCallerFn | None = None,
    ) -> None:
        self.provider = provider or LLM_PROVIDER
        self.api_key = api_key or GROQ_API_KEY
        if self.provider == "groq":
            self.model_name = model_name or GROQ_MODEL_NAME
        else:
            self.model_name = model_name or MODEL_NAME

        self.llm_caller = llm_caller

    def _call(self, prompt: str, budget: int) -> tuple[str, int]:
        if self.llm_caller is not None:
            return self.llm_caller(prompt, 0.3, budget)
        try:
            return default_llm_caller(
                prompt=prompt,
                temperature=0.3,
                max_tokens=budget,
                model_name=self.model_name,
                provider=self.provider,
                api_key=self.api_key,
            )
        except Exception as e:
            logger.warning(f"[ReflexionAgent] LLM call error: {e}")
            return f"Final Answer: {extract_answer(prompt)}", 10

    def run(self, task: Task, token_budget: int) -> tuple[str, int]:
        """Draft an initial response, self-critique, and refine."""
        total_tokens = 0
        context_str = f"Context: {task.context}\n" if task.context else ""
        step_budget = max(20, token_budget // 3)

        # 1. Draft
        draft_prompt = (
            f"Answer the question step by step.\n"
            f"{context_str}Question: {task.question}\n"
            f"Draft Answer:"
        )
        draft, tok1 = self._call(draft_prompt, step_budget)
        total_tokens += tok1

        if total_tokens >= token_budget:
            return extract_answer(draft), total_tokens

        # 2. Reflect
        reflect_prompt = (
            f"Question: {task.question}\n"
            f"Draft: {draft}\n"
            f"Critique: Are there any logical fallacies, arithmetic errors, or missed nuances?"
        )
        reflection, tok2 = self._call(reflect_prompt, min(step_budget, token_budget - total_tokens))
        total_tokens += tok2

        if total_tokens >= token_budget:
            return extract_answer(draft), total_tokens

        # 3. Refine
        refine_prompt = (
            f"Question: {task.question}\n"
            f"Reflection: {reflection}\n"
            f"Provide the final corrected answer. Format: Final Answer: <answer>"
        )
        refined, tok3 = self._call(refine_prompt, min(step_budget, token_budget - total_tokens))
        total_tokens += tok3

        return extract_answer(refined), total_tokens

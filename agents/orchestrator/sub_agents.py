"""
agents/orchestrator/sub_agents.py
=================================
Sub-agent pool for the Multi-Agent System (MAS) Orchestrator (Person 2).

Includes:
  - ReActAgent: Thought-Action-Observation loop for multi-hop reasoning.
  - DebateAgent: Proposer-Critic multi-agent debate and consensus.
  - ReflexionAgent: Self-reflection and iterative answer refinement.

Token accounting rules shared by all sub-agents:
  - The returned token count is the provider's true total (prompt +
    completion) for every call made. It is never clipped to the budget.
  - The budget is enforced before each call: the completion cap is the
    remaining budget minus an estimate of the prompt's size, and no call is
    made once that leaves fewer than MIN_CALL_TOKENS.
  - A failed LLM call raises SubAgentError. It is never turned into an answer.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

from agents.orchestrator.tools import ToolError, calculate, lookup
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

MIN_CALL_TOKENS = 32
"""Smallest completion cap worth spending a call on."""

TRACE_OUTPUT_CHARS = 2000
"""How much of each step's output is kept in the trace."""


class SubAgentError(RuntimeError):
    """An LLM call made by a sub-agent failed."""


def estimate_prompt_tokens(prompt: str) -> int:
    """Rough prompt size (~4 characters per token) used to reserve budget."""
    return len(prompt) // 4 + 1


def completion_cap(prompt: str, remaining_budget: int) -> int:
    """Completion tokens a call may use so prompt + completion fits the budget.

    Returns 0 when the remaining budget cannot fit a useful call.
    """
    cap = remaining_budget - estimate_prompt_tokens(prompt)
    return cap if cap >= MIN_CALL_TOKENS else 0


def _context_block(task: Task) -> str:
    return f"Context: {task.context}\n" if task.context else ""


class _SubAgentBase:
    """Shared provider wiring and budgeted LLM calls."""

    temperature: float = 0.3

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
        # Steps taken by the most recent run(): step name, output, tokens.
        self.last_trace: list[dict] = []

    def _call(self, prompt: str, budget: int, step: str = "call") -> tuple[str, int]:
        try:
            if self.llm_caller is not None:
                response, tokens = self.llm_caller(prompt, self.temperature, budget)
            else:
                response, tokens = default_llm_caller(
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=budget,
                    model_name=self.model_name,
                    provider=self.provider,
                    api_key=self.api_key,
                )
        except Exception as e:
            logger.warning(f"[{type(self).__name__}] LLM call error: {e}")
            raise SubAgentError(f"{type(self).__name__} LLM call failed: {e}") from e
        self.last_trace.append(
            {"step": step, "output": response[:TRACE_OUTPUT_CHARS], "tokens": tokens}
        )
        return response, tokens

    def _budgeted_call(
        self,
        prompt: str,
        remaining_budget: int,
        step_cap: int | None = None,
        step: str = "call",
    ) -> tuple[str, int] | None:
        """Call the LLM if the remaining budget allows; None means 'out of budget'."""
        cap = completion_cap(prompt, remaining_budget)
        if step_cap is not None:
            cap = min(cap, step_cap)
        if cap < MIN_CALL_TOKENS:
            return None
        return self._call(prompt, cap, step)


# 1. ReAct Agent (Decomposition & Multi-Hop Reasoning)

REACT_INSTRUCTIONS = (
    "Answer the question by alternating Thought and Action lines.\n"
    "Available actions:\n"
    "  lookup[term] - returns the sentences of the context that mention the term\n"
    "  calculate[expression] - evaluates arithmetic, e.g. calculate[20000 * 0.85**2]\n"
    "  finish[...] - gives the final, concise answer, e.g. finish[Paris]\n"
    "Write one Thought and one Action per step, then stop and wait for the Observation.\n"
)
_ACTION_RE = re.compile(
    r"Action(?:\s*\d+)?\s*:\s*(lookup|calculate|finish)\s*\[([^\]\n]*)\]", re.IGNORECASE
)


_PLACEHOLDERS = {"", "answer", "the answer", "your answer", "final answer", "...", "<answer>"}


def _is_placeholder(argument: str) -> bool:
    """Whether a finish[...] argument just echoes the instructions."""
    return argument.strip().strip(".").lower() in _PLACEHOLDERS


_FINISH_WRAPPER = re.compile(r"^\s*(?:Action\s*\d*\s*:\s*)?finish\s*\[(.*)\]\s*\.?\s*$", re.I)


def _unwrap_finish(answer: str) -> str:
    """'finish[Paris]' -> 'Paris' (the model sometimes writes the action as its answer)."""
    match = _FINISH_WRAPPER.match(answer)
    return match.group(1).strip() if match else answer


class ReActAgent(_SubAgentBase):
    """ReAct (Yao et al., 2023): Thought -> Action -> Observation with real tools.

    Actions are ``lookup`` (search the task context), ``calculate`` (safe
    arithmetic) and ``finish``. Tool calls appear in ``last_trace`` with zero
    tokens, since they make no LLM call.
    """

    temperature = 0.2

    def __init__(
        self,
        model_name: str | None = None,
        max_steps: int = 4,
        provider: str | None = None,
        api_key: str | None = None,
        llm_caller: LLMCallerFn | None = None,
    ) -> None:
        super().__init__(model_name, provider, api_key, llm_caller)
        self.max_steps = max_steps

    def _run_tool(self, action: str, argument: str, task: Task) -> str:
        try:
            if action == "calculate":
                return calculate(argument)
            return lookup(argument, task.context)
        except ToolError as err:
            return f"Error: {err}"

    def run(self, task: Task, token_budget: int) -> tuple[str, int]:
        """Run the Thought/Action/Observation loop until finish, max_steps or the budget."""
        self.last_trace = []
        total_tokens = 0
        history = [f"{REACT_INSTRUCTIONS}\n{_context_block(task)}Question: {task.question}\n"]
        step_cap = max(MIN_CALL_TOKENS, token_budget // (self.max_steps + 1))
        response = ""

        for step in range(1, self.max_steps + 1):
            prompt = "\n".join(history) + f"\nThought {step}:"
            outcome = self._budgeted_call(
                prompt, token_budget - total_tokens, step_cap, step=f"Thought {step}"
            )
            if outcome is None:
                break
            response, tokens = outcome
            total_tokens += tokens

            match = _ACTION_RE.search(response)
            # Keep only the model's first action; drop any Observation it invented.
            history.append(f"Thought {step}: {response[: match.end()] if match else response}")
            if match is None:
                if "final answer" in response.lower():
                    return _unwrap_finish(extract_answer(response)), total_tokens
                history.append(
                    f"Observation {step}: No valid action. "
                    "Use lookup[...], calculate[...] or finish[...]."
                )
                continue

            action, argument = match.group(1).lower(), match.group(2).strip()
            if action == "finish" and _is_placeholder(argument):
                history.append(
                    f"Observation {step}: finish[] needs the actual answer, not a placeholder."
                )
                continue
            if action == "finish":
                return argument, total_tokens
            observation = self._run_tool(action, argument, task)
            self.last_trace.append(
                {"step": f"{action}[{argument}]", "output": observation, "tokens": 0}
            )
            history.append(f"Observation {step}: {observation}")

        # Out of steps without finishing: ask for the answer if the budget allows.
        synthesis_prompt = "\n".join(history) + "\nBased on the observations above.\nFinal Answer:"
        outcome = self._budgeted_call(
            synthesis_prompt, token_budget - total_tokens, step="Synthesis"
        )
        if outcome is not None:
            response, fin_tokens = outcome
            total_tokens += fin_tokens
            return _unwrap_finish(extract_answer(f"Final Answer: {response}")), total_tokens
        return _unwrap_finish(extract_answer(response)), total_tokens


# 2. Multi-Agent Debate & Consensus Agent


class DebateAgent(_SubAgentBase):
    """Proposer-Critic Multi-Agent Debate (Du et al., 2023)."""

    temperature = 0.4

    def __init__(
        self,
        model_name: str | None = None,
        num_rounds: int = 2,
        provider: str | None = None,
        api_key: str | None = None,
        llm_caller: LLMCallerFn | None = None,
    ) -> None:
        super().__init__(model_name, provider, api_key, llm_caller)
        self.num_rounds = num_rounds

    def run(self, task: Task, token_budget: int) -> tuple[str, int]:
        """Run proposer/critic rounds; every prompt carries the task context."""
        self.last_trace = []
        total_tokens = 0
        context = _context_block(task)
        round_cap = max(MIN_CALL_TOKENS, token_budget // (self.num_rounds * 2 + 1))

        # Round 1: Proposer initial solution
        prop_prompt = (
            "You are Agent Proposer. Answer this question with complete reasoning.\n"
            f"{context}Question: {task.question}\n"
            "Conclude with Final Answer: <answer>"
        )
        outcome = self._budgeted_call(prop_prompt, token_budget, round_cap, step="Proposer")
        if outcome is None:
            return "Unknown", 0
        current_solution, tokens = outcome
        total_tokens += tokens

        for round_no in range(1, self.num_rounds + 1):
            critic_prompt = (
                "You are Agent Critic. Critique the proposed solution for errors or "
                "unverified assumptions, using the context where relevant.\n"
                f"{context}Question: {task.question}\n"
                f"Proposed Solution: {current_solution}\n"
                "Point out any flaws or confirm if correct."
            )
            outcome = self._budgeted_call(
                critic_prompt, token_budget - total_tokens, round_cap, step=f"Critic {round_no}"
            )
            if outcome is None:
                break
            critic_feedback, tokens = outcome
            total_tokens += tokens

            revision_prompt = (
                "You are Agent Proposer. Review the Critic's feedback on your previous "
                "solution and refine your answer.\n"
                f"{context}Question: {task.question}\n"
                f"Your Previous Solution: {current_solution}\n"
                f"Critic Feedback: {critic_feedback}\n"
                "Provide your final refined conclusion with Final Answer: <answer>"
            )
            outcome = self._budgeted_call(
                revision_prompt, token_budget - total_tokens, round_cap, step=f"Revision {round_no}"
            )
            if outcome is None:
                break
            current_solution, tokens = outcome
            total_tokens += tokens

        return extract_answer(current_solution), total_tokens


# 3. Reflexion Agent (Self-Reflection & Refinement)


class ReflexionAgent(_SubAgentBase):
    """Self-reflection agent that drafts, critiques and refines (Self-Refine)."""

    temperature = 0.3

    def run(self, task: Task, token_budget: int) -> tuple[str, int]:
        """Draft an initial response, self-critique, and refine it."""
        self.last_trace = []
        total_tokens = 0
        context = _context_block(task)
        step_cap = max(MIN_CALL_TOKENS, token_budget // 3)

        # 1. Draft
        draft_prompt = (
            "Answer the question step by step.\n"
            f"{context}Question: {task.question}\n"
            "End with Final Answer: <answer>\n"
            "Draft Answer:"
        )
        outcome = self._budgeted_call(draft_prompt, token_budget, step_cap, step="Draft")
        if outcome is None:
            return "Unknown", 0
        draft, tokens = outcome
        total_tokens += tokens

        # 2. Reflect
        reflect_prompt = (
            f"{context}Question: {task.question}\n"
            f"Draft: {draft}\n"
            "Critique: Are there any logical fallacies, arithmetic errors, or missed nuances?"
        )
        outcome = self._budgeted_call(
            reflect_prompt, token_budget - total_tokens, step_cap, step="Reflect"
        )
        if outcome is None:
            return extract_answer(draft), total_tokens
        reflection, tokens = outcome
        total_tokens += tokens

        # 3. Refine
        refine_prompt = (
            f"{context}Question: {task.question}\n"
            f"Draft: {draft}\n"
            f"Reflection: {reflection}\n"
            "Provide the final corrected answer. Format: Final Answer: <answer>"
        )
        outcome = self._budgeted_call(
            refine_prompt, token_budget - total_tokens, step_cap, step="Refine"
        )
        if outcome is None:
            return extract_answer(draft), total_tokens
        refined, tokens = outcome
        total_tokens += tokens

        return extract_answer(refined), total_tokens

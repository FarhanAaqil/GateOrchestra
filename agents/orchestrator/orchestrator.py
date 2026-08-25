"""
agents/orchestrator/orchestrator.py
===================================
Multi-Agent System (MAS) Orchestrator for GateOrchestra (Person 2).

Satisfies OrchestratorFn = Callable[[Task, int], tuple[str, int]].
Dynamically selects and runs sub-agent strategies (ReAct, Multi-Agent Debate, Reflexion)
while strictly adhering to the dynamic token budget cap (`k * probe_tokens`).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from agents.orchestrator.sub_agents import (
    DebateAgent,
    ReActAgent,
    ReflexionAgent,
)
from shared.config import MODEL_NAME
from shared.schemas import Task

logger = logging.getLogger(__name__)

LLMCallerFn = Callable[[str, float, int], tuple[str, int]]


class MASOrchestrator:
    """Multi-Agent System Orchestrator.

    Routes tasks to the most suitable sub-agent strategy:
      - Deep multi-hop tasks (depth >= 3 or context heavy) → ReAct Agent
      - Parallel / broad comparative tasks (parallel >= 2) → Debate Agent
      - Default complex reasoning → Reflexion Agent

    Args:
        model_name: Base LLM name.
        default_strategy: Strategy override ('react', 'debate', 'reflexion', or 'auto').
        llm_caller: Pluggable LLM caller for mock testing or custom backends.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        default_strategy: str = "auto",
        llm_caller: Optional[LLMCallerFn] = None,
    ) -> None:
        self.model_name = model_name
        self.default_strategy = default_strategy
        self.llm_caller = llm_caller

        # Instantiate sub-agent pool
        self.react_agent = ReActAgent(model_name=model_name, llm_caller=llm_caller)
        self.debate_agent = DebateAgent(model_name=model_name, llm_caller=llm_caller)
        self.reflexion_agent = ReflexionAgent(model_name=model_name, llm_caller=llm_caller)

    def select_strategy(self, task: Task) -> str:
        """Select appropriate sub-agent strategy based on task signals."""
        if self.default_strategy != "auto":
            return self.default_strategy

        # Heuristic routing based on task axes and context
        if task.depth_score is not None and task.depth_score >= 3:
            return "react"
        if task.parallel_score is not None and task.parallel_score >= 2:
            return "debate"
        if task.context and len(task.context) > 200:
            return "react"

        return "reflexion"

    def run(self, task: Task, token_budget: int) -> tuple[str, int]:
        """Execute MAS reasoning within token_budget cap.

        Returns:
            tuple of (answer_string, actual_tokens_used).
        """
        strategy = self.select_strategy(task)
        logger.info(
            f"[MASOrchestrator] Running task={task.task_id} with strategy={strategy} budget={token_budget}"
        )

        if strategy == "react":
            ans, tokens = self.react_agent.run(task, token_budget)
        elif strategy == "debate":
            ans, tokens = self.debate_agent.run(task, token_budget)
        else:
            ans, tokens = self.reflexion_agent.run(task, token_budget)

        # Cap reported tokens to budget to ensure strict contract
        safe_tokens = min(tokens, token_budget) if token_budget > 0 else tokens
        return ans, safe_tokens

    def __call__(self, task: Task, token_budget: int) -> tuple[str, int]:
        """Callable interface matching OrchestratorFn."""
        return self.run(task, token_budget)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level default orchestrator
# ─────────────────────────────────────────────────────────────────────────────

_default_orchestrator: Optional[MASOrchestrator] = None


def orchestrator(task: Task, token_budget: int) -> tuple[str, int]:
    """Functional interface matching OrchestratorFn: (Task, int) -> (str, int)."""
    global _default_orchestrator
    if _default_orchestrator is None:
        _default_orchestrator = MASOrchestrator()
    return _default_orchestrator(task, token_budget)

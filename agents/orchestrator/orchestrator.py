"""
agents/orchestrator/orchestrator.py
===================================
Multi-Agent System (MAS) Orchestrator for GateOrchestra (Person 2).

Satisfies OrchestratorFn = Callable[[Task, int], tuple[str, int]].
Dynamically routes sub-tasks across the sub-agent pool (ReAct, Multi-Agent Debate, Reflexion)
using either heuristic rule routing or an online Contextual Bandit (LinUCB) router,
while strictly enforcing the token budget cap (`k * probe_tokens`).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from agents.orchestrator.bandit_router import LinUCBRouter
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
    """Multi-Agent System Orchestrator with Contextual Bandit & Token Pacing.

    Args:
        model_name: Base LLM name.
        default_strategy: Routing strategy ('auto', 'bandit', 'react', 'debate', 'reflexion').
        llm_caller: Pluggable LLM caller for mock testing or custom backends.
        router: Optional custom LinUCBRouter instance.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        default_strategy: str = "auto",
        llm_caller: Optional[LLMCallerFn] = None,
        router: Optional[LinUCBRouter] = None,
    ) -> None:
        self.model_name = model_name
        self.default_strategy = default_strategy
        self.llm_caller = llm_caller

        # Sub-agent pool
        self.react_agent = ReActAgent(model_name=model_name, llm_caller=llm_caller)
        self.debate_agent = DebateAgent(model_name=model_name, llm_caller=llm_caller)
        self.reflexion_agent = ReflexionAgent(model_name=model_name, llm_caller=llm_caller)

        # Contextual Bandit Router
        self.bandit_router = router or LinUCBRouter()

    def select_strategy(self, task: Task) -> str:
        """Select appropriate sub-agent strategy based on task signals or LinUCB."""
        if self.default_strategy == "bandit":
            return self.bandit_router.select_arm(task)

        if self.default_strategy in ("react", "debate", "reflexion"):
            return self.default_strategy

        # Heuristic routing based on task axes and context ('auto')
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

        # Emergency Fallback & Strict Budget Cap enforcement
        safe_tokens = min(tokens, token_budget) if token_budget > 0 else tokens
        safe_ans = ans.strip() if ans and ans.strip() else "Unknown"

        return safe_ans, safe_tokens

    def update_bandit_reward(self, task: Task, chosen_strategy: str, is_correct: bool, tokens_spent: int, budget: int) -> None:
        """Update LinUCB bandit with observed reward."""
        # Reward = +1.0 for correct, -0.2 * (tokens_spent / budget) penalty
        token_penalty = 0.2 * (tokens_spent / max(1, budget))
        reward = (1.0 if is_correct else 0.0) - token_penalty
        self.bandit_router.update(task, chosen_strategy, reward)

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

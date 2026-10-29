"""
agents/orchestrator/orchestrator.py
===================================
Multi-Agent System (MAS) Orchestrator for GateOrchestra (Person 2).

Satisfies OrchestratorFn = Callable[[Task, int], tuple[str, int]].
Routes each task to one sub-agent (ReAct, Multi-Agent Debate, Reflexion) using
either a text heuristic ('auto', the default) or an online LinUCB contextual
bandit ('bandit'). Both routers use only features computable from the task
text; neither reads the dataset's depth/parallel labels.

The token budget (`k * probe_tokens`) is enforced before each LLM call, and
the returned token count is the true spend (see sub_agents.py).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path

from agents.orchestrator.bandit_router import LinUCBRouter
from agents.orchestrator.sub_agents import (
    DebateAgent,
    ReActAgent,
    ReflexionAgent,
)
from gate.feature_extractor import estimate_task_structure
from shared.config import MODEL_NAME
from shared.schemas import Task

logger = logging.getLogger(__name__)

LLMCallerFn = Callable[[str, float, int], tuple[str, int]]

_COMPARISON_CUES = re.compile(
    r"\b(?:compare|comparison|versus|vs)\b"  # explicit comparison
    r"|:\s*[^?]+\bor\b[^?]+\?",  # "Which is X: A or B?"
    flags=re.IGNORECASE,
)
AUTO_DEPTH_THRESHOLD = 3.0
AUTO_PARALLEL_THRESHOLD = 2.5


class MASOrchestrator:
    """Multi-Agent System Orchestrator with Contextual Bandit & Token Pacing.

    Args:
        model_name: Base LLM name.
        default_strategy: Routing strategy ('auto', 'bandit', 'react', 'debate', 'reflexion').
        llm_caller: Pluggable LLM caller for mock testing or custom backends.
        router: Optional custom LinUCBRouter instance.

    Attributes:
        _last_strategy: The sub-agent strategy chosen during the most recent
            ``run()`` call (``None`` before any call).  The pipeline uses this
            to feed the observed reward back to the LinUCB bandit without
            changing the public ``OrchestratorFn`` return type.
    """

    def __init__(
        self,
        model_name: str | None = None,
        default_strategy: str = "auto",
        provider: str | None = None,
        api_key: str | None = None,
        llm_caller: LLMCallerFn | None = None,
        router: LinUCBRouter | None = None,
        bandit_state_path: str | Path | None = None,
        warmstart_traces: list[dict] | list[tuple[Task, str, float]] | str | Path | None = None,
    ) -> None:
        self.model_name = model_name or MODEL_NAME
        self.default_strategy = default_strategy
        self.provider = provider
        self.api_key = api_key
        self.llm_caller = llm_caller

        # Strategy chosen by the most recent run() call; used for bandit updates.
        self._last_strategy: str | None = None
        # Sub-agent steps of the most recent run() call (see sub_agents.py).
        self._last_trace: list[dict] = []

        # Sub-agent pool
        self.react_agent = ReActAgent(
            model_name=model_name,
            provider=provider,
            api_key=api_key,
            llm_caller=llm_caller,
        )
        self.debate_agent = DebateAgent(
            model_name=model_name,
            provider=provider,
            api_key=api_key,
            llm_caller=llm_caller,
        )
        self.reflexion_agent = ReflexionAgent(
            model_name=model_name,
            provider=provider,
            api_key=api_key,
            llm_caller=llm_caller,
        )

        # Contextual Bandit Router
        self.bandit_router = router or LinUCBRouter()
        if bandit_state_path is not None:
            self.load_bandit_state(bandit_state_path)
        if warmstart_traces is not None:
            self.warmstart_bandit(warmstart_traces)

    def save_bandit_state(self, path: str | Path) -> None:
        """Persist LinUCB router state to disk."""
        self.bandit_router.save(path)

    def load_bandit_state(self, path: str | Path) -> None:
        """Load LinUCB router state from disk."""
        self.bandit_router.load(path)

    def warmstart_bandit(
        self,
        traces: list[dict] | list[tuple[Task, str, float]] | str | Path,
    ) -> int:
        """Warm-start the internal LinUCB bandit router from historical traces."""
        return self.bandit_router.warmstart_from_traces(traces)

    def select_strategy(self, task: Task) -> str:
        """Select appropriate sub-agent strategy based on task signals or LinUCB."""
        if self.default_strategy == "bandit":
            return self.bandit_router.select_arm(task)

        if self.default_strategy in ("react", "debate", "reflexion"):
            return self.default_strategy

        # 'auto': text-only heuristic (no dataset labels).
        estimated_depth, estimated_parallel = estimate_task_structure(task)
        if _COMPARISON_CUES.search(task.question) or estimated_parallel >= AUTO_PARALLEL_THRESHOLD:
            return "debate"
        if task.context or estimated_depth >= AUTO_DEPTH_THRESHOLD:
            return "react"
        return "reflexion"

    def run(self, task: Task, token_budget: int) -> tuple[str, int]:
        """Execute MAS reasoning within token_budget cap.

        Side-effect: sets ``self._last_strategy`` to the arm chosen for this
        task so that callers can pass it to ``update_bandit_reward()`` after
        the result is evaluated.

        Returns:
            tuple of (answer_string, actual_tokens_used). The token count is the
            true spend and can exceed ``token_budget`` by at most one call's
            prompt-size estimation error.

        Raises:
            SubAgentError: If an LLM call fails.
        """
        strategy = self.select_strategy(task)
        # Cache the chosen arm so the pipeline can retrieve it for the bandit update.
        self._last_strategy = strategy
        logger.info(
            f"[MASOrchestrator] Running task={task.task_id} with strategy={strategy} budget={token_budget}"
        )

        agent: ReActAgent | DebateAgent | ReflexionAgent
        if strategy == "react":
            agent = self.react_agent
        elif strategy == "debate":
            agent = self.debate_agent
        else:
            agent = self.reflexion_agent
        self._last_trace = []
        try:
            ans, tokens = agent.run(task, token_budget)
        finally:
            self._last_trace = list(agent.last_trace)

        if tokens > token_budget:
            logger.warning(
                f"[MASOrchestrator] task={task.task_id} spent {tokens} tokens, "
                f"over its budget of {token_budget}"
            )
        safe_ans = ans.strip() if ans and ans.strip() else "Unknown"
        return safe_ans, tokens

    def update_bandit_reward(
        self, task: Task, chosen_strategy: str, is_correct: bool, tokens_spent: int, budget: int
    ) -> None:
        """Update LinUCB bandit with observed reward."""
        # Reward = +1.0 for correct, -0.2 * (tokens_spent / budget) penalty
        token_penalty = 0.2 * (tokens_spent / max(1, budget))
        reward = (1.0 if is_correct else 0.0) - token_penalty
        self.bandit_router.update(task, chosen_strategy, reward)

    def __call__(self, task: Task, token_budget: int) -> tuple[str, int]:
        """Callable interface matching OrchestratorFn."""
        return self.run(task, token_budget)


# Module-level default orchestrator

_default_orchestrator: MASOrchestrator | None = None


def get_default_mas_orchestrator() -> MASOrchestrator:
    """Return the module-level default MASOrchestrator instance."""
    global _default_orchestrator
    if _default_orchestrator is None:
        _default_orchestrator = MASOrchestrator()
    return _default_orchestrator


def orchestrator(task: Task, token_budget: int) -> tuple[str, int]:
    """Functional interface matching OrchestratorFn: (Task, int) -> (str, int)."""
    return get_default_mas_orchestrator()(task, token_budget)

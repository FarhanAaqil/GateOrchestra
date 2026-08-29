"""
agents/orchestrator/bandit_router.py
====================================
Contextual Multi-Armed Bandit (LinUCB) Sub-Agent Router (Person 2).

Learns to dynamically route evaluation tasks to the optimal sub-agent strategy
(ReAct, Multi-Agent Debate, Reflexion) to maximize accuracy-per-token.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from shared.schemas import Task

logger = logging.getLogger(__name__)


class LinUCBRouter:
    """Disjoint Linear Upper Confidence Bound (LinUCB) Router for Sub-Agent Selection.

    Args:
        arms: List of available strategy names (defaults to ['react', 'debate', 'reflexion']).
        alpha: Exploration parameter (higher alpha -> more exploration of under-tested sub-agents).
        feature_dim: Dimensionality of the task context feature vector (default: 6).
    """

    def __init__(
        self,
        arms: list[str] | None = None,
        alpha: float = 0.5,
        feature_dim: int = 6,
    ) -> None:
        self.arms = arms or ["react", "debate", "reflexion"]
        self.alpha = alpha
        self.d = feature_dim

        # LinUCB state matrices for each arm
        # A_a = d x d identity matrix, b_a = d-dimensional zero vector
        self.A: dict[str, np.ndarray] = {
            arm: np.identity(self.d, dtype=np.float64) for arm in self.arms
        }
        self.b: dict[str, np.ndarray] = {
            arm: np.zeros((self.d, 1), dtype=np.float64) for arm in self.arms
        }

    def extract_context_features(self, task: Task) -> np.ndarray:
        """Extract standardized feature vector x in R^d from a Task."""
        word_count = len(task.question.split())
        has_context = 1.0 if task.context else 0.0
        context_len = len(task.context) if task.context else 0
        depth = float(task.depth_score) if task.depth_score is not None else 2.0
        parallel = float(task.parallel_score) if task.parallel_score is not None else 1.0

        feat = np.array(
            [
                1.0,  # Bias / intercept
                min(1.0, word_count / 50.0),
                has_context,
                min(1.0, context_len / 500.0),
                min(1.0, depth / 5.0),
                min(1.0, parallel / 4.0),
            ],
            dtype=np.float64,
        ).reshape((self.d, 1))

        return feat

    def select_arm(self, task: Task) -> str:
        """Select the best sub-agent arm for the given task using Upper Confidence Bounds."""
        x = self.extract_context_features(task)
        best_arm = self.arms[0]
        max_p = -float("inf")

        for arm in self.arms:
            A_inv = np.linalg.inv(self.A[arm])
            theta_hat = A_inv @ self.b[arm]

            # UCB score = expected reward + exploration bonus
            mean = float((theta_hat.T @ x).item())
            var = float(np.sqrt((x.T @ A_inv @ x).item()))
            p = mean + self.alpha * var

            if p > max_p:
                max_p = p
                best_arm = arm

        logger.debug(
            f"[LinUCBRouter] task={task.task_id} chosen_arm={best_arm} ucb_score={max_p:.3f}"
        )
        return best_arm

    def update(self, task: Task, arm: str, reward: float) -> None:
        """Update LinUCB state with observed reward for the chosen arm.

        Args:
            task: The evaluated task.
            arm: The arm executed ('react', 'debate', 'reflexion').
            reward: Scalar reward signal (e.g. 1.0 for correct answer minus token cost penalty).
        """
        if arm not in self.A:
            return

        x = self.extract_context_features(task)
        self.A[arm] += x @ x.T
        self.b[arm] += reward * x

    def save(self, path: str | Path) -> None:
        """Persist router weights to disk."""
        data = {
            "arms": self.arms,
            "alpha": self.alpha,
            "d": self.d,
            "A": {arm: self.A[arm].tolist() for arm in self.arms},
            "b": {arm: self.b[arm].tolist() for arm in self.arms},
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        """Load router weights from disk."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.arms = data["arms"]
        self.alpha = data["alpha"]
        self.d = data["d"]
        self.A = {arm: np.array(mat, dtype=np.float64) for arm, mat in data["A"].items()}
        self.b = {arm: np.array(vec, dtype=np.float64) for arm, vec in data["b"].items()}

"""
agents/probe_agent.py
=====================
CoT-SC (Chain-of-Thought with Self-Consistency) Probe Agent.

Capabilities:
  - Majority-locked early exit (optional): stops sampling once the leading
    answer already holds a strict majority of n_samples, so the remaining
    samples cannot change the vote.
  - Majority voting with the canonical grader's equivalence
    (shared.grading.answers_equivalent): "Paris" and "Paris, France" agree,
    "30 cm" and "60 cm" do not.
  - Failed LLM calls are failures, not votes: they lower consistency and, if
    every sample fails, ProbeError is raised.
  - Pluggable LLM callers and zero-shot CoT prompting.
  - Generates validated ProbeResult conforming to shared/schemas.py.

Token budget: ``token_budget`` is the completion-token budget for the whole
probe, split evenly across samples. ``ProbeResult.tokens_used`` reports the
provider's true total (prompt + completion) for every sample.

Usage::

    from agents.probe_agent import ProbeAgent, probe_agent
    from shared.schemas import Task

    task = Task(task_id="demo_001", question="What is 2 + 2?")
    agent = ProbeAgent(early_exit=True)
    result = agent.run(task)
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable

from agents.providers import (
    default_llm_caller,
)
from shared.config import (
    COT_SC_N_SAMPLES,
    COT_SC_TEMPERATURE,
    GROQ_API_BASE,
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
    LLM_PROVIDER,
    MODEL_API_BASE,
    MODEL_NAME,
    PROBE_TOKEN_BUDGET,
)
from shared.grading import answers_equivalent
from shared.schemas import ProbeResult, Task

logger = logging.getLogger(__name__)

# LLM caller signature: (prompt, temperature, max_tokens) -> (response_text, tokens_used)
LLMCallerFn = Callable[[str, float, int], tuple[str, int]]


# Answer Extraction and Normalization Helpers


def normalize_answer(text: str) -> str:
    """Normalize answer string for robust majority voting comparison."""
    if not text:
        return ""
    text = text.lower().strip()
    # Remove markdown bold/italic and surrounding quotes/punctuation
    text = re.sub(r"[*_`'\"]", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def extract_answer(raw_output: str) -> str:
    """Extract candidate final answer from a CoT reasoning trace.

    Looks for patterns like:
      - Final Answer: <ans>
      - The answer is: <ans>
      - #### <ans>
      - \boxed{<ans>}
      - Answer: <ans>
    Falls back to the last non-empty line if no explicit pattern is found.
    """
    if not raw_output or not raw_output.strip():
        return "Unknown"

    cleaned = raw_output.strip()

    # Strip reasoning tags if present
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    if not cleaned:
        cleaned = raw_output.strip()

    # Pattern priority list
    patterns = [
        r"(?:final\s+answer|the\s+final\s+answer\s+is)\s*[:=]?\s*([^\n\r]+)",
        r"(?:the\s+answer\s+is|answer)\s*[:=]?\s*([^\n\r]+)",
        r"\\boxed\{([^}]+)\}",
        r"####\s*([^\n\r]+)",
        r"\*\*answer\*\*\s*[:=]?\s*([^\n\r]+)",
    ]

    for pat in patterns:
        matches = list(re.finditer(pat, cleaned, flags=re.IGNORECASE))
        if matches:
            ans = matches[-1].group(1).strip()
            # Clean trailing periods / markdown / quotes
            ans = re.sub(r"^[*\s:=\"']+|[*\s.\"']+$", "", ans)
            if ans:
                return ans

    # Fallback: take the last non-empty line
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines:
        last_line = lines[-1]
        last_line = re.sub(r"^[*\s:=\"']+|[*\s.\"']+$", "", last_line)
        if last_line:
            return last_line

    return cleaned


def build_cot_prompt(task: Task) -> str:
    """Construct a Chain-of-Thought prompt for a given Task."""
    parts: list[str] = []
    if task.context:
        parts.append(f"Context:\n{task.context.strip()}\n")
    parts.append(f"Question: {task.question.strip()}")
    parts.append(
        "Please reason step by step and provide your final answer clearly on a new line in the format:\n"
        "Final Answer: <concise answer>"
    )
    return "\n\n".join(parts)


class ProbeError(RuntimeError):
    """Raised when no probe sample produced a usable response."""


# ProbeAgent Class


class ProbeAgent:
    """Chain-of-Thought Self-Consistency (CoT-SC) Probe Agent.

    Features:
      - Multi-Provider Support: Works with local Ollama or Groq Cloud.
      - Majority-locked early exit: once the leading answer holds a strict
        majority of n_samples, the remaining samples are skipped.
      - Equivalence-based voting via shared.grading.answers_equivalent.

    Args:
        model_name: Name of the LLM (defaults to config.MODEL_NAME or GROQ_MODEL_NAME).
        api_base: Base URL for LLM API (defaults to config.MODEL_API_BASE or GROQ_API_BASE).
        n_samples: Number of CoT samples for self-consistency (defaults to config.COT_SC_N_SAMPLES).
        temperature: Sampling temperature (defaults to config.COT_SC_TEMPERATURE).
        token_budget: Completion-token budget for the whole probe, split evenly across
            samples (defaults to config.PROBE_TOKEN_BUDGET).
        early_exit: Stop sampling once the majority answer can no longer change.
        provider: 'ollama' | 'groq' (defaults to config.LLM_PROVIDER).
        api_key: Optional Groq API key (defaults to config.GROQ_API_KEY).
        llm_caller: Optional custom callable (prompt, temp, budget) -> (text, tokens).
    """

    def __init__(
        self,
        model_name: str | None = None,
        api_base: str | None = None,
        n_samples: int = COT_SC_N_SAMPLES,
        temperature: float = COT_SC_TEMPERATURE,
        token_budget: int = PROBE_TOKEN_BUDGET,
        early_exit: bool = False,
        provider: str | None = None,
        api_key: str | None = None,
        llm_caller: LLMCallerFn | None = None,
    ) -> None:
        if n_samples < 1:
            raise ValueError(f"n_samples must be ≥ 1, got {n_samples}")
        if temperature < 0.0:
            raise ValueError(f"temperature must be ≥ 0.0, got {temperature}")
        if token_budget < 1:
            raise ValueError(f"token_budget must be ≥ 1, got {token_budget}")

        self.provider = provider or LLM_PROVIDER
        self.api_key = api_key or GROQ_API_KEY
        if self.provider == "groq":
            self.model_name = model_name or GROQ_MODEL_NAME
            self.api_base = api_base or GROQ_API_BASE
        else:
            self.model_name = model_name or MODEL_NAME
            self.api_base = api_base or MODEL_API_BASE

        self.n_samples = n_samples
        self.temperature = temperature
        self.token_budget = token_budget
        self.early_exit = early_exit
        self._llm_caller = llm_caller

    def _call(self, prompt: str, sample_budget: int) -> tuple[str, int]:
        """Execute a single LLM generation call."""
        if self._llm_caller is not None:
            return self._llm_caller(prompt, self.temperature, sample_budget)
        return default_llm_caller(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=sample_budget,
            model_name=self.model_name,
            api_base=self.api_base,
            provider=self.provider,
            api_key=self.api_key,
        )

    def run(self, task: Task) -> ProbeResult:
        """Run CoT-SC probing on the provided Task.

        Returns:
            ProbeResult with majority answer, consistency_score, tokens_used, etc.

        Raises:
            ProbeError: If every sample failed.
        """
        start_time = time.perf_counter()
        prompt = build_cot_prompt(task)
        sample_budget = max(1, self.token_budget // self.n_samples)

        raw_outputs: list[str] = []
        extracted_answers: list[str] = []  # successful samples only
        failures: list[str] = []
        total_tokens = 0

        for i in range(self.n_samples):
            try:
                response_text, tokens = self._call(prompt, sample_budget)
            except Exception as err:
                logger.error(f"[ProbeAgent] Sample {i+1}/{self.n_samples} failed: {err}")
                failures.append(f"{type(err).__name__}: {err}")
                raw_outputs.append(f"Error: {type(err).__name__}: {err}")
                continue

            total_tokens += tokens
            if not response_text.strip():
                raw_outputs.append("No response")
                continue
            raw_outputs.append(response_text.strip())
            extracted_answers.append(extract_answer(response_text))

            if self.early_exit and self._majority_locked(extracted_answers):
                logger.debug(
                    f"[ProbeAgent] Early exit after {i+1}/{self.n_samples} samples: "
                    "majority can no longer change"
                )
                break

        if not extracted_answers:
            raise ProbeError(
                f"All {len(raw_outputs)} probe samples failed for task_id={task.task_id!r}: "
                f"{failures[-1] if failures else 'empty responses'}"
            )

        # Failed or empty samples count against agreement (denominator = samples drawn).
        majority_answer, agreeing = self._majority_cluster(extracted_answers)
        consistency_score = agreeing / len(raw_outputs)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return ProbeResult(
            task_id=task.task_id,
            answer=majority_answer,
            consistency_score=consistency_score,
            tokens_used=total_tokens,
            latency_ms=latency_ms,
            raw_outputs=raw_outputs,
            model_name=self.model_name,
        )

    def _majority_locked(self, answers: list[str]) -> bool:
        """True once the leading answer holds a strict majority of n_samples."""
        if len(answers) < 2:
            return False
        _, leading = self._majority_cluster(answers)
        return leading * 2 > self.n_samples

    def __call__(self, task: Task) -> ProbeResult:
        """Allow ProbeAgent instance to be used directly as a Callable[[Task], ProbeResult]."""
        return self.run(task)

    @staticmethod
    def _majority_cluster(answers: list[str]) -> tuple[str, int]:
        """Group equivalent answers; return (representative answer, cluster size).

        Ties go to the cluster seen first.
        """
        clusters: list[list[str]] = []
        for answer in answers:
            for cluster in clusters:
                if answers_equivalent(answer, cluster[0]):
                    cluster.append(answer)
                    break
            else:
                clusters.append([answer])
        largest = max(clusters, key=len)
        return largest[0], len(largest)

    @classmethod
    def _majority_vote(cls, answers: list[str]) -> tuple[str, float]:
        """Return (majority_answer, fraction of answers that agree with it)."""
        if not answers:
            return "Unknown", 0.0
        answer, size = cls._majority_cluster(answers)
        return answer, size / len(answers)


# Module-level convenience function (satisfies ProbeAgentFn interface)

_default_agent: ProbeAgent | None = None


def probe_agent(task: Task) -> ProbeResult:
    """Functional interface matching ProbeAgentFn: (Task) -> ProbeResult."""
    global _default_agent
    if _default_agent is None:
        _default_agent = ProbeAgent()
    return _default_agent(task)

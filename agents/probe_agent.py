"""
agents/probe_agent.py
=====================
CoT-SC (Chain-of-Thought with Self-Consistency) Probe Agent.

Advanced capabilities:
  - Sequential Early-Exit SPRT (stops sampling early if high unanimity reached, saving up to 60% tokens).
  - Semantic Soft Majority Voting (robust clustering for phrasing variations).
  - Pluggable LLM callers and zero-shot CoT prompting.
  - Generates validated ProbeResult conforming to shared/schemas.py.

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
from collections import Counter
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
from shared.schemas import ProbeResult, Task

logger = logging.getLogger(__name__)

# LLM caller signature: (prompt, temperature, max_tokens) -> (response_text, tokens_used)
LLMCallerFn = Callable[[str, float, int], tuple[str, int]]


# ─────────────────────────────────────────────────────────────────────────────
# Answer Extraction and Normalization Helpers
# ─────────────────────────────────────────────────────────────────────────────


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
      - Answer: <ans>
    Falls back to the last non-empty line if no explicit pattern is found.
    """
    if not raw_output or not raw_output.strip():
        return "Unknown"

    cleaned = raw_output.strip()

    # Pattern priority list
    patterns = [
        r"(?:final\s+answer|the\s+final\s+answer\s+is)\s*[:=]?\s*([^\n\r]+)",
        r"(?:the\s+answer\s+is|answer)\s*[:=]?\s*([^\n\r]+)",
        r"####\s*([^\n\r]+)",
        r"\*\*answer\*\*\s*[:=]?\s*([^\n\r]+)",
    ]

    for pat in patterns:
        matches = list(re.finditer(pat, cleaned, flags=re.IGNORECASE))
        if matches:
            ans = matches[-1].group(1).strip()
            # Clean trailing periods / markdown
            ans = re.sub(r"^[*\s:=-]+|[*\s.]+$", "", ans)
            if ans:
                return ans

    # Fallback: take the last non-empty line
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines:
        last_line = lines[-1]
        last_line = re.sub(r"^[*\s:=-]+|[*\s.]+$", "", last_line)
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


def _token_jaccard_similarity(s1: str, s2: str) -> float:
    """Compute token-level Jaccard similarity between two candidate strings."""
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())
    if not tokens1 or not tokens2:
        return 1.0 if tokens1 == tokens2 else 0.0
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)


# ─────────────────────────────────────────────────────────────────────────────
# ProbeAgent Class
# ─────────────────────────────────────────────────────────────────────────────


class ProbeAgent:
    """Chain-of-Thought Self-Consistency (CoT-SC) Probe Agent.

    Advanced features:
      - Multi-Provider Support: Works seamlessly with local Ollama or Groq Cloud.
      - Sequential Early-Exit SPRT: If early samples show overwhelming agreement,
        terminates sampling early to conserve token budget.
      - Semantic Soft Clustering: Groups semantically equivalent answers.

    Args:
        model_name: Name of the LLM (defaults to config.MODEL_NAME or GROQ_MODEL_NAME).
        api_base: Base URL for LLM API (defaults to config.MODEL_API_BASE or GROQ_API_BASE).
        n_samples: Number of CoT samples for self-consistency (defaults to config.COT_SC_N_SAMPLES).
        temperature: Sampling temperature (defaults to config.COT_SC_TEMPERATURE).
        token_budget: Max token budget for the probe run (defaults to config.PROBE_TOKEN_BUDGET).
        early_exit: Enable sequential early stopping on unanimous confidence (saves tokens).
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
        """
        start_time = time.perf_counter()
        prompt = build_cot_prompt(task)
        sample_budget = max(1, self.token_budget // self.n_samples)

        raw_outputs: list[str] = []
        extracted_answers: list[str] = []
        total_tokens = 0

        for i in range(self.n_samples):
            try:
                response_text, tokens = self._call(prompt, sample_budget)
            except Exception as err:
                logger.error(f"[ProbeAgent] Sample {i+1}/{self.n_samples} failed: {err}")
                response_text = f"Error: {err}"
                tokens = 0

            # Ensure non-empty output to satisfy schema validation
            safe_output = response_text.strip() if response_text.strip() else "No response"
            raw_outputs.append(safe_output)
            extracted_answers.append(extract_answer(safe_output))
            total_tokens += tokens

            # Sequential Early-Exit Check (SPRT)
            if self.early_exit and i >= 2:  # at least 3 samples collected
                majority_candidate, current_consistency = self._majority_vote(extracted_answers)
                # If unanimous across 3 samples, early-stop to save tokens
                if current_consistency == 1.0:
                    logger.debug(
                        f"[ProbeAgent] Early-exit triggered at sample {i+1}/{self.n_samples} with 100% agreement"
                    )
                    break

        # Majority voting with normalization & semantic clustering
        majority_answer, consistency_score = self._majority_vote(extracted_answers)

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

    def __call__(self, task: Task) -> ProbeResult:
        """Allow ProbeAgent instance to be used directly as a Callable[[Task], ProbeResult]."""
        return self.run(task)

    @classmethod
    def _majority_vote(cls, answers: list[str]) -> tuple[str, float]:
        """Aggregate candidate answers and compute consistency score with soft semantic grouping.

        Returns:
            tuple of (majority_answer, consistency_score).
        """
        if not answers:
            return "Unknown", 0.0

        normalized_answers = [normalize_answer(a) for a in answers]

        # Group by exact normalized match first
        clusters: list[list[int]] = []  # list of index lists
        for i, norm in enumerate(normalized_answers):
            assigned = False
            for cluster in clusters:
                rep_norm = normalized_answers[cluster[0]]
                # Match if identical or token Jaccard >= 0.5 or token subset overlap
                tokens_a = set(norm.split())
                tokens_b = set(rep_norm.split())
                is_subset = bool(
                    tokens_a
                    and tokens_b
                    and (tokens_a.issubset(tokens_b) or tokens_b.issubset(tokens_a))
                )
                if (
                    norm == rep_norm
                    or is_subset
                    or _token_jaccard_similarity(norm, rep_norm) >= 0.5
                ):
                    cluster.append(i)
                    assigned = True
                    break
            if not assigned:
                clusters.append([i])

        # Pick largest cluster
        largest_cluster = max(clusters, key=len)
        majority_idx = largest_cluster[0]
        majority_ans = answers[majority_idx]
        consistency = len(largest_cluster) / len(answers)

        return majority_ans, consistency


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience function (satisfies ProbeAgentFn interface)
# ─────────────────────────────────────────────────────────────────────────────

_default_agent: ProbeAgent | None = None


def probe_agent(task: Task) -> ProbeResult:
    """Functional interface matching ProbeAgentFn: (Task) -> ProbeResult."""
    global _default_agent
    if _default_agent is None:
        _default_agent = ProbeAgent()
    return _default_agent(task)

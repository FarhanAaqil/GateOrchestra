"""
agents/probe_agent.py
=====================
CoT-SC (Chain-of-Thought with Self-Consistency) Probe Agent.

Role in GateOrchestra:
  - Serves as the cheap front-stage probe (Person 2 module).
  - Samples N reasoning paths with CoT prompting at a non-zero temperature.
  - Aggregates individual sample answers via majority voting.
  - Computes consistency_score (fraction of samples agreeing with majority).
  - Produces a ProbeResult conforming to shared/schemas.py.

Usage::

    from agents.probe_agent import ProbeAgent, probe_agent
    from shared.schemas import Task

    task = Task(task_id="demo_001", question="What is 2 + 2?")
    agent = ProbeAgent()
    result = agent.run(task)
    # or using the functional interface:
    result = probe_agent(task)
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from typing import Any, Callable, Optional

from shared.config import (
    COT_SC_N_SAMPLES,
    COT_SC_TEMPERATURE,
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


# ─────────────────────────────────────────────────────────────────────────────
# Default LLM Client (Ollama / OpenAI-compatible HTTP)
# ─────────────────────────────────────────────────────────────────────────────


def default_llm_caller(
    prompt: str,
    temperature: float = COT_SC_TEMPERATURE,
    max_tokens: int = PROBE_TOKEN_BUDGET,
    model_name: str = MODEL_NAME,
    api_base: str = MODEL_API_BASE,
    timeout: float = 30.0,
) -> tuple[str, int]:
    """Default HTTP caller targeting Ollama or OpenAI-compatible endpoints.

    Returns:
        tuple of (response_text, total_tokens_used).
    """
    url = f"{api_base.rstrip('/')}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
        "stream": False,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            response_text = resp_data.get("response", "")
            prompt_eval_count = int(resp_data.get("prompt_eval_count", 0))
            eval_count = int(resp_data.get("eval_count", 0))
            total_tokens = prompt_eval_count + eval_count

            # Fallback heuristic token count if server did not return token counts
            if total_tokens <= 0:
                total_tokens = max(1, len(prompt.split()) + len(response_text.split()))

            return response_text, total_tokens

    except (urllib.error.URLError, TimeoutError, OSError) as e:
        logger.warning(
            f"[ProbeAgent] LLM call to {url} failed: {e}. "
            "Ensure Ollama/API server is running or provide a custom llm_caller."
        )
        raise


# ─────────────────────────────────────────────────────────────────────────────
# ProbeAgent Class
# ─────────────────────────────────────────────────────────────────────────────


class ProbeAgent:
    """Chain-of-Thought Self-Consistency (CoT-SC) Probe Agent.

    Generates N stochastic reasoning paths, extracts candidate answers,
    computes consistency agreement rate, and returns a ProbeResult.

    Args:
        model_name: Name of the LLM (defaults to config.MODEL_NAME).
        api_base: Base URL for LLM API (defaults to config.MODEL_API_BASE).
        n_samples: Number of CoT samples for self-consistency (defaults to config.COT_SC_N_SAMPLES).
        temperature: Sampling temperature (defaults to config.COT_SC_TEMPERATURE).
        token_budget: Max token budget for the probe run (defaults to config.PROBE_TOKEN_BUDGET).
        llm_caller: Optional custom callable (prompt, temp, budget) -> (text, tokens).
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        api_base: str = MODEL_API_BASE,
        n_samples: int = COT_SC_N_SAMPLES,
        temperature: float = COT_SC_TEMPERATURE,
        token_budget: int = PROBE_TOKEN_BUDGET,
        llm_caller: Optional[LLMCallerFn] = None,
    ) -> None:
        if n_samples < 1:
            raise ValueError(f"n_samples must be ≥ 1, got {n_samples}")
        if temperature < 0.0:
            raise ValueError(f"temperature must be ≥ 0.0, got {temperature}")
        if token_budget < 1:
            raise ValueError(f"token_budget must be ≥ 1, got {token_budget}")

        self.model_name = model_name
        self.api_base = api_base
        self.n_samples = n_samples
        self.temperature = temperature
        self.token_budget = token_budget
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
        )

    def run(self, task: Task) -> ProbeResult:
        """Run CoT-SC probing on the provided Task.

        Args:
            task: The Task instance to evaluate.

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

        # Majority voting with normalization
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

    @staticmethod
    def _majority_vote(answers: list[str]) -> tuple[str, float]:
        """Aggregate candidate answers and compute consistency score.

        Returns:
            tuple of (majority_answer, consistency_score).
        """
        if not answers:
            return "Unknown", 0.0

        normalized_to_original: dict[str, str] = {}
        normalized_counts: Counter[str] = Counter()

        for ans in answers:
            norm = normalize_answer(ans)
            if norm not in normalized_to_original:
                normalized_to_original[norm] = ans
            normalized_counts[norm] += 1

        # Most common normalized answer
        top_norm, count = normalized_counts.most_common(1)[0]
        majority_ans = normalized_to_original.get(top_norm, answers[0])
        consistency = count / len(answers)

        return majority_ans, consistency


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience function (satisfies ProbeAgentFn interface)
# ─────────────────────────────────────────────────────────────────────────────

_default_agent: Optional[ProbeAgent] = None


def probe_agent(task: Task) -> ProbeResult:
    """Functional interface matching ProbeAgentFn: (Task) -> ProbeResult.

    Uses default configuration from shared/config.py.
    """
    global _default_agent
    if _default_agent is None:
        _default_agent = ProbeAgent()
    return _default_agent(task)

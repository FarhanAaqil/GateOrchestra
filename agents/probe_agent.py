"""Real CoT-SC probe agent backed by the local Ollama HTTP API."""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from shared.config import (
    COT_SC_N_SAMPLES,
    COT_SC_TEMPERATURE,
    MODEL_API_BASE,
)
from shared.schemas import ProbeResult, Task

logger = logging.getLogger(__name__)

MODEL_NAME = "qwen2.5:7b-instruct"
OLLAMA_TIMEOUT_SECONDS = 120


class OllamaProbeError(RuntimeError):
    """Raised when Ollama cannot produce a probe sample."""


def probe_agent(task: Task) -> ProbeResult:
    """Generate independent CoT samples and return their majority answer.

    Each request is sent independently with non-zero temperature. Responses
    are reduced to final answers before voting, while the original generated
    text is retained in ``raw_outputs`` for inspection.
    """
    started = time.perf_counter()
    outputs: list[str] = []
    answers: list[str] = []
    total_tokens = 0

    for sample_index in range(COT_SC_N_SAMPLES):
        response = _generate_sample(task)
        output = _response_text(response)
        if not output.strip():
            raise OllamaProbeError(
                f"Ollama returned an empty response for sample {sample_index + 1} "
                f"of task {task.task_id!r}"
            )

        outputs.append(output)
        answers.append(_extract_answer(output))
        total_tokens += _response_tokens(response)

    counts = Counter(_vote_key(answer) for answer in answers)
    majority_key, majority_count = counts.most_common(1)[0]
    majority_answer = next(answer for answer in answers if _vote_key(answer) == majority_key)

    return ProbeResult(
        task_id=task.task_id,
        answer=majority_answer,
        consistency_score=majority_count / len(answers),
        tokens_used=total_tokens,
        latency_ms=(time.perf_counter() - started) * 1000,
        raw_outputs=outputs,
        model_name=MODEL_NAME,
    )


def _generate_sample(task: Task) -> dict:
    """Call Ollama once; separate calls provide independent CoT samples."""
    prompt = _build_prompt(task)
    payload = json.dumps(
        {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": COT_SC_TEMPERATURE},
        }
    ).encode("utf-8")
    request = Request(
        f"{MODEL_API_BASE.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise OllamaProbeError(
            f"Unable to generate probe sample from Ollama at {MODEL_API_BASE}: {exc}"
        ) from exc

    if not isinstance(result, dict):
        raise OllamaProbeError("Ollama returned a response that was not a JSON object")
    if result.get("error"):
        raise OllamaProbeError(f"Ollama returned an error: {result['error']}")
    return result


def _build_prompt(task: Task) -> str:
    context = task.context.strip() if task.context else "(no supporting context)"
    return (
        "Solve the following question carefully. Think through the reasoning "
        "internally, then end with exactly one concise line in the form "
        "'Final answer: <answer>'. Do not include multiple possible answers.\n\n"
        f"Question: {task.question}\n"
        f"Context: {context}"
    )


def _response_text(response: dict) -> str:
    output = response.get("response")
    if not isinstance(output, str):
        raise OllamaProbeError("Ollama response did not contain a text 'response' field")
    return output.strip()


def _response_tokens(response: dict) -> int:
    """Use Ollama's prompt and completion counts when supplied."""
    prompt_tokens = response.get("prompt_eval_count", 0)
    completion_tokens = response.get("eval_count", 0)
    return sum(value for value in (prompt_tokens, completion_tokens) if isinstance(value, int))


def _extract_answer(output: str) -> str:
    """Extract the requested final-answer line, with a conservative fallback."""
    matches = re.findall(r"(?im)^\s*final\s+answer\s*:\s*(.+?)\s*$", output)
    answer = matches[-1] if matches else output.splitlines()[-1].strip()
    answer = re.sub(r"^[-*`]\s*|[`*_]+$", "", answer).strip()
    if not answer:
        raise OllamaProbeError("Ollama produced no extractable final answer")
    return answer


def _vote_key(answer: str) -> str:
    """Normalize only casing, punctuation, and whitespace for agreement voting."""
    normalized = re.sub(r"[^\w\s]", "", answer.casefold())
    return " ".join(normalized.split())

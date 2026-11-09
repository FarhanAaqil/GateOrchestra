"""
assistant/probe.py
==================
The chat probe: a few short, cheap answers to the same message, run in
parallel, whose agreement tells the gate how sure a single model call is.

Two samples go out at once. If they agree, that is the answer (no third call);
if not, a third breaks the tie. Each sample also labels the request type, so
small talk and creative requests (where agreement means nothing) skip the gate.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from agents.probe_agent import ProbeAgent, answers_equivalent, extract_answer
from assistant.llm import Completion, Gateway, Message, ProviderError, json_generation_failed
from shared.schemas import ProbeResult

PROBE_SYSTEM = (
    "Classify the user's latest message and answer it as briefly as possible. Reply with JSON only: "
    '{"type": "factual|math|code|reasoning|creative|conversation", '
    '"answer": "<shortest correct answer: a word, number, name or one sentence>"}'
)
OPEN_ENDED = {"creative", "conversation"}
TYPES = {"factual", "math", "code", "reasoning", "creative", "conversation"}
_TYPE = re.compile(r"type\W*([a-z]+)", re.IGNORECASE)


@dataclass
class ChatProbe:
    result: ProbeResult
    task_type: str
    completions: list[Completion]

    @property
    def tokens(self) -> int:
        return sum(c.total_tokens for c in self.completions)

    @property
    def open_ended(self) -> bool:
        return self.task_type in OPEN_ENDED


def parse_sample(text: str) -> tuple[str, str]:
    """(type, answer) from a JSON probe reply; tolerant of models that ignore JSON mode."""
    try:
        data = json.loads(text)
        kind = str(data.get("type", "")).strip().lower()
        answer = str(data.get("answer", "")).strip()
    except (json.JSONDecodeError, AttributeError):
        match = _TYPE.search(text or "")
        kind = match.group(1).lower() if match else ""
        answer = extract_answer(text)
    return (kind if kind in TYPES else "factual"), (answer or "Unknown")


def _compact_history(history: list[Message], limit: int = 4) -> list[Message]:
    out = []
    for m in history[-limit:]:
        if m["role"] in ("user", "assistant") and m.get("content"):
            out.append({"role": m["role"], "content": str(m["content"])[:600]})
    return out


def run_probe(
    gateway: Gateway,
    question: str,
    history: list[Message],
    *,
    samples: int = 3,
    max_tokens: int = 400,
    context: str | None = None,
    task_id: str = "chat",
) -> ChatProbe:
    """Ask the probe model 2 (or 3) times and measure agreement."""
    start = time.perf_counter()
    user = question if not context else f"{question}\n\n(Reference material is available.)"
    messages: list[Message] = [
        {"role": "system", "content": PROBE_SYSTEM},
        *_compact_history(history),
        {"role": "user", "content": user},
    ]

    def sample(_: int) -> Completion:
        try:
            return gateway.complete(
                "probe", messages, max_tokens=max_tokens, temperature=0.8, json_mode=True
            )
        except ProviderError as err:
            if not json_generation_failed(err):
                raise
        # A reasoning model can spend the whole budget thinking before it writes the JSON
        # (likelier on long requests). Retry with room to finish, then without JSON mode,
        # which parse_sample tolerates.
        try:
            return gateway.complete(
                "probe", messages, max_tokens=max_tokens * 3, temperature=0.8, json_mode=True
            )
        except ProviderError as err:
            if not json_generation_failed(err):
                raise
        return gateway.complete(
            "probe", messages, max_tokens=max_tokens * 3, temperature=0.8, json_mode=False
        )

    first_round = min(2, samples)
    with ThreadPoolExecutor(max_workers=first_round) as pool:
        completions = list(pool.map(sample, range(first_round)))
    parsed = [parse_sample(c.text) for c in completions]
    types = [t for t, _ in parsed]
    answers = [a for _, a in parsed]

    agree = len(answers) >= 2 and answers_equivalent(answers[0], answers[1])
    if not agree and samples > first_round and not all(t in OPEN_ENDED for t in types):
        completions.append(sample(first_round))
        kind, answer = parse_sample(completions[-1].text)
        types.append(kind)
        answers.append(answer)

    majority, agreeing = ProbeAgent._majority_cluster(answers)
    task_type = max(set(types), key=types.count)
    result = ProbeResult(
        task_id=task_id,
        answer=majority,
        consistency_score=agreeing / len(answers),
        tokens_used=max(1, sum(c.total_tokens for c in completions)),
        latency_ms=(time.perf_counter() - start) * 1000,
        raw_outputs=answers,
        model_name=completions[0].model if completions else None,
    )
    return ChatProbe(result=result, task_type=task_type, completions=completions)

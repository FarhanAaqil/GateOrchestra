"""
assistant/agent.py
==================
The ESCALATE path: a tool-using research agent.

It calls tools through native function calling (Groq), for up to
``max_rounds`` rounds, and returns research notes: facts with sources,
computed values, code that was run. The final user-facing answer is written
afterwards by the answer model from these notes. Providers without tool
support simply return notes from the model's own knowledge.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from agents.orchestrator.tools import ToolError, calculate
from assistant.llm import Completion, Gateway, Message

logger = logging.getLogger(__name__)

AGENT_SYSTEM = (
    "You are the research step of an assistant. Use the tools when they make the answer more "
    "accurate: calculate for arithmetic, search_wikipedia for facts you are not sure of or that "
    "may have changed, run_python to test code before recommending it, search_documents for the "
    "user's uploaded files. Then reply with concise research notes: the facts (with source "
    "titles or URLs), computed values, and any code you verified with its output. Do not write "
    "the final answer to the user."
)
MAX_TOOL_OUTPUT = 4000


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[[dict[str, Any]], str]

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def calculator_tool() -> Tool:
    def run(args: dict[str, Any]) -> str:
        try:
            return calculate(str(args.get("expression", "")))
        except ToolError as err:
            return f"Error: {err}"

    return Tool(
        name="calculate",
        description="Evaluate an arithmetic expression exactly, e.g. '20000 * 0.85**2' or 'sqrt(2)'.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Arithmetic expression"}
            },
            "required": ["expression"],
        },
        run=run,
    )


@dataclass
class AgentResult:
    notes: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    completions: list[Completion] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(c.total_tokens for c in self.completions)


def _assistant_message(completion: Completion) -> Message:
    """The assistant turn to send back with tool results (only fields the API accepts)."""
    raw = completion.raw_message or {}
    return {
        "role": "assistant",
        "content": raw.get("content") or "",
        "tool_calls": raw.get("tool_calls") or [],
    }


def run_tool_call(
    tools: dict[str, Tool], call: dict[str, Any], round_index: int = 1
) -> dict[str, Any]:
    """Run one model-requested tool call; never raises."""
    fn = call.get("function") or {}
    name = fn.get("name", "")
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except json.JSONDecodeError:
        args = {}
    tool = tools.get(name)
    try:
        output = tool.run(args) if tool else f"Error: unknown tool {name!r}"
    except Exception as err:  # a broken tool must not break the chat
        logger.warning(f"tool {name} failed: {err}")
        output = f"Error: {type(err).__name__}: {err}"
    output = output[:MAX_TOOL_OUTPUT]
    return {
        "name": name,
        "input": args,
        "output": output[:600],
        "round": round_index,
        "full_output": output,
    }


def run_agent(
    gateway: Gateway,
    question: str,
    history: list[Message],
    tools: list[Tool],
    *,
    max_rounds: int = 4,
    context: str | None = None,
) -> Iterator[dict[str, Any] | AgentResult]:
    """Yield a {"type": "tool", ...} event per tool call, then the AgentResult."""
    by_name = {t.name: t for t in tools}
    system = AGENT_SYSTEM + (
        f"\n\nRelevant excerpts from the user's documents:\n{context}" if context else ""
    )
    messages: list[Message] = [
        {"role": "system", "content": system},
        *history,
        {"role": "user", "content": question},
    ]
    result = AgentResult(notes="")

    for round_index in range(max_rounds):
        completion = gateway.complete(
            "agent",
            messages,
            max_tokens=1200,
            temperature=0.2,
            tools=[t.spec() for t in tools] if tools else None,
        )
        result.completions.append(completion)
        if not completion.tool_calls or not completion.tools_used:
            result.notes = completion.text
            break
        messages.append(_assistant_message(completion))
        for call in completion.tool_calls:
            record = run_tool_call(by_name, call, round_index + 1)
            full = record.pop("full_output")
            result.tool_calls.append(record)
            yield {"type": "tool", **record}
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": full})
    else:
        # Out of rounds: ask for notes from what was gathered.
        messages.append(
            {"role": "user", "content": "Summarise your research notes now, without more tools."}
        )
        completion = gateway.complete("agent", messages, max_tokens=900, temperature=0.2)
        result.completions.append(completion)
        result.notes = completion.text
    yield result

"""
assistant/pipeline.py
=====================
One chat turn, as a stream of events for the console:

  meta     conversation and message ids
  probe    the probe's samples, agreement and request type
  gate     which gate decided, STOP or ESCALATE, and every gate's verdict
  tool     each tool the agent called (ESCALATE only)
  delta    a piece of the streamed answer
  done     the saved answer with its token, model and timing breakdown
  error    what went wrong; needs_key = ask the user for another API key
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from assistant.agent import AgentResult, run_agent, run_tool_call
from assistant.llm import Message, ProvidersExhausted
from assistant.probe import run_probe
from gate.feature_extractor import extract_features
from shared.schemas import Task

if TYPE_CHECKING:
    from assistant.service import AssistantService

logger = logging.getLogger(__name__)

ANSWER_SYSTEM = (
    "You are GateOrchestra, a helpful, accurate assistant. Answer in GitHub-flavoured Markdown. "
    "Put code in fenced blocks with a language tag. Be complete but concise: start directly with "
    "the answer (no heading such as 'Answer'), "
    "then explain only as much as helps. If research notes are provided, rely on them, keep any "
    "numbers they computed, and cite their sources as links. If you are unsure, say so. "
    "Write math in LaTeX as \\( ... \\) inline or \\[ ... \\] on its own line; never wrap math in "
    "single $ signs, which mean money."
)


class ChatError(Exception):
    def __init__(self, code: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.code = code
        self.extra = extra


def _history_for_model(rows: list[dict[str, Any]]) -> list[Message]:
    return [
        {"role": r["role"], "content": r["content"]}
        for r in rows
        if r["role"] in ("user", "assistant")
    ]


class ChatPipeline:
    def __init__(self, service: AssistantService) -> None:
        self.service = service

    def run(
        self,
        message: str,
        conversation_id: str | None = None,
        *,
        gate_mode: str | None = None,
        document_ids: list[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        svc, settings = self.service, self.service.settings
        text = (message or "").strip()
        if not text:
            yield {"type": "error", "code": "empty", "message": "Type a message first."}
            return
        if len(text) > settings.max_input_chars:
            yield {
                "type": "error",
                "code": "too_long",
                "message": f"Messages are limited to {settings.max_input_chars} characters.",
            }
            return

        store = svc.store
        if conversation_id and not store.conversation_exists(conversation_id):
            conversation_id = None
        cid = conversation_id or store.create_conversation(text[:60])
        history = _history_for_model(store.messages(cid, limit=settings.history_messages))
        user_mid = store.add_message(cid, "user", text)
        started = time.perf_counter()
        self.partial = ""
        try:
            yield {"type": "meta", "conversation_id": cid, "user_message_id": user_mid}
            yield from self._answer(cid, user_mid, text, history, gate_mode, document_ids, started)
        except GeneratorExit:
            # The client stopped or disconnected: keep what was written, marked stopped.
            if self.partial.strip():
                store.add_message(
                    cid,
                    "assistant",
                    self.partial,
                    {"stopped": True, "latency_ms": round((time.perf_counter() - started) * 1000)},
                )
            else:
                store.delete_message(user_mid)
            raise
        except ProvidersExhausted as err:
            store.delete_message(user_mid)
            store.log_event("error", {"code": "needs_key", "attempts": err.attempts})
            yield {
                "type": "error",
                "code": "needs_key",
                "needs_key": err.needs_key,
                "message": str(err),
                "attempts": [
                    {"provider": a["provider"], "reason": a["reason"]} for a in err.attempts
                ],
            }
        except ChatError as err:
            store.delete_message(user_mid)
            yield {"type": "error", "code": err.code, "message": str(err), **err.extra}
        except Exception as err:
            logger.exception("chat turn failed")
            store.delete_message(user_mid)
            store.log_event(
                "error", {"code": "internal", "detail": f"{type(err).__name__}: {err}"[:300]}
            )
            yield {
                "type": "error",
                "code": "internal",
                "message": f"Something went wrong: {type(err).__name__}.",
            }

    def _answer(
        self,
        cid: str,
        user_mid: str,
        text: str,
        history: list[Message],
        gate_mode: str | None,
        document_ids: list[str] | None,
        started: float,
    ) -> Iterator[dict[str, Any]]:
        svc, settings = self.service, self.service.settings
        tokens: dict[str, int] = {"probe": 0, "agent": 0, "answer": 0}
        models: dict[str, str] = {}

        context = svc.retrieve_context(text, document_ids) if document_ids is not None else None

        # 1. Probe
        probe = run_probe(
            svc.gateway,
            text,
            history,
            samples=settings.probe_samples,
            max_tokens=settings.probe_max_tokens,
            context=context,
            task_id=user_mid,
        )
        tokens["probe"] = probe.tokens
        models["probe"] = f"{probe.completions[0].provider}:{probe.completions[0].model}"
        yield {
            "type": "probe",
            "task_type": probe.task_type,
            "consistency": probe.result.consistency_score,
            "samples": list(probe.result.raw_outputs),
            "answer": probe.result.answer,
            "tokens": probe.tokens,
            "latency_ms": round(probe.result.latency_ms),
        }

        # 2. Gate
        features = extract_features(
            Task(task_id=user_mid, question=text, context=context), probe.result
        )
        verdicts = svc.gates.decide_all(features)
        selection_scores: list[dict[str, Any]] = []
        if probe.open_ended:
            gate, decision, reason = (
                "triage",
                "STOP",
                f"{probe.task_type} request: nothing to verify",
            )
        else:
            gate, reason = svc.choose_gate(gate_mode)
            decision = verdicts[gate].decision
            if svc.last_selection is not None and svc.last_selection.gate == gate:
                selection_scores = [sc.as_dict() for sc in svc.last_selection.scores]
        yield {
            "type": "gate",
            "gate": gate,
            "decision": decision,
            "reason": reason,
            "verdicts": {g: v.decision for g, v in verdicts.items()},
            "selection": selection_scores,
            "features": features.model_dump(mode="json"),
        }

        # 3. Agent (ESCALATE)
        notes = ""
        tool_calls: list[dict[str, Any]] = []
        if decision == "ESCALATE":
            for item in run_agent(
                svc.gateway,
                text,
                history,
                svc.tools(),
                max_rounds=settings.agent_max_rounds,
                context=context,
            ):
                if isinstance(item, AgentResult):
                    notes, tool_calls = item.notes, item.tool_calls
                    tokens["agent"] = item.tokens
                    if item.completions:
                        models["agent"] = (
                            f"{item.completions[0].provider}:{item.completions[0].model}"
                        )
                else:
                    yield item

        # 4. Answer (streamed)
        role = svc.answer_role(decision, probe)
        system = ANSWER_SYSTEM
        if context:
            system += f"\n\nExcerpts from the user's documents (cite the file name):\n{context}"
        if notes:
            system += f"\n\nResearch notes:\n{notes}"
        messages: list[Message] = [
            {"role": "system", "content": system},
            *history,
            {"role": "user", "content": text},
        ]
        yield {"type": "answer_start", "model_role": role}
        answer = ""
        answer_tools = {t.name: t for t in svc.answer_tools()}
        for _ in range(3):  # at most two tool rounds, then a final answer
            calls: list[dict[str, Any]] = []
            for event in svc.gateway.stream(
                role,
                messages,
                max_tokens=settings.answer_max_tokens,
                tools=[t.spec() for t in answer_tools.values()] or None,
            ):
                if event.type == "delta":
                    answer += event.text
                    self.partial = answer
                    yield {"type": "delta", "text": event.text}
                else:
                    tokens["answer"] += event.prompt_tokens + event.completion_tokens
                    models["answer"] = f"{event.provider}:{event.model}"
                    calls = event.tool_calls
            if not calls:
                break
            messages.append({"role": "assistant", "content": answer or "", "tool_calls": calls})
            for call in calls:
                record = run_tool_call(answer_tools, call)
                tool_calls.append(record)
                yield {"type": "tool", **record}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": record["full_output"],
                    }
                )
                record.pop("full_output", None)
            answer_tools = {}  # final round answers without tools
        if not answer.strip():
            raise ChatError("empty_answer", "The model returned an empty answer. Try again.")

        self.partial = ""  # saved below as a complete answer
        meta = {
            "gate": gate,
            "decision": decision,
            "gate_reason": reason,
            "task_type": probe.task_type,
            "consistency": probe.result.consistency_score,
            "tokens": tokens,
            "total_tokens": sum(tokens.values()),
            "models": models,
            "tools": tool_calls,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "verdicts": {g: v.decision for g, v in verdicts.items()},
            "cached": False,
        }
        mid = svc.store.add_message(cid, "assistant", answer, meta)
        svc.store.log_event("chat", {"message_id": mid, **meta})
        yield {"type": "done", "message_id": mid, "conversation_id": cid, "meta": meta}

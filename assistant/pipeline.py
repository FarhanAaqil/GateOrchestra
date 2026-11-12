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
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from assistant.agent import AgentResult, run_agent, run_tool_call
from assistant.llm import Message, ProvidersExhausted
from assistant.probe import run_probe
from assistant.security import guard_threshold
from assistant.smalltalk import is_small_talk
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
    "numbers they computed, and cite their sources inline as Markdown links built from the URLs in "
    "the notes, e.g. [Wikipedia: Lusail Stadium](https://en.wikipedia.org/wiki/Lusail_Stadium); "
    "never write placeholder markers such as 【source】. If you are unsure, say so. "
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
        replace_last: bool = False,
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
        # Regenerate or edit: the model must not see the exchange being replaced, but it is only
        # deleted once the new answer is saved, so a failed attempt keeps the old one.
        replaced = self._last_exchange_ids(cid) if replace_last and conversation_id else []
        kept = [r for r in store.messages(cid) if r["id"] not in replaced]
        history = _history_for_model(kept[-settings.history_messages :])
        attached = [d["name"] for d in store.documents() if d["id"] in (document_ids or [])]
        user_mid = store.add_message(
            cid, "user", text, {"documents": attached} if attached else None
        )
        if conversation_id and not is_small_talk(text):
            asked = [r["content"] for r in kept if r["role"] == "user"]
            if asked and all(is_small_talk(q) for q in asked):
                # A chat that opened with "hi" is named after its first real question.
                store.rename_conversation(cid, text[:60])
        started = time.perf_counter()
        self.partial = ""
        try:
            yield {"type": "meta", "conversation_id": cid, "user_message_id": user_mid}
            yield from self._answer(
                cid, user_mid, text, history, gate_mode, document_ids, started, replace_last
            )
            self._discard(replaced)
        except GeneratorExit:
            # The client stopped or disconnected: keep what was written, marked stopped.
            if self.partial.strip():
                store.add_message(
                    cid,
                    "assistant",
                    self.partial,
                    {"stopped": True, "latency_ms": round((time.perf_counter() - started) * 1000)},
                )
                self._discard(replaced)
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

    def _last_exchange_ids(self, cid: str) -> list[str]:
        """Ids of the last answer and the question it answered (what regenerate replaces)."""
        rows = self.service.store.messages(cid)
        ids: list[str] = []
        if rows and rows[-1]["role"] == "assistant":
            ids.append(rows.pop()["id"])
        if rows and rows[-1]["role"] == "user":
            ids.append(rows.pop()["id"])
        return ids

    def _discard(self, message_ids: list[str]) -> None:
        for message_id in message_ids:
            self.service.store.delete_message(message_id)

    def _screen(self, text: str) -> tuple[float | None, int]:
        """Prompt-injection score and how long the check took, in milliseconds."""
        started = time.perf_counter()
        score = self.service.guard_score(text)
        return score, round((time.perf_counter() - started) * 1000)

    def _write_answer(
        self,
        role: str,
        system: str,
        history: list[Message],
        text: str,
        tokens: dict[str, int],
        models: dict[str, str],
        tool_calls: list[dict[str, Any]],
        started: float,
        timings: dict[str, int],
    ) -> Iterator[dict[str, Any]]:
        """Stream the final answer (with at most two tool rounds); returns the full text."""
        svc, settings = self.service, self.service.settings
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
                    if "first_text_ms" not in timings and event.text.strip():
                        timings["first_text_ms"] = round((time.perf_counter() - started) * 1000)
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
        return answer

    def _save(self, cid: str, answer: str, meta: dict[str, Any], started: float) -> dict[str, Any]:
        """Store the finished answer and build the `done` event."""
        meta["timings"]["total_ms"] = round((time.perf_counter() - started) * 1000)
        meta["latency_ms"] = meta["timings"]["total_ms"]
        mid = self.service.store.add_message(cid, "assistant", answer, meta)
        self.service.store.log_event("chat", {"message_id": mid, **meta})
        return {"type": "done", "message_id": mid, "conversation_id": cid, "meta": meta}

    def _small_talk(
        self, cid: str, text: str, history: list[Message], started: float
    ) -> Iterator[dict[str, Any]]:
        """Greetings and thanks: no screen, no probe, no gate, just a short reply."""
        tokens: dict[str, int] = {"probe": 0, "agent": 0, "answer": 0}
        models: dict[str, str] = {}
        tool_calls: list[dict[str, Any]] = []
        timings: dict[str, int] = {"safety_ms": 0, "probe_ms": 0, "gate_ms": 0}
        reason = "small talk: nothing to verify"
        yield {
            "type": "gate",
            "gate": "triage",
            "decision": "STOP",
            "reason": reason,
            "verdicts": {},
            "selection": [],
        }
        answer = yield from self._write_answer(
            "fast", ANSWER_SYSTEM, history, text, tokens, models, tool_calls, started, timings
        )
        self.partial = ""
        meta = {
            "gate": "triage",
            "decision": "STOP",
            "gate_reason": reason,
            "task_type": "conversation",
            "consistency": None,
            "tokens": tokens,
            "total_tokens": sum(tokens.values()),
            "models": models,
            "tools": tool_calls,
            "verdicts": {},
            "cached": False,
            "timings": timings,
        }
        yield self._save(cid, answer, meta, started)

    def _answer(
        self,
        cid: str,
        user_mid: str,
        text: str,
        history: list[Message],
        gate_mode: str | None,
        document_ids: list[str] | None,
        started: float,
        replace_last: bool = False,
    ) -> Iterator[dict[str, Any]]:
        svc, settings = self.service, self.service.settings
        tokens: dict[str, int] = {"probe": 0, "agent": 0, "answer": 0}
        models: dict[str, str] = {}
        timings: dict[str, int] = {}

        # 0. Greetings and thanks take the fast path; a repeated question may be cached.
        if document_ids is None and is_small_talk(text):
            yield from self._small_talk(cid, text, history, started)
            return
        cacheable = (
            not history
            and document_ids is None
            and (gate_mode or "auto") == "auto"
            and settings.cache_ttl > 0
            and not replace_last  # a regeneration must produce a new answer
        )
        cache_key = svc.cache_key(text) if cacheable else ""
        cached = svc.store.cache_get(cache_key, settings.cache_ttl) if cacheable else None
        if cached:
            yield from self._replay(cid, cached, started)
            return

        context = svc.retrieve_context(text, document_ids) if document_ids is not None else None

        # 1. Probe, with the prompt-injection screen running alongside it.
        probe_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=1) as screen_pool:
            screen = screen_pool.submit(self._screen, text)
            probe = run_probe(
                svc.gateway,
                text,
                history,
                samples=settings.probe_samples,
                max_tokens=settings.probe_max_tokens,
                context=context,
                task_id=user_mid,
            )
            timings["probe_ms"] = round((time.perf_counter() - probe_started) * 1000)
            score, timings["safety_ms"] = screen.result()
        if score is not None and score >= guard_threshold():
            svc.store.log_event("blocked", {"score": score})
            raise ChatError(
                "blocked",
                "This message looks like an attempt to override the assistant's "
                "instructions, so it was not answered.",
                score=round(score, 3),
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
        gate_started = time.perf_counter()
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
        timings["gate_ms"] = round((time.perf_counter() - gate_started) * 1000)
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
                svc.tools(document_ids),
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
        answer = yield from self._write_answer(
            role, system, history, text, tokens, models, tool_calls, started, timings
        )

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
            "verdicts": {g: v.decision for g, v in verdicts.items()},
            "cached": False,
            "timings": timings,
        }
        done = self._save(cid, answer, meta, started)
        if cacheable and not tool_calls_need_freshness(tool_calls):
            svc.store.cache_put(cache_key, text, answer, meta)
        yield done

    def _replay(self, cid: str, cached: dict[str, Any], started: float) -> Iterator[dict[str, Any]]:
        """Stream a cached answer: no model calls, zero tokens."""
        original = cached["meta"]
        meta = {
            **original,
            "cached": True,
            "tokens": {"probe": 0, "agent": 0, "answer": 0},
            "total_tokens": 0,
            "tools": original.get("tools", []),
            "latency_ms": round((time.perf_counter() - started) * 1000),
        }
        yield {
            "type": "gate",
            "gate": original.get("gate", "triage"),
            "decision": original.get("decision", "STOP"),
            "reason": "answered from the cache",
            "verdicts": original.get("verdicts", {}),
            "selection": [],
        }
        yield {"type": "answer_start", "model_role": "cache"}
        answer = cached["answer"]
        for start in range(0, len(answer), 120):
            yield {"type": "delta", "text": answer[start : start + 120]}
        mid = self.service.store.add_message(cid, "assistant", answer, meta)
        self.service.store.log_event("chat", {"message_id": mid, **meta})
        yield {"type": "done", "message_id": mid, "conversation_id": cid, "meta": meta}


def tool_calls_need_freshness(tool_calls: list[dict[str, Any]]) -> bool:
    """Answers built on live lookups or code runs are not cached."""
    return any(c.get("name") in {"search_wikipedia", "run_python"} for c in tool_calls)

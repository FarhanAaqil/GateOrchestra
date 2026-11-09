"""
assistant/service.py
====================
The assistant's shared state: settings, key pool, provider gateway, storage
and gates. The API holds one instance (get_service); tests swap in their own
(set_service).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from typing import Any

from dotenv import dotenv_values

from assistant import documents as docs
from assistant.agent import Tool, calculator_tool
from assistant.config import AssistantSettings, get_settings
from assistant.gates import GATES, GateSet
from assistant.keys import KeyPool, mask, persist_key, valid_key_shape
from assistant.llm import Gateway, ProviderError
from assistant.probe import ChatProbe
from assistant.sandbox import run_python, sandbox_status
from assistant.security import parse_guard_score
from assistant.selector import GateSelector, Selection
from assistant.store import Store
from assistant.web import search_wikipedia

logger = logging.getLogger(__name__)


class KeyRejected(Exception):
    pass


class AssistantService:
    def __init__(
        self,
        settings: AssistantSettings | None = None,
        *,
        gateway: Gateway | None = None,
        store: Store | None = None,
        gates: GateSet | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.pool = gateway.pool if gateway else KeyPool.from_env(self._key_environment())
        self.gateway = gateway or Gateway(self.settings, self.pool)
        self.store = store or Store(self.settings.db_path)
        self.gates = gates or GateSet()
        self.slots = threading.BoundedSemaphore(self.settings.max_concurrent)
        self.selector = GateSelector(
            self.store, self.settings.gate_margin, self.settings.gate_explore
        )
        self.last_selection: Selection | None = None

    def _key_environment(self) -> dict[str, str]:
        """The process environment plus keys saved earlier from the console.

        Keys added through the popup live in ``settings.env_path``. The real
        environment wins, so a key set by the operator is never overridden.
        """
        merged = dict(os.environ)
        path = self.settings.env_path
        if path.exists():
            for name, value in (dotenv_values(path) or {}).items():
                if value and not merged.get(name):
                    merged[name] = value
        return merged

    # Routing decisions (extended by later phases)

    def choose_gate(self, mode: str | None) -> tuple[str, str]:
        """A named gate if one is requested; otherwise the selector's pick."""
        requested = (mode or self.settings.gate_mode or "auto").strip()
        if requested in GATES and requested in self.gates.available():
            return requested, "chosen by you"
        selection = self.selector.choose(self.gates.available())
        self.last_selection = selection
        return selection.gate, selection.reason

    def answer_role(self, decision: str, probe: ChatProbe) -> str:
        """Model cascade: a confident, simple direct answer goes to the fast model."""
        if (
            decision == "STOP"
            and probe.result.consistency_score >= 1.0
            and probe.task_type in ("factual", "conversation", "creative")
        ):
            return "fast"
        return "answer"

    def rate(self, message_id: str, rating: int) -> dict[str, Any]:
        """Record a thumbs up (1), down (-1) or clear (0) on an assistant answer."""
        message = self.store.message(message_id)
        if message is None or message["role"] != "assistant":
            raise LookupError("No such answer.")
        if rating == 0:
            self.store.clear_feedback(message_id)
        else:
            self.store.set_feedback(
                message_id, 1 if rating > 0 else -1, message["meta"].get("gate")
            )
        return {"message_id": message_id, "rating": rating}

    def gate_report(self) -> dict[str, Any]:
        """Every gate's expected cost and rated accuracy, and which one auto would pick now."""
        scores = self.selector.scores(self.gates.available())
        best = max((s.accuracy for s in scores), default=0.0)
        for s in scores:
            s.eligible = s.accuracy >= best - self.selector.margin
        eligible = [s for s in scores if s.eligible]
        if eligible:
            min(eligible, key=lambda s: (s.expected_tokens, s.gate)).chosen = True
        return {
            "margin": self.selector.margin,
            "explore": self.selector.explore,
            "gates": [s.as_dict() for s in scores],
        }

    def tools(self, document_ids: list[str] | None = None) -> list[Tool]:
        """Tools for the research agent (ESCALATE)."""
        tools = [calculator_tool(), wikipedia_tool()]
        if sandbox_status()[0]:
            tools.append(python_tool())
        if document_ids is not None and self.store.documents():
            tools.append(self.document_tool(document_ids))
        return tools

    def document_tool(self, document_ids: list[str]) -> Tool:
        def run(args: dict[str, Any]) -> str:
            hits = docs.top_chunks(
                str(args.get("query", "")), self.store.all_chunks(document_ids or None)
            )
            return docs.format_context(hits) or "Nothing in the documents matches that."

        return Tool(
            name="search_documents",
            description="Search the user's uploaded documents for passages about a query.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            run=run,
        )

    def answer_tools(self) -> list[Tool]:
        """Cheap, exact tools the answer model may call on any path."""
        return [calculator_tool()]

    def retrieve_context(self, question: str, document_ids: list[str] | None) -> str | None:
        """Best-matching passages from the chosen documents ([] = all documents)."""
        if document_ids is None:
            return None
        hits = docs.top_chunks(question, self.store.all_chunks(document_ids or None))
        return docs.format_context(hits) or None

    def guard_score(self, text: str) -> float | None:
        """Prompt-injection probability from the guard model, or None if it is unavailable."""
        if not self.settings.guard_enabled or "groq" not in self.gateway.provider_order():
            return None
        try:
            result = self.gateway.complete(
                "guard",
                [{"role": "user", "content": text[:4000]}],
                max_tokens=8,
                providers=["groq"],
            )
        except Exception as err:  # never let the guard take the chat down
            logger.warning(f"prompt guard unavailable: {type(err).__name__}")
            return None
        return parse_guard_score(result.text)

    @staticmethod
    def cache_key(text: str) -> str:
        normalized = " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def add_document(self, name: str, content_base64: str) -> dict[str, Any]:
        data = docs.decode_upload(content_base64)
        text = docs.extract_text(name, data)
        chunks = docs.chunk_text(text)
        did = self.store.add_document(name, chunks)
        return {"id": did, "name": name, "chars": len(text), "chunks": len(chunks)}

    # Keys

    def add_key(self, provider: str, key: str) -> dict[str, Any]:
        """Check a key with its provider, then keep it in memory and in .env."""
        provider = provider.strip().lower()
        key = key.strip()
        if provider not in ("groq", "gemini"):
            raise KeyRejected("Provider must be groq or gemini.")
        if not valid_key_shape(key):
            raise KeyRejected("That does not look like an API key.")
        try:
            models = self.gateway.validate_key(provider, key)
        except ProviderError as err:
            raise KeyRejected(
                "The provider rejected this key."
                if err.kind == "invalid_key"
                else f"Could not check the key: {err.kind}."
            ) from err
        if provider == "gemini" and models and self.settings.gemini_model not in models:
            preferred = [m for m in models if "flash" in m and "lite" not in m] or models
            object.__setattr__(self.settings, "gemini_model", sorted(preferred)[-1])
        if provider not in self.settings.providers:
            object.__setattr__(self.settings, "providers", [*self.settings.providers, provider])
        self.pool.add(provider, key)
        persist_key(self.settings.env_path, provider, key)
        return {"provider": provider, "key": mask(key), "models": len(models)}

    def status(self) -> dict[str, Any]:
        s = self.settings
        return {
            "providers": self.gateway.provider_order(),
            "keys": self.pool.status(),
            "models": {
                "probe": s.probe_model,
                "fast": s.fast_model,
                "answer": s.answer_model,
                "agent": s.agent_model,
                "guard": s.guard_model,
                "gemini": s.gemini_model,
            },
            "gates": self.gates.available(),
            "gate_mode": s.gate_mode,
            "local_fallback": s.local_fallback,
        }


_service: AssistantService | None = None
_service_lock = threading.Lock()


def get_service() -> AssistantService:
    global _service
    with _service_lock:
        if _service is None:
            _service = AssistantService()
        return _service


def set_service(service: AssistantService | None) -> None:
    global _service
    with _service_lock:
        _service = service


def wikipedia_tool() -> Tool:
    return Tool(
        name="search_wikipedia",
        description="Search Wikipedia for facts, people, places, events or definitions. Returns article "
        "summaries with their URLs, which you should cite.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        run=lambda args: search_wikipedia(str(args.get("query", ""))),
    )


def python_tool() -> Tool:
    return Tool(
        name="run_python",
        description="Run a short Python 3 program in an isolated sandbox (no network, 15 s) and return "
        "its printed output. Use it to test code or compute results; print what you need to see.",
        parameters={
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
        run=lambda args: run_python(str(args.get("code", ""))),
    )

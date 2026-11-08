"""
assistant/service.py
====================
The assistant's shared state: settings, key pool, provider gateway, storage
and gates. The API holds one instance (get_service); tests swap in their own
(set_service).
"""

from __future__ import annotations

import threading
from typing import Any

from assistant.agent import Tool, calculator_tool
from assistant.config import AssistantSettings, get_settings
from assistant.gates import GATES, GateSet
from assistant.keys import KeyPool, mask, persist_key, valid_key_shape
from assistant.llm import Gateway, ProviderError
from assistant.probe import ChatProbe
from assistant.selector import GateSelector, Selection
from assistant.store import Store


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
        self.pool = gateway.pool if gateway else KeyPool.from_env()
        self.gateway = gateway or Gateway(self.settings, self.pool)
        self.store = store or Store(self.settings.db_path)
        self.gates = gates or GateSet()
        self.slots = threading.BoundedSemaphore(self.settings.max_concurrent)
        self.selector = GateSelector(
            self.store, self.settings.gate_margin, self.settings.gate_explore
        )
        self.last_selection: Selection | None = None

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

    def tools(self) -> list[Tool]:
        """Tools for the research agent (ESCALATE)."""
        return [calculator_tool()]

    def answer_tools(self) -> list[Tool]:
        """Cheap, exact tools the answer model may call on any path."""
        return [calculator_tool()]

    def retrieve_context(self, question: str, document_ids: list[str] | None) -> str | None:
        return None

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

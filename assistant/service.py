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

    # Routing decisions (extended by later phases)

    def choose_gate(self, mode: str | None) -> tuple[str, str]:
        requested = (mode or self.settings.gate_mode or "auto").strip()
        if requested in GATES and requested in self.gates.available():
            return requested, "chosen in settings"
        gate = self.gates.default_gate()
        return gate, "default gate"

    def answer_role(self, decision: str, probe: ChatProbe) -> str:
        return "answer"

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

"""
assistant/config.py
===================
Settings for the chat assistant, read from the environment (.env).

Secrets (API keys) are only ever read here and in assistant/keys.py; they are
never sent to the browser.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from shared.config import LOGS_DIR, MODEL_API_BASE, MODEL_NAME, ROOT_DIR


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _flag(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").lower() in {"1", "true", "yes", "on"}


def _list(name: str, default: str) -> list[str]:
    return [item.strip() for item in _env(name, default).split(",") if item.strip()]


@dataclass(frozen=True)
class AssistantSettings:
    # Which providers to try, in order. Ollama is the local fallback.
    providers: list[str] = field(
        default_factory=lambda: _list("ASSISTANT_PROVIDERS", "groq,gemini")
    )
    local_fallback: bool = field(default_factory=lambda: _flag("ASSISTANT_LOCAL_FALLBACK", False))

    # Groq models per role. Each Groq model has its own rate limit, so spreading
    # the roles across models multiplies the throughput of one key.
    probe_model: str = field(
        default_factory=lambda: _env("ASSISTANT_PROBE_MODEL", "openai/gpt-oss-20b")
    )
    fast_model: str = field(
        default_factory=lambda: _env("ASSISTANT_FAST_MODEL", "openai/gpt-oss-20b")
    )
    answer_model: str = field(
        default_factory=lambda: _env("ASSISTANT_ANSWER_MODEL", "openai/gpt-oss-120b")
    )
    agent_model: str = field(
        default_factory=lambda: _env("ASSISTANT_AGENT_MODEL", "qwen/qwen3.8-27b")
    )
    guard_model: str = field(
        default_factory=lambda: _env("ASSISTANT_GUARD_MODEL", "meta-llama/llama-prompt-guard-2-86m")
    )
    reasoning_effort: str = field(default_factory=lambda: _env("ASSISTANT_REASONING_EFFORT", "low"))

    groq_base: str = field(
        default_factory=lambda: _env("GROQ_API_BASE", "https://api.groq.com/openai/v1")
    )
    gemini_base: str = field(
        default_factory=lambda: _env(
            "GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta"
        )
    )
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL_NAME", "gemini-2.5-flash"))
    ollama_base: str = MODEL_API_BASE
    ollama_model: str = MODEL_NAME

    # Probe: two samples in parallel, a third only if they disagree.
    probe_samples: int = field(default_factory=lambda: int(_env("ASSISTANT_PROBE_SAMPLES", "3")))
    probe_max_tokens: int = field(
        default_factory=lambda: int(_env("ASSISTANT_PROBE_MAX_TOKENS", "400"))
    )
    answer_max_tokens: int = field(
        default_factory=lambda: int(_env("ASSISTANT_ANSWER_MAX_TOKENS", "2048"))
    )
    agent_max_rounds: int = field(
        default_factory=lambda: int(_env("ASSISTANT_AGENT_MAX_ROUNDS", "4"))
    )
    history_messages: int = field(
        default_factory=lambda: int(_env("ASSISTANT_HISTORY_MESSAGES", "12"))
    )
    max_input_chars: int = field(
        default_factory=lambda: int(_env("ASSISTANT_MAX_INPUT_CHARS", "8000"))
    )

    # Gate selection (Phase 2): cheapest gate within this accuracy margin of the best.
    gate_mode: str = field(default_factory=lambda: _env("ASSISTANT_GATE_MODE", "auto"))
    gate_margin: float = field(default_factory=lambda: float(_env("ASSISTANT_GATE_MARGIN", "0.02")))
    gate_explore: float = field(
        default_factory=lambda: float(_env("ASSISTANT_GATE_EXPLORE", "0.1"))
    )

    guard_enabled: bool = field(default_factory=lambda: _flag("ASSISTANT_GUARD", True))
    cache_ttl: float = field(default_factory=lambda: float(_env("ASSISTANT_CACHE_TTL", "86400")))

    timeout_seconds: float = field(default_factory=lambda: float(_env("ASSISTANT_TIMEOUT", "90")))
    stream_timeout_seconds: float = field(
        default_factory=lambda: float(_env("ASSISTANT_STREAM_TIMEOUT", "30"))
    )
    """How long a streamed answer may stay silent before it is treated as hung and retried."""
    quick_timeout_seconds: float = field(
        default_factory=lambda: float(_env("ASSISTANT_QUICK_TIMEOUT", "12"))
    )
    """Timeout for the quick check and the safety screen. They normally take 0.3-2 s, so a
    call that has not answered after this long is hung; failing over beats waiting 90 s."""
    max_concurrent: int = field(default_factory=lambda: int(_env("ASSISTANT_MAX_CONCURRENT", "8")))

    db_path: Path = field(
        default_factory=lambda: Path(_env("ASSISTANT_DB_PATH", str(LOGS_DIR / "assistant.db")))
    )
    # Where keys added from the console are saved. Locally that is .env; in Docker point it
    # at the data volume (ASSISTANT_ENV_PATH=/app/logs/keys.env), which the API can write.
    env_path: Path = field(
        default_factory=lambda: Path(_env("ASSISTANT_ENV_PATH", str(ROOT_DIR / ".env")))
    )


def get_settings() -> AssistantSettings:
    """Fresh settings (re-reads the environment, e.g. after a key is added)."""
    return AssistantSettings()

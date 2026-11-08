"""
assistant/keys.py
=================
API keys per provider: rotation, cooldown after rate limits, and adding keys.

Keys come only from the environment / .env:
  GROQ_API_KEY, GROQ_API_KEYS (comma-separated extras)
  GEMINI_API_KEY, GEMINI_API_KEYS

A key that hits a rate limit cools down for as long as the provider asks; a
key the provider rejects (401/403) is marked invalid. When every key of every
provider is unusable the chat asks the user for another key, which is checked
against the provider and then written to .env, never anywhere else. Keys are
never returned to callers in full: status reports show the last 4 characters.
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

PROVIDER_ENV = {
    "groq": ("GROQ_API_KEY", "GROQ_API_KEYS"),
    "gemini": ("GEMINI_API_KEY", "GEMINI_API_KEYS"),
}
_KEY_SHAPE = re.compile(r"^[A-Za-z0-9_\-]{20,200}$")


def mask(key: str) -> str:
    return f"…{key[-4:]}" if len(key) >= 8 else "…"


@dataclass
class KeyState:
    key: str
    cooldown_until: float = 0.0
    invalid: bool = False
    last_error: str = ""
    uses: int = 0
    model_cooldowns: dict[str, float] = field(default_factory=dict)

    def usable(self, now: float, model: str | None = None) -> bool:
        if self.invalid or now < self.cooldown_until:
            return False
        return model is None or now >= self.model_cooldowns.get(model, 0.0)


@dataclass
class KeyPool:
    """Thread-safe rotating pool of keys for each provider."""

    keys: dict[str, list[KeyState]] = field(default_factory=dict)
    _cursor: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> KeyPool:
        source = dict(os.environ) if env is None else env
        pool = cls()
        for provider, (single, many) in PROVIDER_ENV.items():
            found: list[str] = []
            for raw in [source.get(single, ""), *source.get(many, "").split(",")]:
                key = raw.strip()
                if key and key not in found and "your_" not in key:
                    found.append(key)
            pool.keys[provider] = [KeyState(k) for k in found]
        return pool

    def acquire(self, provider: str, model: str | None = None) -> str | None:
        """The next key usable for ``provider`` (and ``model``), round robin, or None."""
        with self._lock:
            states = self.keys.get(provider, [])
            now = time.time()
            for offset in range(len(states)):
                index = (self._cursor.get(provider, 0) + offset) % len(states)
                if states[index].usable(now, model):
                    self._cursor[provider] = index + 1
                    states[index].uses += 1
                    return states[index].key
            return None

    def report_rate_limited(
        self, provider: str, key: str, retry_after: float, model: str | None = None
    ) -> None:
        """Cool a key down; limits are per model, so only that model when known."""
        until = time.time() + max(1.0, retry_after)
        with self._lock:
            for state in self.keys.get(provider, []):
                if state.key == key:
                    if model:
                        state.model_cooldowns[model] = until
                    else:
                        state.cooldown_until = until
                    state.last_error = (
                        f"rate limited{f' on {model}' if model else ''} for {retry_after:.0f}s"
                    )

    def report_invalid(self, provider: str, key: str, reason: str) -> None:
        with self._lock:
            for state in self.keys.get(provider, []):
                if state.key == key:
                    state.invalid = True
                    state.last_error = reason

    def add(self, provider: str, key: str) -> bool:
        """Add a key in memory; False if it was already there."""
        with self._lock:
            states = self.keys.setdefault(provider, [])
            for state in states:
                if state.key == key:
                    state.invalid = False
                    state.cooldown_until = 0.0
                    return False
            states.append(KeyState(key))
            return True

    def why_unusable(self, provider: str) -> tuple[str, str]:
        """Why no key of ``provider`` can be used right now: no_key, invalid_key or rate_limited."""
        now = time.time()
        with self._lock:
            states = self.keys.get(provider, [])
            if not states:
                return "no_key", "no key configured"
            if all(s.invalid for s in states):
                return "invalid_key", "every key was rejected"
            ends = [s.cooldown_until for s in states if not s.invalid and s.cooldown_until > now]
            ends += [
                t for s in states if not s.invalid for t in s.model_cooldowns.values() if t > now
            ]
            wait = (min(ends) - now) if ends else 0.0
            return "rate_limited", f"rate limited; next model frees up in {wait:.0f}s"

    def status(self) -> dict[str, list[dict[str, object]]]:
        """Masked, safe-to-show state of every key."""
        now = time.time()
        with self._lock:
            return {
                provider: [
                    {
                        "key": mask(s.key),
                        "usable": s.usable(now),
                        "invalid": s.invalid,
                        "cooldown_seconds": max(
                            0, round(max([s.cooldown_until, *s.model_cooldowns.values()]) - now)
                        ),
                        "last_error": s.last_error,
                        "uses": s.uses,
                    }
                    for s in states
                ]
                for provider, states in self.keys.items()
            }


def valid_key_shape(key: str) -> bool:
    return bool(_KEY_SHAPE.match(key.strip()))


_ENV_LOCK = threading.Lock()


def persist_key(env_path: Path, provider: str, key: str) -> None:
    """Append ``key`` to the provider's extra-keys line in .env (atomic rewrite)."""
    single, many = PROVIDER_ENV[provider]
    with _ENV_LOCK:
        lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
        existing: list[str] = []
        many_index = None
        for i, line in enumerate(lines):
            name, _, value = line.partition("=")
            name = name.strip()
            if name == single and value.strip():
                existing.append(value.strip())
            elif name == many:
                many_index = i
                existing += [v.strip() for v in value.split(",") if v.strip()]
        if key in existing:
            return
        if not any(
            n.partition("=")[0].strip() == single and n.partition("=")[2].strip() for n in lines
        ):
            lines.append(f"{single}={key}")
        else:
            extras = [
                v.strip()
                for v in (
                    lines[many_index].partition("=")[2] if many_index is not None else ""
                ).split(",")
                if v.strip()
            ]
            new_line = f"{many}={','.join([*extras, key])}"
            if many_index is None:
                lines.append(new_line)
            else:
                lines[many_index] = new_line
        tmp = env_path.with_name(f".{env_path.name}.tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp.replace(env_path)
    # Keep this process in step with the file.
    if not os.environ.get(single):
        os.environ[single] = key
    else:
        os.environ[many] = ",".join(filter(None, [os.environ.get(many, ""), key]))

"""
assistant/llm.py
================
One gateway in front of every model provider.

    gateway.complete(role, messages, ...)        -> Completion
    gateway.stream(role, messages, ...)          -> iterator of StreamEvent

Each role (probe, fast, answer, agent, guard) maps to a model per provider.
Providers are tried in order (ASSISTANT_PROVIDERS, then Ollama if local
fallback is on) and, within a provider, every usable key in turn. A short
rate limit (a few seconds) is waited out; a long one cools the key down and the
next key is tried. When nothing is left, ProvidersExhausted says why, so the
console can ask the user for another key.

Only Groq is given tools (native OpenAI-style function calling). Gemini and
Ollama answer without tools; the caller is told via Completion.tools_used.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from agents.providers import rate_limit_delay
from assistant.config import AssistantSettings
from assistant.keys import KeyPool

logger = logging.getLogger(__name__)

ROLES = ("probe", "fast", "answer", "agent", "guard")
SHORT_WAIT_SECONDS = 8.0
QUICK_ROLES = frozenset({"probe", "guard"})
STREAM_RETRIES = 2  # extra attempts when a stream fails before any text

# Shortens the HTTP timeout for the current call (set by Gateway.complete for quick roles).
_timeout_override: ContextVar[float | None] = ContextVar("llm_timeout_override", default=None)
USER_AGENT = "GateOrchestra/1.0"
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

Message = dict[str, Any]


class ProviderError(Exception):
    """One provider call failed; ``kind`` says whether to wait, rotate or move on."""

    def __init__(self, kind: str, message: str, retry_after: float = 0.0) -> None:
        super().__init__(message)
        self.kind = kind  # rate_limited | invalid_key | unavailable | bad_request
        self.retry_after = retry_after


class ProvidersExhausted(Exception):
    """Every provider and key failed. ``needs_key`` means another key would help."""

    def __init__(self, attempts: list[dict[str, str]]) -> None:
        self.attempts = attempts
        self.needs_key = (
            any(a["reason"] in {"rate_limited", "invalid_key", "no_key"} for a in attempts)
            or not attempts
        )
        summary = "; ".join(f"{a['provider']}: {a['reason']}" for a in attempts) or "no provider"
        super().__init__(f"No model provider could answer ({summary}).")


@dataclass
class Completion:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    tools_used: bool = False
    raw_message: Message = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class StreamEvent:
    type: str  # "delta" | "end"
    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    provider: str = ""
    model: str = ""


# HTTP helpers


def _open(url: str, body: dict[str, Any] | None, headers: dict[str, str], timeout: float) -> Any:
    override = _timeout_override.get()
    if override is not None:
        timeout = min(timeout, override)
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method="POST" if data is not None else "GET",
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT, **headers},
    )
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", "replace")[:400]
        if err.code == 429:
            wait = rate_limit_delay(err.headers.get("Retry-After"), detail, 0, 2.0)
            hinted = re.search(r"try again in ([\dhms.]+)", detail)
            raise ProviderError(
                "rate_limited",
                detail,
                retry_after=wait if hinted or err.headers.get("Retry-After") else 60.0,
            ) from err
        if err.code in (401, 403) or "API_KEY_INVALID" in detail or "API key not valid" in detail:
            raise ProviderError("invalid_key", f"HTTP {err.code}: {detail}") from err
        if err.code in (400, 404, 413, 422):
            raise ProviderError("bad_request", f"HTTP {err.code}: {detail}") from err
        raise ProviderError("unavailable", f"HTTP {err.code}: {detail}") from err
    except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
        raise ProviderError("unavailable", f"{type(err).__name__}: {err}") from err


def _sse_lines(response: Any) -> Iterator[dict[str, Any]]:
    for raw in response:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            return
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


def tool_use_failed(err: Exception) -> bool:
    """Groq rejected a malformed or invented tool call the model generated."""
    text = str(err)
    return "tool_use_failed" in text or "tool call validation failed" in text


def json_generation_failed(err: Exception) -> bool:
    """Groq could not finish a JSON-mode reply (usually: the token budget ran out first)."""
    return "json_validate_failed" in str(err) or "Failed to generate JSON" in str(err)


def _chunk_error(chunk: dict[str, Any]) -> None:
    """Raise if a streamed chunk carries an error instead of tokens."""
    error = chunk.get("error")
    if not error:
        return
    message = json.dumps(error)[:400] if not isinstance(error, str) else error
    raise ProviderError("bad_request" if "tool" in message else "unavailable", message)


def _model_problem(err: ProviderError) -> bool:
    text = str(err).lower()
    return "model" in text and any(
        w in text for w in ("not found", "decommission", "does not exist", "not supported")
    )


def clean_text(text: str) -> str:
    """Drop <think> blocks some reasoning models put in their visible output."""
    return _THINK.sub("", text or "").strip()


# Provider clients


class GroqClient:
    name = "groq"

    def __init__(
        self,
        base: str,
        timeout: float,
        reasoning_effort: str,
        stream_timeout: float | None = None,
    ) -> None:
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.stream_timeout = min(timeout, stream_timeout) if stream_timeout else timeout
        self.reasoning_effort = reasoning_effort

    def _body(
        self,
        model: str,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
        tools: list[dict] | None,
        stream: bool,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if model.startswith("openai/gpt-oss"):
            body["reasoning_effort"] = self.reasoning_effort
        elif "qwen3" in model:
            body["reasoning_format"] = "hidden"
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        return body

    def complete(
        self,
        key: str,
        model: str,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
        tools: list[dict] | None,
        json_mode: bool = False,
    ) -> Completion:
        start = time.perf_counter()
        response = _open(
            f"{self.base}/chat/completions",
            self._body(model, messages, max_tokens, temperature, tools, False, json_mode),
            {"Authorization": f"Bearer {key}"},
            self.timeout,
        )
        data = json.load(response)
        message = data["choices"][0]["message"]
        usage = data.get("usage") or {}
        return Completion(
            text=clean_text(message.get("content") or ""),
            tool_calls=message.get("tool_calls") or [],
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            provider=self.name,
            model=model,
            latency_ms=(time.perf_counter() - start) * 1000,
            tools_used=bool(tools),
            raw_message=message,
        )

    def stream(
        self,
        key: str,
        model: str,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
        tools: list[dict] | None = None,
    ) -> Iterator[StreamEvent]:
        response = _open(
            f"{self.base}/chat/completions",
            self._body(model, messages, max_tokens, temperature, tools, True),
            {"Authorization": f"Bearer {key}"},
            self.stream_timeout,
        )

        def events() -> Iterator[StreamEvent]:
            usage: dict[str, Any] = {}
            calls: dict[int, dict[str, Any]] = {}
            with response:
                for chunk in _sse_lines(response):
                    _chunk_error(chunk)
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            yield StreamEvent("delta", text=delta["content"])
                        for part in delta.get("tool_calls") or []:
                            call = calls.setdefault(
                                int(part.get("index", 0)),
                                {
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                },
                            )
                            call["id"] = part.get("id") or call["id"]
                            fn = part.get("function") or {}
                            call["function"]["name"] += fn.get("name") or ""
                            call["function"]["arguments"] += fn.get("arguments") or ""
            yield StreamEvent(
                "end",
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                provider=self.name,
                model=model,
                tool_calls=[calls[i] for i in sorted(calls)],
            )

        return events()

    def list_models(self, key: str) -> list[str]:
        response = _open(f"{self.base}/models", None, {"Authorization": f"Bearer {key}"}, 20)
        return [m["id"] for m in json.load(response).get("data", [])]


def _gemini_payload(
    messages: list[Message], max_tokens: int, temperature: float, json_mode: bool = False
) -> dict[str, Any]:
    system = "\n\n".join(
        m["content"] for m in messages if m["role"] == "system" and m.get("content")
    )
    contents = []
    for m in messages:
        if m["role"] == "system" or not m.get("content"):
            continue
        role = "model" if m["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": str(m["content"])}]})
    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
    }
    if json_mode:
        payload["generationConfig"]["responseMimeType"] = "application/json"
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    return payload


class GeminiClient:
    name = "gemini"

    def __init__(self, base: str, timeout: float, stream_timeout: float | None = None) -> None:
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.stream_timeout = min(timeout, stream_timeout) if stream_timeout else timeout

    def complete(
        self,
        key: str,
        model: str,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
        tools: list[dict] | None,
        json_mode: bool = False,
    ) -> Completion:
        start = time.perf_counter()
        response = _open(
            f"{self.base}/models/{model}:generateContent",
            _gemini_payload(messages, max_tokens, temperature, json_mode),
            {"x-goog-api-key": key},
            self.timeout,
        )
        data = json.load(response)
        parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
        usage = data.get("usageMetadata") or {}
        return Completion(
            text=clean_text("".join(p.get("text", "") for p in parts)),
            prompt_tokens=int(usage.get("promptTokenCount") or 0),
            completion_tokens=int(usage.get("candidatesTokenCount") or 0),
            provider=self.name,
            model=model,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    def stream(
        self,
        key: str,
        model: str,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
        tools: list[dict] | None = None,
    ) -> Iterator[StreamEvent]:
        response = _open(
            f"{self.base}/models/{model}:streamGenerateContent?alt=sse",
            _gemini_payload(messages, max_tokens, temperature),
            {"x-goog-api-key": key},
            self.stream_timeout,
        )

        def events() -> Iterator[StreamEvent]:
            usage: dict[str, Any] = {}
            with response:
                for chunk in _sse_lines(response):
                    usage = chunk.get("usageMetadata") or usage
                    for cand in chunk.get("candidates") or []:
                        for part in (cand.get("content") or {}).get("parts") or []:
                            if part.get("text"):
                                yield StreamEvent("delta", text=part["text"])
            yield StreamEvent(
                "end",
                prompt_tokens=int(usage.get("promptTokenCount") or 0),
                completion_tokens=int(usage.get("candidatesTokenCount") or 0),
                provider=self.name,
                model=model,
            )

        return events()

    def list_models(self, key: str) -> list[str]:
        response = _open(f"{self.base}/models", None, {"x-goog-api-key": key}, 20)
        return [
            m["name"].split("/", 1)[-1]
            for m in json.load(response).get("models", [])
            if "generateContent" in (m.get("supportedGenerationMethods") or [])
        ]


class OllamaClient:
    name = "ollama"

    def __init__(self, base: str, timeout: float) -> None:
        self.base = base.rstrip("/")
        self.timeout = timeout

    def _body(
        self,
        model: str,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
        stream: bool,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        plain = [
            {"role": m["role"], "content": str(m.get("content") or "")}
            for m in messages
            if m["role"] in ("system", "user", "assistant")
        ]
        body: dict[str, Any] = {
            "model": model,
            "messages": plain,
            "stream": stream,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }
        if json_mode:
            body["format"] = "json"
        return body

    def complete(
        self,
        key: str,
        model: str,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
        tools: list[dict] | None,
        json_mode: bool = False,
    ) -> Completion:
        start = time.perf_counter()
        data = json.load(
            _open(
                f"{self.base}/api/chat",
                self._body(model, messages, max_tokens, temperature, False, json_mode),
                {},
                self.timeout,
            )
        )
        return Completion(
            text=clean_text((data.get("message") or {}).get("content", "")),
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
            provider=self.name,
            model=model,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    def stream(
        self,
        key: str,
        model: str,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
        tools: list[dict] | None = None,
    ) -> Iterator[StreamEvent]:
        response = _open(
            f"{self.base}/api/chat",
            self._body(model, messages, max_tokens, temperature, True),
            {},
            self.timeout,
        )

        def events() -> Iterator[StreamEvent]:
            last: dict[str, Any] = {}
            with response:
                for raw in response:
                    try:
                        chunk = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    text = (chunk.get("message") or {}).get("content")
                    if text:
                        yield StreamEvent("delta", text=text)
                    if chunk.get("done"):
                        last = chunk
            yield StreamEvent(
                "end",
                prompt_tokens=int(last.get("prompt_eval_count") or 0),
                completion_tokens=int(last.get("eval_count") or 0),
                provider=self.name,
                model=model,
            )

        return events()


class MockClient:
    """Offline, deterministic stand-in (ASSISTANT_PROVIDERS=mock) for tests and CI."""

    name = "mock"

    def _reply(self, messages: list[Message]) -> str:
        system = next((str(m.get("content")) for m in messages if m["role"] == "system"), "")
        question = next(
            (str(m.get("content")) for m in reversed(messages) if m["role"] == "user"), ""
        )
        if "Reply with JSON only" in system:
            kind = (
                "conversation"
                if question.lower().startswith(("hi", "hello", "thanks"))
                else "factual"
            )
            return json.dumps({"type": kind, "answer": "42"})
        if "research step" in system:
            return "Notes: the answer is 42 (offline mock)."
        return f"**Offline answer** to: {question[:80]}\n\n```python\nprint(42)\n```"

    def complete(
        self,
        key: str,
        model: str,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
        tools: list[dict] | None,
        json_mode: bool = False,
    ) -> Completion:
        text = self._reply(messages)
        return Completion(
            text=text,
            prompt_tokens=sum(len(str(m.get("content", ""))) // 4 for m in messages),
            completion_tokens=len(text) // 4,
            provider=self.name,
            model=model,
            latency_ms=1.0,
        )

    def stream(
        self,
        key: str,
        model: str,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
        tools: list[dict] | None = None,
    ) -> Iterator[StreamEvent]:
        text = self._reply(messages)

        def events() -> Iterator[StreamEvent]:
            for word in re.findall(r"\S+\s*", text):
                yield StreamEvent("delta", text=word)
            yield StreamEvent(
                "end",
                prompt_tokens=10,
                completion_tokens=len(text) // 4,
                provider=self.name,
                model=model,
            )

        return events()

    def list_models(self, key: str) -> list[str]:
        return ["mock"]


# Gateway

UsageHook = Callable[[str, Completion | StreamEvent], None]


class Gateway:
    """Route each role to a provider and key, with rotation and fallback."""

    def __init__(
        self,
        settings: AssistantSettings,
        pool: KeyPool,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.pool = pool
        self.sleep = sleep
        self.clients: dict[str, Any] = {
            "groq": GroqClient(
                settings.groq_base,
                settings.timeout_seconds,
                settings.reasoning_effort,
                settings.stream_timeout_seconds,
            ),
            "gemini": GeminiClient(
                settings.gemini_base, settings.timeout_seconds, settings.stream_timeout_seconds
            ),
            "ollama": OllamaClient(settings.ollama_base, settings.timeout_seconds),
            "mock": MockClient(),
        }
        self.usage_hooks: list[UsageHook] = []

    def provider_order(self) -> list[str]:
        if "mock" in self.settings.providers:
            return ["mock"]
        order = [p for p in self.settings.providers if p in self.clients and p != "ollama"]
        if self.settings.local_fallback or "ollama" in self.settings.providers:
            order.append("ollama")
        return order

    def models_for(self, provider: str, role: str) -> list[str]:
        """The role's model first, then the other Groq models as capacity fallbacks."""
        primary = self.model_for(provider, role)
        if provider != "groq":
            return [primary]
        backups = [self.settings.answer_model, self.settings.agent_model, self.settings.fast_model]
        out = [primary]
        for model in backups:
            if model not in out and not (role == "probe" and model == self.settings.agent_model):
                out.append(model)
        return out

    def model_for(self, provider: str, role: str) -> str:
        if provider == "gemini":
            return self.settings.gemini_model
        if provider == "ollama":
            return self.settings.ollama_model
        if provider == "mock":
            return f"mock-{role}"
        return str(getattr(self.settings, f"{role}_model"))

    def _another_key(self, provider: str, model: str, tried: set[str]) -> bool:
        """True if a key that has not been tried yet is usable for this model right now."""
        now = time.time()
        return any(
            state.key not in tried and state.usable(now, model)
            for state in self.pool.keys.get(provider, [])
        )

    def _attempt(
        self,
        provider: str,
        role: str,
        call: Callable[[Any, str, str | None], Any],
        attempts: list[dict[str, str]],
    ) -> Any:
        """Run ``call`` over the provider's models and keys; None if the provider is out.

        Models are tried in order; for each, every key that is not cooling down
        on that model. A short rate limit is waited out, a long one cools that
        key down for that model only (limits are per model), and a bad key is
        retired. A model that is over capacity or missing moves on to the next.
        """
        client = self.clients[provider]
        keyless = provider in ("ollama", "mock")
        had_key = False
        for model in self.models_for(provider, role):
            tried: set[str] = set()
            model_failed = False
            while not model_failed:
                key = None if keyless else self.pool.acquire(provider, model)
                if not keyless and (key is None or key in tried):
                    break
                had_key = True
                if key:
                    tried.add(key)
                retried = False
                while True:
                    try:
                        return call(client, model, key)
                    except ProviderError as err:
                        detail = f"{model}: {str(err)[:200]}"
                        if (
                            err.kind == "rate_limited"
                            and err.retry_after <= SHORT_WAIT_SECONDS
                            and not retried
                            and not self._another_key(provider, model, tried)
                        ):
                            retried = True
                            self.sleep(err.retry_after)
                            continue
                        if err.kind == "rate_limited":
                            self.pool.report_rate_limited(
                                provider, key or "", err.retry_after, model
                            )
                            attempts.append(
                                {"provider": provider, "reason": "rate_limited", "detail": detail}
                            )
                        elif err.kind == "invalid_key":
                            self.pool.report_invalid(provider, key or "", str(err)[:120])
                            attempts.append(
                                {"provider": provider, "reason": "invalid_key", "detail": detail}
                            )
                        elif err.kind == "bad_request" and not _model_problem(err):
                            raise
                        else:
                            attempts.append(
                                {"provider": provider, "reason": "unavailable", "detail": detail}
                            )
                            model_failed = True
                        break
                if keyless:
                    break
        if not had_key:
            reason, detail = self.pool.why_unusable(provider)
            attempts.append({"provider": provider, "reason": reason, "detail": detail})
        return None

    def complete(
        self,
        role: str,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        tools: list[dict] | None = None,
        providers: list[str] | None = None,
        json_mode: bool = False,
    ) -> Completion:
        # Quick calls fail over fast instead of waiting out a hung connection.
        quick = self.settings.quick_timeout_seconds if role in QUICK_ROLES else None
        token = _timeout_override.set(quick)
        try:
            return self._complete(
                role, messages, max_tokens, temperature, tools, providers, json_mode
            )
        finally:
            _timeout_override.reset(token)

    def _complete(
        self,
        role: str,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
        tools: list[dict] | None,
        providers: list[str] | None,
        json_mode: bool,
    ) -> Completion:
        attempts: list[dict[str, str]] = []
        for provider in providers or self.provider_order():
            provider_tools = tools if provider == "groq" else None

            def call(
                client: Any, model: str, key: str | None, provider_tools=provider_tools
            ) -> Completion:
                return client.complete(
                    key or "", model, messages, max_tokens, temperature, provider_tools, json_mode
                )

            try:
                result = self._attempt(provider, role, call, attempts)
            except ProviderError as err:
                if not (provider_tools and tool_use_failed(err)):
                    raise
                logger.info(f"{provider}: malformed tool call, retrying without tools")

                def plain(client: Any, model: str, key: str | None) -> Completion:
                    return client.complete(
                        key or "", model, messages, max_tokens, temperature, None, json_mode
                    )

                result = self._attempt(provider, role, plain, attempts)
            if result is not None:
                for hook in self.usage_hooks:
                    hook(role, result)
                return result
        raise ProvidersExhausted(attempts)

    def _open_stream(
        self,
        role: str,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
        tools: list[dict] | None,
    ) -> Iterator[StreamEvent]:
        attempts: list[dict[str, str]] = []
        for provider in self.provider_order():
            provider_tools = tools if provider == "groq" else None

            def call(
                client: Any, model: str, key: str | None, provider_tools=provider_tools
            ) -> Iterator[StreamEvent]:
                return client.stream(
                    key or "", model, messages, max_tokens, temperature, provider_tools
                )

            try:
                events = self._attempt(provider, role, call, attempts)
            except ProviderError as err:
                if not (provider_tools and tool_use_failed(err)):
                    raise
                return self._open_stream(role, messages, max_tokens, temperature, None)
            if events is not None:
                return events
        raise ProvidersExhausted(attempts)

    def stream(
        self,
        role: str,
        messages: list[Message],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.4,
        tools: list[dict] | None = None,
    ) -> Iterator[StreamEvent]:
        """Open a stream with the first provider that accepts the request.

        Only Groq streams receive ``tools``; their assembled tool calls arrive
        on the final "end" event. If the provider rejects a malformed tool call
        before any text arrives, the stream restarts without tools.
        """
        events = self._open_stream(role, messages, max_tokens, temperature, tools)
        return self._with_hooks(
            role, self._resilient(role, messages, max_tokens, temperature, tools, events)
        )

    def _resilient(
        self,
        role: str,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
        tools: list[dict] | None,
        events: Iterator[StreamEvent],
    ) -> Iterator[StreamEvent]:
        started = False
        for retry in range(STREAM_RETRIES + 1):
            try:
                for event in events:
                    started = started or event.type == "delta"
                    yield event
                return
            except (TimeoutError, ConnectionError, OSError) as err:
                # A hung or dropped connection surfaces as a raw socket error.
                failure: ProviderError = ProviderError(
                    "unavailable", f"{type(err).__name__}: {err}"
                )
            except ProviderError as err:
                if tools and not started and tool_use_failed(err):
                    logger.info("malformed tool call mid-stream, restarting without tools")
                    yield from self._open_stream(role, messages, max_tokens, temperature, None)
                    return
                if (
                    started
                    or err.kind not in ("unavailable", "rate_limited")
                    or retry == STREAM_RETRIES
                ):
                    raise
                failure = err
            if started or retry == STREAM_RETRIES:
                raise failure  # text already reached the reader: it cannot be restarted
            logger.info(f"stream failed before any text ({failure.kind}); trying another key")
            events = self._open_stream(role, messages, max_tokens, temperature, tools)

    def _with_hooks(self, role: str, events: Iterator[StreamEvent]) -> Iterator[StreamEvent]:
        for event in events:
            if event.type == "end":
                for hook in self.usage_hooks:
                    hook(role, event)
            yield event

    def validate_key(self, provider: str, key: str) -> list[str]:
        """Models the key can use; raises ProviderError if the key is refused."""
        return self.clients[provider].list_models(key)

"""
agents/providers.py
===================
LLM Provider Abstraction Layer for GateOrchestra (Person 2).

Supports:
  1. Ollama (Default local inference: Qwen2.5-7B-Instruct at http://localhost:11434)
  2. Groq (Cloud inference: e.g. llama-3.3-70b-versatile via OpenAI-compatible API)

Preserves custom `llm_caller` injection for testing and seamless switching via environment variables.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable

from shared.config import (
    GROQ_API_BASE,
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
    LLM_PROVIDER,
    LLM_REQUEST_TIMEOUT_SECONDS,
    MODEL_API_BASE,
    MODEL_NAME,
    PROBE_TOKEN_BUDGET,
)

logger = logging.getLogger(__name__)

# Standard LLM Caller signature: (prompt, temperature, max_tokens) -> (response_text, tokens_used)
LLMCallerFn = Callable[[str, float, int], tuple[str, int]]


# 1. Ollama Caller (Local Default)


def call_ollama(
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = PROBE_TOKEN_BUDGET,
    model_name: str | None = None,
    api_base: str | None = None,
    timeout: float = LLM_REQUEST_TIMEOUT_SECONDS,
) -> tuple[str, int]:
    """Execute generation against a local Ollama server.

    Returns:
        tuple of (response_text, total_tokens_used).
    """
    model = model_name or MODEL_NAME
    base = api_base or MODEL_API_BASE
    url = f"{base.rstrip('/')}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
        "stream": False,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            response_text = resp_data.get("response", "")
            prompt_eval_count = int(resp_data.get("prompt_eval_count", 0))
            eval_count = int(resp_data.get("eval_count", 0))
            total_tokens = prompt_eval_count + eval_count

            if total_tokens <= 0:
                total_tokens = max(1, len(prompt.split()) + len(response_text.split()))

            return response_text, total_tokens

    except (urllib.error.URLError, TimeoutError, OSError) as e:
        logger.warning(
            f"[OllamaProvider] Call to {url} failed: {e}. "
            "Ensure Ollama is running or provide a custom llm_caller."
        )
        raise


# 2. Groq Caller (Cloud OpenAI-compatible API)

GROQ_MAX_RETRIES = int(os.getenv("GATE_GROQ_MAX_RETRIES", "10"))
MAX_RATE_LIMIT_WAIT_SECONDS = 90.0
TOOL_CHOICE_RETRIES = 3
_TRY_AGAIN_RE = re.compile(r"try again in\s+(?:(\d+)m)?\s*(\d+(?:\.\d+)?)\s*(ms|s)", re.IGNORECASE)


def rate_limit_delay(
    retry_after: str | None, message: str, attempt: int, base_delay: float
) -> float:
    """Seconds to wait after a 429, preferring what the server says.

    Groq sends a ``Retry-After`` header and a "Please try again in 420ms" /
    "in 1m2.5s" hint in the body. Fall back to linear backoff when neither is
    present. A small margin avoids retrying a hair too early.
    """
    wait: float | None = None
    if retry_after:
        try:
            wait = float(retry_after)
        except ValueError:
            wait = None
    match = _TRY_AGAIN_RE.search(message or "")
    if match:
        minutes, amount, unit = match.groups()
        hinted = float(amount) / (1000.0 if unit.lower() == "ms" else 1.0)
        hinted += 60.0 * float(minutes or 0)
        wait = max(wait or 0.0, hinted)
    if wait is None:
        wait = base_delay * (attempt + 1)
    return min(wait + 0.25, MAX_RATE_LIMIT_WAIT_SECONDS)


_groq_key_lock = threading.Lock()
_groq_key_cursor = 0
_groq_key_cooldowns: dict[str, float] = {}


def groq_key_pool() -> list[str]:
    """Groq keys to rotate over: GROQ_API_KEY plus the comma-separated GROQ_API_KEYS.

    Groq limits are per account, so extra keys only add quota when they belong to
    different accounts. Placeholder values and duplicates are skipped.
    """
    raw = [os.getenv("GROQ_API_KEY") or GROQ_API_KEY, *os.getenv("GROQ_API_KEYS", "").split(",")]
    keys: list[str] = []
    for item in raw:
        key = item.strip()
        if key and "your_" not in key and key not in keys:
            keys.append(key)
    return keys


def _next_groq_key(pool: list[str]) -> tuple[str, float]:
    """Round-robin over keys that are not cooling down; (key, seconds to wait if all are)."""
    global _groq_key_cursor
    with _groq_key_lock:
        now = time.time()
        for offset in range(len(pool)):
            index = (_groq_key_cursor + offset) % len(pool)
            if _groq_key_cooldowns.get(pool[index], 0.0) <= now:
                _groq_key_cursor = index + 1
                return pool[index], 0.0
        soonest = min(pool, key=lambda k: _groq_key_cooldowns.get(k, 0.0))
        return soonest, _groq_key_cooldowns.get(soonest, 0.0) - now


def _cool_groq_key(key: str, seconds: float) -> None:
    with _groq_key_lock:
        _groq_key_cooldowns[key] = time.time() + seconds


def call_groq(
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = PROBE_TOKEN_BUDGET,
    model_name: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float = LLM_REQUEST_TIMEOUT_SECONDS,
) -> tuple[str, int]:
    """Execute generation against Groq API via OpenAI-compatible endpoint.

    Returns:
        tuple of (response_text, total_tokens_used).
    """
    primary = os.getenv("GROQ_API_KEY") or GROQ_API_KEY
    # The configured key stands for the whole pool; any other explicit key is used alone.
    pool = [api_key] if api_key is not None and api_key != primary else groq_key_pool()
    if not pool or not pool[0]:
        raise ValueError(
            "GROQ_API_KEY is not set. Please set the GROQ_API_KEY environment variable "
            "or pass api_key to use the Groq provider."
        )

    model = model_name or GROQ_MODEL_NAME
    base = api_base or GROQ_API_BASE
    url = f"{base.rstrip('/')}/chat/completions"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    data = json.dumps(payload).encode("utf-8")
    max_retries = GROQ_MAX_RETRIES
    base_delay = 4.0

    for attempt in range(max_retries + 1):
        key, wait_for_key = _next_groq_key(pool)
        if wait_for_key > 0:
            time.sleep(min(wait_for_key, MAX_RATE_LIMIT_WAIT_SECONDS))
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
                "User-Agent": "GateOrchestra/0.1.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                choices = resp_data.get("choices", [])
                response_text = ""
                if choices and "message" in choices[0]:
                    response_text = choices[0]["message"].get("content", "")

                usage = resp_data.get("usage", {})
                total_tokens = int(usage.get("total_tokens", 0))

                if total_tokens <= 0:
                    prompt_toks = int(usage.get("prompt_tokens", 0))
                    comp_toks = int(usage.get("completion_tokens", 0))
                    total_tokens = prompt_toks + comp_toks

                if total_tokens <= 0:
                    total_tokens = max(1, len(prompt.split()) + len(response_text.split()))

                return response_text, total_tokens

        except urllib.error.HTTPError as e:
            raw_err = ""
            try:
                raw_err = e.read().decode("utf-8")
                error_body = json.loads(raw_err)
                err_msg = error_body.get("error", {}).get("message", raw_err)
            except Exception:
                err_msg = str(e)

            if e.code == 429 and attempt < max_retries:
                delay = rate_limit_delay(e.headers.get("Retry-After"), err_msg, attempt, base_delay)
                if len(pool) > 1:
                    # Another key may have quota left: cool this one down and move on.
                    _cool_groq_key(key, delay)
                    logger.warning(
                        f"[GroqProvider] A key was rate limited (429); rotating "
                        f"(attempt {attempt + 1}/{max_retries})."
                    )
                    continue
                logger.warning(
                    f"[GroqProvider] Rate limited (429). Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(delay)
                continue

            if e.code == 400 and "Tool choice is none" in err_msg and attempt < TOOL_CHOICE_RETRIES:
                # gpt-oss now and then answers with a tool call although none is offered.
                # Sampling again usually fixes it; never switch to another model, because
                # that would silently mix two models' answers into one benchmark run.
                logger.warning(
                    f"[GroqProvider] {model} made an unrequested tool call; sampling again "
                    f"(attempt {attempt + 1}/{TOOL_CHOICE_RETRIES})."
                )
                continue

            logger.error(f"[GroqProvider] Groq API returned HTTP {e.code}: {err_msg}")
            raise RuntimeError(
                f"Groq API Error ({e.code}): {err_msg}. "
                f"Please verify model '{model}' is available on your Groq account."
            ) from e

        except (urllib.error.URLError, TimeoutError, OSError) as e:
            logger.warning(f"[GroqProvider] Network call to {url} failed: {e}")
            raise

    raise RuntimeError(
        f"[GroqProvider] Call to {url} failed: Max retries ({max_retries}) exceeded."
    )


# 3. Unified Dispatcher & Factory


def get_llm_caller(
    provider: str | None = None,
    model_name: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    timeout: float = LLM_REQUEST_TIMEOUT_SECONDS,
) -> LLMCallerFn:
    """Factory returning a standard LLMCallerFn for the selected provider.

    Args:
        provider: 'ollama' | 'groq' (defaults to config.LLM_PROVIDER or 'ollama').
        model_name: Model override.
        api_base: Base URL override.
        api_key: API key override (for Groq).
        timeout: Network timeout in seconds.
    """
    prov = (provider or LLM_PROVIDER or "ollama").lower()

    if prov == "groq":

        def groq_caller(prompt: str, temperature: float, max_tokens: int) -> tuple[str, int]:
            return call_groq(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                model_name=model_name,
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
            )

        return groq_caller

    # Default to Ollama
    def ollama_caller(prompt: str, temperature: float, max_tokens: int) -> tuple[str, int]:
        return call_ollama(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model_name=model_name,
            api_base=api_base,
            timeout=timeout,
        )

    return ollama_caller


def default_llm_caller(
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = PROBE_TOKEN_BUDGET,
    model_name: str | None = None,
    api_base: str | None = None,
    provider: str | None = None,
    api_key: str | None = None,
    timeout: float = LLM_REQUEST_TIMEOUT_SECONDS,
) -> tuple[str, int]:
    """Default unified HTTP caller dispatching to either Ollama or Groq."""
    caller = get_llm_caller(
        provider=provider,
        model_name=model_name,
        api_base=api_base,
        api_key=api_key,
        timeout=timeout,
    )
    return caller(prompt, temperature, max_tokens)

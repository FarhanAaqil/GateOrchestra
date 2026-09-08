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
import urllib.error
import urllib.request
from collections.abc import Callable

from shared.config import (
    GROQ_API_BASE,
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
    LLM_PROVIDER,
    MODEL_API_BASE,
    MODEL_NAME,
    PROBE_TOKEN_BUDGET,
)

logger = logging.getLogger(__name__)

# Standard LLM Caller signature: (prompt, temperature, max_tokens) -> (response_text, tokens_used)
LLMCallerFn = Callable[[str, float, int], tuple[str, int]]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ollama Caller (Local Default)
# ─────────────────────────────────────────────────────────────────────────────


def call_ollama(
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = PROBE_TOKEN_BUDGET,
    model_name: str | None = None,
    api_base: str | None = None,
    timeout: float = 30.0,
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


# ─────────────────────────────────────────────────────────────────────────────
# 2. Groq Caller (Cloud OpenAI-compatible API)
# ─────────────────────────────────────────────────────────────────────────────


def call_groq(
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = PROBE_TOKEN_BUDGET,
    model_name: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float = 30.0,
) -> tuple[str, int]:
    """Execute generation against Groq API via OpenAI-compatible endpoint.

    Returns:
        tuple of (response_text, total_tokens_used).
    """
    key = api_key or GROQ_API_KEY
    if not key:
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
        try:
            raw_err = e.read().decode("utf-8")
            error_body = json.loads(raw_err)
            err_msg = error_body.get("error", {}).get("message", raw_err)
        except Exception:
            err_msg = str(e)
        logger.error(f"[GroqProvider] Groq API returned HTTP {e.code}: {err_msg}")
        raise RuntimeError(
            f"Groq API Error ({e.code}): {err_msg}. "
            f"Please verify model '{model}' is available on your Groq account (e.g., 'groq/compound', 'qwen/qwen3.6-27b')."
        ) from e

    except (urllib.error.URLError, TimeoutError, OSError) as e:
        logger.warning(f"[GroqProvider] Network call to {url} failed: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 3. Unified Dispatcher & Factory
# ─────────────────────────────────────────────────────────────────────────────


def get_llm_caller(
    provider: str | None = None,
    model_name: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
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
    timeout: float = 30.0,
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

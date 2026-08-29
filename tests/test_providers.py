"""
tests/test_providers.py
=======================
Unit tests for LLM provider abstraction (Ollama & Groq).
Tests mock network calls completely so no real API keys or running servers are needed.
"""

import io
import json
import urllib.request
from unittest.mock import MagicMock, patch

import pytest

from agents.providers import (
    call_groq,
    call_ollama,
    default_llm_caller,
    get_llm_caller,
)
from agents.probe_agent import ProbeAgent
from agents.orchestrator import MASOrchestrator
from shared.schemas import Task


# ─────────────────────────────────────────────────────────────────────────────
# 1. Test Ollama Provider
# ─────────────────────────────────────────────────────────────────────────────


class TestOllamaProvider:
    @patch("urllib.request.urlopen")
    def test_call_ollama_success(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "response": "Final Answer: Paris",
            "prompt_eval_count": 25,
            "eval_count": 15,
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        text, tokens = call_ollama(
            prompt="What is the capital of France?",
            temperature=0.7,
            max_tokens=200,
            model_name="Qwen2.5-7B-Instruct",
            api_base="http://localhost:11434",
        )

        assert text == "Final Answer: Paris"
        assert tokens == 40

        # Verify request format
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://localhost:11434/api/generate"
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["model"] == "Qwen2.5-7B-Instruct"
        assert payload["temperature"] == 0.7


# ─────────────────────────────────────────────────────────────────────────────
# 2. Test Groq Provider
# ─────────────────────────────────────────────────────────────────────────────


class TestGroqProvider:
    @patch("urllib.request.urlopen")
    def test_call_groq_success(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Step 1: 2+2=4.\nFinal Answer: 4",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 15,
                "total_tokens": 35,
            },
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        text, tokens = call_groq(
            prompt="What is 2+2?",
            temperature=0.5,
            max_tokens=100,
            model_name="llama-3.3-70b-versatile",
            api_key="gsk_mock_test_key_12345",
            api_base="https://api.groq.com/openai/v1",
        )

        assert "Final Answer: 4" in text
        assert tokens == 35

        # Verify auth headers and OpenAI payload
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "https://api.groq.com/openai/v1/chat/completions"
        assert req.headers["Authorization"] == "Bearer gsk_mock_test_key_12345"
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["model"] == "llama-3.3-70b-versatile"
        assert payload["messages"][0]["content"] == "What is 2+2?"

    def test_call_groq_missing_api_key_raises(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "")
        with pytest.raises(ValueError, match="GROQ_API_KEY is not set"):
            call_groq(prompt="Test", api_key="")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Test Provider Dispatcher & Factory
# ─────────────────────────────────────────────────────────────────────────────


class TestProviderDispatcher:
    @patch("agents.providers.call_groq")
    def test_get_llm_caller_groq_dispatch(self, mock_call_groq):
        mock_call_groq.return_value = ("Groq answer", 50)
        caller = get_llm_caller(provider="groq", api_key="test_key")
        res, toks = caller("Hello", 0.7, 100)
        assert res == "Groq answer"
        assert toks == 50
        assert mock_call_groq.called

    @patch("agents.providers.call_ollama")
    def test_get_llm_caller_ollama_dispatch(self, mock_call_ollama):
        mock_call_ollama.return_value = ("Ollama answer", 30)
        caller = get_llm_caller(provider="ollama")
        res, toks = caller("Hello", 0.7, 100)
        assert res == "Ollama answer"
        assert toks == 30
        assert mock_call_ollama.called

    @patch("agents.providers.call_groq")
    def test_default_llm_caller_via_env_var(self, mock_call_groq, monkeypatch):
        monkeypatch.setenv("GATE_LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test_env")
        mock_call_groq.return_value = ("Groq env response", 25)

        text, toks = default_llm_caller("Prompt", provider="groq", api_key="gsk_test_env")
        assert text == "Groq env response"
        assert toks == 25


# ─────────────────────────────────────────────────────────────────────────────
# 4. Test Agents Multi-Provider Integration
# ─────────────────────────────────────────────────────────────────────────────


class TestAgentMultiProviderIntegration:
    @patch("agents.providers.call_groq")
    def test_probe_agent_with_groq_provider(self, mock_call_groq):
        mock_call_groq.return_value = ("Thinking... Final Answer: 42", 20)
        task = Task(task_id="t_groq_01", question="What is 6 * 7?")

        agent = ProbeAgent(
            n_samples=3,
            provider="groq",
            api_key="mock_key",
        )
        res = agent.run(task)

        assert res.answer == "42"
        assert res.consistency_score == 1.0
        assert res.tokens_used == 60
        assert mock_call_groq.call_count == 3

    @patch("agents.providers.call_groq")
    def test_orchestrator_with_groq_provider(self, mock_call_groq):
        mock_call_groq.return_value = ("Final Answer: Euro", 30)
        task = Task(
            task_id="t_groq_orch",
            question="What is the currency of France?",
            depth_score=3,
        )

        orch = MASOrchestrator(
            provider="groq",
            api_key="mock_key",
        )
        ans, tokens = orch.run(task, token_budget=100)

        assert ans == "Euro"
        assert tokens <= 100

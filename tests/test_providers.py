"""
tests/test_providers.py
=======================
Unit tests for LLM provider abstraction (Ollama & Groq).
Tests mock network calls completely so no real API keys or running servers are needed.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.orchestrator import MASOrchestrator
from agents.probe_agent import ProbeAgent
from agents.providers import (
    call_groq,
    call_ollama,
    default_llm_caller,
    get_llm_caller,
)
from shared.schemas import Task

# 1. Test Ollama Provider


class TestOllamaProvider:
    @patch("urllib.request.urlopen")
    def test_call_ollama_success(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "response": "Final Answer: Paris",
                "prompt_eval_count": 25,
                "eval_count": 15,
            }
        ).encode("utf-8")
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


# 2. Test Groq Provider


class TestGroqProvider:
    @patch("urllib.request.urlopen")
    def test_call_groq_success(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
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
            }
        ).encode("utf-8")
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
        monkeypatch.setenv("GROQ_API_KEYS", "")
        monkeypatch.setattr("agents.providers.GROQ_API_KEY", "")
        with pytest.raises(ValueError, match="GROQ_API_KEY is not set"):
            call_groq(prompt="Test", api_key="")


# 3. Test Provider Dispatcher & Factory


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


# 4. Test Agents Multi-Provider Integration


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
        ans, tokens = orch.run(task, token_budget=600)

        assert ans == "Euro"
        assert tokens <= 600


class TestRateLimitBackoff:
    def test_uses_the_servers_retry_hint(self) -> None:
        from agents.providers import rate_limit_delay

        assert rate_limit_delay(None, "Please try again in 420ms.", 0, 4.0) == pytest.approx(0.67)
        assert rate_limit_delay(None, "Please try again in 1m2.5s.", 0, 4.0) == pytest.approx(62.75)
        assert rate_limit_delay("7", "", 0, 4.0) == pytest.approx(7.25)

    def test_falls_back_to_linear_backoff_and_caps_the_wait(self) -> None:
        from agents.providers import MAX_RATE_LIMIT_WAIT_SECONDS, rate_limit_delay

        assert rate_limit_delay(None, "busy", 2, 4.0) == pytest.approx(12.25)
        assert rate_limit_delay("600", "", 0, 4.0) == MAX_RATE_LIMIT_WAIT_SECONDS

    @patch("agents.providers.time.sleep")
    @patch("urllib.request.urlopen")
    def test_retries_a_429_then_succeeds(self, mock_urlopen, mock_sleep) -> None:
        import io
        import urllib.error

        from agents.providers import call_groq

        limited = urllib.error.HTTPError(
            "https://api.groq.com",
            429,
            "Too Many Requests",
            {"Retry-After": "2"},
            io.BytesIO(b'{"error": {"message": "Rate limit reached. Please try again in 1.5s."}}'),
        )
        ok = MagicMock()
        ok.read.return_value = json.dumps(
            {"choices": [{"message": {"content": "Final Answer: 4"}}], "usage": {"total_tokens": 9}}
        ).encode()
        ok.__enter__.return_value = ok
        mock_urlopen.side_effect = [limited, ok]

        assert call_groq("2+2?", api_key="k") == ("Final Answer: 4", 9)
        mock_sleep.assert_called_once_with(pytest.approx(2.25))


class TestGroqKeyRotation:
    @pytest.fixture(autouse=True)
    def fresh_state(self, monkeypatch):
        import agents.providers as providers

        providers._groq_key_cooldowns.clear()
        monkeypatch.setattr(providers, "_groq_key_cursor", 0)
        monkeypatch.setattr(providers, "GROQ_API_KEY", "")
        monkeypatch.setenv("GROQ_API_KEY", "key-a")
        monkeypatch.setenv("GROQ_API_KEYS", "key-b, key-c ,key-a,your_key_here")
        yield
        providers._groq_key_cooldowns.clear()

    @staticmethod
    def _ok_response():
        response = MagicMock()
        response.read.return_value = json.dumps(
            {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 7}}
        ).encode("utf-8")
        response.__enter__.return_value = response
        return response

    def test_pool_skips_duplicates_and_placeholders(self):
        from agents.providers import groq_key_pool

        assert groq_key_pool() == ["key-a", "key-b", "key-c"]

    @patch("urllib.request.urlopen")
    def test_calls_rotate_round_robin(self, mock_urlopen):
        mock_urlopen.return_value = self._ok_response()
        for _ in range(4):
            call_groq("hi", api_key="key-a")
        used = [c[0][0].headers["Authorization"] for c in mock_urlopen.call_args_list]
        assert used == ["Bearer key-a", "Bearer key-b", "Bearer key-c", "Bearer key-a"]

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_rate_limited_key_hands_over_to_the_next_without_sleeping(self, mock_urlopen, sleep):
        import urllib.error

        limited = urllib.error.HTTPError(
            "https://api.groq.com", 429, "Too Many Requests", {"Retry-After": "30"}, None
        )
        limited.read = lambda: b'{"error": {"message": "rate limit"}}'  # type: ignore[method-assign]
        mock_urlopen.side_effect = [limited, self._ok_response()]

        text, tokens = call_groq("hi")

        assert (text, tokens) == ("ok", 7)
        used = [c[0][0].headers["Authorization"] for c in mock_urlopen.call_args_list]
        assert used == ["Bearer key-a", "Bearer key-b"]
        sleep.assert_not_called()

    @patch("urllib.request.urlopen")
    def test_a_different_explicit_key_is_used_alone(self, mock_urlopen):
        mock_urlopen.return_value = self._ok_response()
        for _ in range(3):
            call_groq("hi", api_key="someone-elses-key")
        used = {c[0][0].headers["Authorization"] for c in mock_urlopen.call_args_list}
        assert used == {"Bearer someone-elses-key"}

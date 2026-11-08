"""Tests for the chat assistant: keys, provider gateway, probe, pipeline and API."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from assistant import llm
from assistant.config import AssistantSettings
from assistant.gates import GateSet
from assistant.keys import KeyPool, mask, persist_key, valid_key_shape
from assistant.llm import Completion, Gateway, ProviderError, ProvidersExhausted, StreamEvent
from assistant.pipeline import ChatPipeline
from assistant.probe import parse_sample, run_probe
from assistant.service import AssistantService, KeyRejected, set_service
from assistant.store import Store

KEY_A = "gsk_" + "a" * 40
KEY_B = "gsk_" + "b" * 40


def settings(tmp_path: Path, **kw) -> AssistantSettings:
    return AssistantSettings(db_path=tmp_path / "a.db", env_path=tmp_path / ".env", **kw)


# Keys


def test_pool_rotates_cools_down_and_masks() -> None:
    pool = KeyPool.from_env(
        {"GROQ_API_KEY": KEY_A, "GROQ_API_KEYS": f"{KEY_B}, {KEY_A}", "GEMINI_API_KEY": "your_key"}
    )
    assert [s.key for s in pool.keys["groq"]] == [KEY_A, KEY_B]
    assert pool.keys["gemini"] == []  # placeholder ignored
    assert pool.acquire("groq") == KEY_A
    assert pool.acquire("groq") == KEY_B
    pool.report_rate_limited("groq", KEY_A, 60)
    assert pool.acquire("groq") == KEY_B
    pool.report_invalid("groq", KEY_B, "401")
    assert pool.acquire("groq") is None
    status = pool.status()["groq"]
    assert status[0]["key"] == mask(KEY_A) == "…aaaa"
    assert KEY_A not in json.dumps(status)
    assert (
        pool.add("groq", KEY_B) is False and pool.acquire("groq") == KEY_B
    )  # re-adding revives it


def test_persist_key_appends_without_duplicates(tmp_path: Path, monkeypatch) -> None:
    for name in ("GROQ_API_KEY", "GROQ_API_KEYS", "GEMINI_API_KEY", "GEMINI_API_KEYS"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text("GATE_LLM_PROVIDER=ollama\nGROQ_API_KEY=" + KEY_A + "\n", encoding="utf-8")
    persist_key(env, "groq", KEY_B)
    persist_key(env, "groq", KEY_B)
    persist_key(env, "gemini", "AIza" + "c" * 35)
    text = env.read_text(encoding="utf-8")
    assert text.count(KEY_B) == 1
    assert "GATE_LLM_PROVIDER=ollama" in text
    assert f"GROQ_API_KEYS={KEY_B}" in text
    assert "GEMINI_API_KEY=AIza" in text
    assert (
        valid_key_shape(KEY_A)
        and not valid_key_shape("short")
        and not valid_key_shape("has space " * 5)
    )


# Gateway with fake providers


class FakeClient:
    def __init__(self, script: list) -> None:
        self.script = script
        self.calls: list[tuple[str, str]] = []

    def complete(self, key, model, messages, max_tokens, temperature, tools, json_mode=False):
        self.calls.append((key, model))
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return Completion(
            text=step, provider="groq", model=model, prompt_tokens=3, completion_tokens=2
        )

    def stream(self, key, model, messages, max_tokens, temperature, tools=None):
        self.calls.append((key, model))
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return iter(
            [StreamEvent("delta", text=step), StreamEvent("end", provider="groq", model=model)]
        )


def gateway(
    tmp_path: Path, script: list, keys: dict[str, str]
) -> tuple[Gateway, FakeClient, list[float]]:
    slept: list[float] = []
    gw = Gateway(
        settings(tmp_path, providers=["groq", "gemini"]), KeyPool.from_env(keys), sleep=slept.append
    )
    fake = FakeClient(script)
    gw.clients["groq"] = fake
    return gw, fake, slept


def test_short_rate_limit_is_waited_out(tmp_path: Path) -> None:
    gw, fake, slept = gateway(
        tmp_path, [ProviderError("rate_limited", "x", 2.0), "ok"], {"GROQ_API_KEY": KEY_A}
    )
    assert gw.complete("probe", []).text == "ok"
    assert slept == [2.0] and fake.calls == [(KEY_A, gw.settings.probe_model)] * 2


def test_long_rate_limit_rotates_to_next_key(tmp_path: Path) -> None:
    gw, fake, _ = gateway(
        tmp_path,
        [ProviderError("rate_limited", "x", 600), "ok"],
        {"GROQ_API_KEY": KEY_A, "GROQ_API_KEYS": KEY_B},
    )
    assert gw.complete("answer", []).text == "ok"
    assert [k for k, _ in fake.calls] == [KEY_A, KEY_B]
    assert gw.pool.status()["groq"][0]["cooldown_seconds"] > 500


def test_over_capacity_model_falls_back_to_another_model(tmp_path: Path) -> None:
    gw, fake, _ = gateway(
        tmp_path, [ProviderError("unavailable", "over capacity"), "ok"], {"GROQ_API_KEY": KEY_A}
    )
    result = gw.complete("agent", [])
    assert result.text == "ok"
    assert (
        fake.calls[0][1] == gw.settings.agent_model and fake.calls[1][1] != gw.settings.agent_model
    )


def test_everything_failing_asks_for_a_key(tmp_path: Path) -> None:
    gw, _, _ = gateway(tmp_path, [ProviderError("invalid_key", "401")], {"GROQ_API_KEY": KEY_A})
    with pytest.raises(ProvidersExhausted) as info:
        gw.complete("probe", [])
    reasons = {a["provider"]: a["reason"] for a in info.value.attempts}
    assert reasons == {"groq": "invalid_key", "gemini": "no_key"}
    assert info.value.needs_key


def test_bad_request_is_not_hidden(tmp_path: Path) -> None:
    gw, _, _ = gateway(
        tmp_path,
        [ProviderError("bad_request", "HTTP 400: messages invalid")],
        {"GROQ_API_KEY": KEY_A},
    )
    with pytest.raises(ProviderError):
        gw.complete("probe", [])


def test_stream_falls_through_providers(tmp_path: Path) -> None:
    gw, _, _ = gateway(
        tmp_path, [ProviderError("unavailable", "down")] * 3, {"GROQ_API_KEY": KEY_A}
    )
    with pytest.raises(ProvidersExhausted):
        list(gw.stream("answer", []))


# Wire formats


class FakeResponse(io.BytesIO):
    pass


def test_groq_stream_assembles_text_usage_and_tool_calls(monkeypatch) -> None:
    chunks = [
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "calc", "arguments": '{"ex'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": 'pression": "1+1"}'}}]
                    }
                }
            ]
        },
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 7}},
    ]
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    monkeypatch.setattr(llm, "_open", lambda *a, **k: FakeResponse(body.encode()))
    events = list(
        llm.GroqClient("https://x", 5, "low").stream("k", "m", [], 10, 0.2, tools=[{"x": 1}])
    )
    assert "".join(e.text for e in events if e.type == "delta") == "Hello"
    end = events[-1]
    assert (end.prompt_tokens, end.completion_tokens) == (5, 7)
    assert end.tool_calls == [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "calc", "arguments": '{"expression": "1+1"}'},
        }
    ]


def test_groq_body_sets_reasoning_json_and_tools() -> None:
    client = llm.GroqClient("https://x", 5, "low")
    body = client._body("openai/gpt-oss-20b", [], 10, 0.1, [{"t": 1}], True, json_mode=True)
    assert body["reasoning_effort"] == "low" and body["response_format"] == {"type": "json_object"}
    assert body["tool_choice"] == "auto" and body["stream_options"] == {"include_usage": True}
    assert (
        client._body("qwen/qwen3.8-27b", [], 10, 0.1, None, False)["reasoning_format"] == "hidden"
    )


def test_gemini_payload_maps_roles_and_system() -> None:
    payload = llm._gemini_payload(
        [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ],
        50,
        0.3,
        json_mode=True,
    )
    assert payload["systemInstruction"]["parts"][0]["text"] == "be brief"
    assert [c["role"] for c in payload["contents"]] == ["user", "model"]
    assert payload["generationConfig"]["responseMimeType"] == "application/json"


def test_http_errors_are_classified(monkeypatch) -> None:
    import urllib.error

    def raiser(code, body, headers=None):
        def _raise(*a, **k):
            raise urllib.error.HTTPError("u", code, "x", headers or {}, io.BytesIO(body.encode()))

        return _raise

    for code, body, kind in [
        (401, "bad key", "invalid_key"),
        (
            400,
            '{"error": {"status": "INVALID_ARGUMENT", "details": [{"reason": "API_KEY_INVALID"}]}}',
            "invalid_key",
        ),
        (503, "over capacity", "unavailable"),
        (400, "bad", "bad_request"),
        (429, "try again in 3s", "rate_limited"),
    ]:
        monkeypatch.setattr(llm.urllib.request, "urlopen", raiser(code, body))
        with pytest.raises(ProviderError) as info:
            llm._open("https://x", {}, {}, 5)
        assert info.value.kind == kind
        if kind == "rate_limited":
            assert 3 <= info.value.retry_after <= 4
    assert llm.clean_text("<think>hmm</think> Paris") == "Paris"


# Probe


def test_parse_sample_reads_json_and_tolerates_text() -> None:
    assert parse_sample('{"type": "math", "answer": "42"}') == ("math", "42")
    assert parse_sample('{"type": "weird", "answer": ""}') == ("factual", "Unknown")
    assert parse_sample("Type: code\nFinal Answer: use a loop") == ("code", "use a loop")


class ScriptedGateway:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls = 0

    def complete(self, role, messages, **kw):
        self.calls += 1
        return Completion(
            text=self.replies[self.calls - 1],
            provider="groq",
            model="m",
            prompt_tokens=10,
            completion_tokens=5,
        )


def test_probe_stops_after_two_agreeing_samples() -> None:
    gw = ScriptedGateway(
        ['{"type":"factual","answer":"Paris"}', '{"type":"factual","answer":"paris"}']
    )
    probe = run_probe(gw, "Capital of France?", [])
    assert gw.calls == 2 and probe.result.consistency_score == 1.0 and probe.tokens == 30


def test_probe_breaks_a_tie_with_a_third_sample() -> None:
    gw = ScriptedGateway(
        [
            '{"type":"factual","answer":"Messi"}',
            '{"type":"factual","answer":"Mbappe"}',
            '{"type":"factual","answer":"Mbappe"}',
        ]
    )
    probe = run_probe(gw, "Top scorer?", [])
    assert gw.calls == 3 and probe.result.answer == "Mbappe"
    assert probe.result.consistency_score == pytest.approx(2 / 3)


def test_probe_skips_the_tiebreak_for_small_talk() -> None:
    gw = ScriptedGateway(
        ['{"type":"conversation","answer":"hi!"}', '{"type":"conversation","answer":"hello"}']
    )
    probe = run_probe(gw, "hello", [])
    assert gw.calls == 2 and probe.open_ended


# Pipeline and API on the offline mock provider


@pytest.fixture()
def service(tmp_path: Path) -> AssistantService:
    svc = AssistantService(settings(tmp_path, providers=["mock"]), gates=GateSet())
    set_service(svc)
    yield svc
    set_service(None)


def test_pipeline_streams_and_stores_a_turn(service: AssistantService) -> None:
    events = list(ChatPipeline(service).run("What is the meaning of life?"))
    kinds = [e["type"] for e in events]
    assert kinds[:3] == ["meta", "probe", "gate"] and kinds[-1] == "done" and "delta" in kinds
    done = events[-1]
    answer = "".join(e["text"] for e in events if e["type"] == "delta")
    stored = service.store.messages(done["conversation_id"])
    assert [m["role"] for m in stored] == ["user", "assistant"]
    assert stored[1]["content"] == answer and stored[1]["meta"]["gate"] == done["meta"]["gate"]
    assert done["meta"]["total_tokens"] == sum(done["meta"]["tokens"].values()) > 0


def test_pipeline_triages_small_talk_and_forced_escalation(service: AssistantService) -> None:
    gate = next(e for e in ChatPipeline(service).run("hello there") if e["type"] == "gate")
    assert (gate["gate"], gate["decision"]) == ("triage", "STOP")
    events = list(ChatPipeline(service).run("Why is the sky blue?", gate_mode="always_escalate"))
    gate = next(e for e in events if e["type"] == "gate")
    assert (gate["gate"], gate["decision"]) == ("always_escalate", "ESCALATE")
    assert events[-1]["meta"]["tokens"]["agent"] > 0


def test_pipeline_rejects_empty_and_huge_messages(service: AssistantService) -> None:
    assert list(ChatPipeline(service).run("   "))[0]["code"] == "empty"
    assert list(ChatPipeline(service).run("x" * 9000))[0]["code"] == "too_long"


def test_failed_turn_removes_the_user_message_and_asks_for_a_key(
    service: AssistantService, monkeypatch
) -> None:
    def exhausted(*a, **k):
        raise ProvidersExhausted([{"provider": "groq", "reason": "rate_limited", "detail": ""}])

    monkeypatch.setattr(service.gateway, "complete", exhausted)
    events = list(ChatPipeline(service).run("anything"))
    error = events[-1]
    assert error["code"] == "needs_key" and error["needs_key"] is True
    assert service.store.messages(events[0]["conversation_id"]) == []


def test_chat_api_streams_sse_and_lists_conversations(service: AssistantService) -> None:
    from api.main import app

    client = TestClient(app)
    with client.stream("POST", "/chat", json={"message": "Name a prime number"}) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        events = [
            json.loads(line[6:]) for line in response.iter_lines() if line.startswith("data: ")
        ]
    assert events[-1]["type"] == "done"
    cid = events[0]["conversation_id"]
    listed = client.get("/conversations").json()["conversations"]
    assert listed[0]["id"] == cid
    convo = client.get(f"/conversations/{cid}").json()
    assert [m["role"] for m in convo["messages"]] == ["user", "assistant"]
    assert client.patch(f"/conversations/{cid}", json={"title": "Primes"}).json() == {
        "status": "renamed"
    }
    assert client.delete(f"/conversations/{cid}").json() == {"status": "deleted"}
    assert client.get(f"/conversations/{cid}").status_code == 404


def test_status_never_leaks_keys(service: AssistantService) -> None:
    from api.main import app

    service.pool.add("groq", KEY_A)
    body = TestClient(app).get("/assistant/status").text
    assert KEY_A not in body and "…aaaa" in body


def test_adding_keys_is_checked_and_local_only(service: AssistantService, monkeypatch) -> None:
    from api.main import app

    client = TestClient(app)
    assert (
        client.post("/assistant/keys", json={"provider": "groq", "key": "x" * 10}).status_code
        == 422
    )
    monkeypatch.setattr(
        service.gateway,
        "validate_key",
        lambda p, k: (_ for _ in ()).throw(ProviderError("invalid_key", "401")),
    )
    response = client.post("/assistant/keys", json={"provider": "groq", "key": KEY_B})
    assert response.status_code == 400 and "rejected" in response.json()["detail"]
    monkeypatch.setattr(service.gateway, "validate_key", lambda p, k: ["openai/gpt-oss-20b"])
    added = client.post("/assistant/keys", json={"provider": "groq", "key": KEY_B}).json()
    assert added == {"provider": "groq", "key": "…bbbb", "models": 1}
    assert KEY_B in service.settings.env_path.read_text(encoding="utf-8")
    remote = client.post(
        "/assistant/keys",
        json={"provider": "groq", "key": KEY_B},
        headers={"x-forwarded-for": "8.8.8.8"},
    )
    assert remote.status_code == 403
    with pytest.raises(KeyRejected):
        service.add_key("openai", KEY_B)


def test_store_cache_documents_and_feedback(tmp_path: Path) -> None:
    store = Store(tmp_path / "s.db")
    cid = store.create_conversation("t")
    mid = store.add_message(cid, "assistant", "hi", {"gate": "learned"})
    store.set_feedback(mid, 1, "learned")
    store.set_feedback(mid, -1, "learned")
    assert store.feedback_by_gate() == {"learned": {"up": 0, "down": 1}}
    store.cache_put("k", "q", "a", {"x": 1})
    assert store.cache_get("k", 60)["answer"] == "a" and store.cache_get("k", -1) is None
    assert store.cache_entries(60)[0]["key"] == "k" and store.cache_clear() == 1
    did = store.add_document("notes.md", ["alpha", "beta"])
    assert store.documents()[0]["chunks"] == 2
    assert {c["text"] for c in store.all_chunks([did])} == {"alpha", "beta"}
    store.delete_document(did)
    assert store.all_chunks() == []
    store.log_event("chat", {"n": 1})
    assert store.events("chat")[0]["n"] == 1


def test_stopping_mid_answer_keeps_the_partial_text(service: AssistantService) -> None:
    turn = ChatPipeline(service).run("Tell me a long story")
    cid = None
    for event in turn:
        if event["type"] == "meta":
            cid = event["conversation_id"]
        if event["type"] == "delta":
            break
    turn.close()  # the client went away
    stored = service.store.messages(cid)
    assert [m["role"] for m in stored] == ["user", "assistant"]
    assert stored[1]["meta"]["stopped"] is True and stored[1]["content"]


def test_stopping_before_any_text_removes_the_question(service: AssistantService) -> None:
    turn = ChatPipeline(service).run("Tell me a long story")
    meta = next(turn)
    turn.close()
    assert service.store.messages(meta["conversation_id"]) == []


def test_exhausted_keys_report_why(tmp_path: Path) -> None:
    pool = KeyPool.from_env({"GROQ_API_KEY": KEY_A, "GROQ_API_KEYS": KEY_B})
    assert KeyPool.from_env({}).why_unusable("groq")[0] == "no_key"
    pool.report_invalid("groq", KEY_A, "401")
    pool.report_invalid("groq", KEY_B, "401")
    assert pool.why_unusable("groq")[0] == "invalid_key"
    pool.add("groq", KEY_B)
    pool.report_rate_limited("groq", KEY_B, 120)
    assert pool.why_unusable("groq")[0] == "rate_limited"
    gw = Gateway(settings(tmp_path, providers=["groq"]), pool)
    with pytest.raises(ProvidersExhausted) as info:
        gw.complete("probe", [])
    assert info.value.attempts[0]["reason"] == "rate_limited"


TOOL_FAIL = ProviderError("bad_request", 'HTTP 400: {"error": {"code": "tool_use_failed"}}')


def test_malformed_tool_call_is_retried_without_tools(tmp_path: Path) -> None:
    gw, fake, _ = gateway(tmp_path, [TOOL_FAIL, "plain answer"], {"GROQ_API_KEY": KEY_A})
    seen_tools: list = []
    original = fake.complete

    def spy(key, model, messages, max_tokens, temperature, tools, json_mode=False):
        seen_tools.append(tools)
        return original(key, model, messages, max_tokens, temperature, tools, json_mode)

    fake.complete = spy
    assert gw.complete("agent", [], tools=[{"t": 1}]).text == "plain answer"
    assert seen_tools == [[{"t": 1}], None]


def test_stream_restarts_without_tools_when_a_tool_call_fails_mid_stream(tmp_path: Path) -> None:
    gw, fake, _ = gateway(tmp_path, [], {"GROQ_API_KEY": KEY_A})
    calls: list = []

    def stream(key, model, messages, max_tokens, temperature, tools=None):
        calls.append(tools)

        def broken():
            raise TOOL_FAIL
            yield  # pragma: no cover

        if tools:
            return broken()
        return iter(
            [StreamEvent("delta", text="ok"), StreamEvent("end", provider="groq", model=model)]
        )

    fake.stream = stream
    events = list(gw.stream("answer", [], tools=[{"t": 1}]))
    assert [e.text for e in events if e.type == "delta"] == ["ok"] and calls == [[{"t": 1}], None]


def test_stream_error_chunks_raise() -> None:
    with pytest.raises(ProviderError) as info:
        llm._chunk_error(
            {"error": {"message": "tool call validation failed", "code": "tool_use_failed"}}
        )
    assert llm.tool_use_failed(info.value)
    llm._chunk_error({"choices": []})  # no error, no raise


def test_rate_limit_on_one_model_keeps_the_key_for_other_models(tmp_path: Path) -> None:
    gw, fake, _ = gateway(
        tmp_path, [ProviderError("rate_limited", "x", 600), "ok"], {"GROQ_API_KEY": KEY_A}
    )
    assert gw.complete("answer", []).text == "ok"
    (k1, m1), (k2, m2) = fake.calls
    assert k1 == k2 == KEY_A and m1 != m2
    assert gw.pool.acquire("groq", m1) is None and gw.pool.acquire("groq", m2) == KEY_A


class ToolScript:
    """Gateway stand-in whose agent replies are scripted, tool calls included."""

    def __init__(self, replies: list[Completion]) -> None:
        self.replies = replies
        self.messages: list[list] = []

    def complete(self, role, messages, **kw):
        self.messages.append(list(messages))
        return self.replies.pop(0)


def _tool_reply(name: str, args: str) -> Completion:
    call = {"id": "c1", "type": "function", "function": {"name": name, "arguments": args}}
    return Completion(
        text="",
        tool_calls=[call],
        tools_used=True,
        raw_message={"role": "assistant", "content": None, "tool_calls": [call]},
    )


def test_agent_runs_tools_and_returns_notes() -> None:
    from assistant.agent import AgentResult, calculator_tool, run_agent

    gw = ToolScript(
        [
            _tool_reply("calculate", '{"expression": "6*7"}'),
            _tool_reply("nope", "not json"),
            Completion(text="Notes: 42", tools_used=True),
        ]
    )
    items = list(run_agent(gw, "what is 6*7", [], [calculator_tool()]))
    tools = [i for i in items if not isinstance(i, AgentResult)]
    result = items[-1]
    assert [t["name"] for t in tools] == ["calculate", "nope"]
    assert tools[0]["output"] == "42" and "unknown tool" in tools[1]["output"]
    assert result.notes == "Notes: 42" and len(result.tool_calls) == 2
    sent = gw.messages[1]
    assert sent[-1] == {"role": "tool", "tool_call_id": "c1", "content": "42"}


def test_agent_summarises_when_out_of_rounds() -> None:
    from assistant.agent import calculator_tool, run_agent

    gw = ToolScript(
        [_tool_reply("calculate", '{"expression": "1+1"}'), Completion(text="Summary notes")]
    )
    result = list(run_agent(gw, "q", [], [calculator_tool()], max_rounds=1))[-1]
    assert result.notes == "Summary notes" and gw.messages[-1][-1]["content"].startswith(
        "Summarise"
    )


def test_tool_errors_never_break_the_turn() -> None:
    from assistant.agent import Tool, run_tool_call

    boom = Tool("boom", "fails", {"type": "object"}, lambda args: 1 / 0)
    record = run_tool_call({"boom": boom}, {"function": {"name": "boom", "arguments": "{}"}})
    assert record["output"].startswith("Error: ZeroDivisionError")

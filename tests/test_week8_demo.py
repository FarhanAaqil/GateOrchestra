"""tests/test_week8_demo.py
========================
Unit and integration tests for Week 8 Capstone Live Demonstration,
offline fail-safe mode, LinUCB mock orchestrator, and demo summary API.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from api.main import app
from scripts.week8_demo import (
    DemoTaskRecord,
    MockMASOrchestrator,
    get_demo_gate,
    run_demo,
)
from shared.schemas import Task

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def sample_demo_tasks() -> list[Task]:
    """Provide minimal deterministic tasks for demonstration testing."""
    return [
        Task(
            task_id="demo_test_01",
            question="What is 15 + 27?",
            ground_truth="42",
            source_dataset="template_arithmetic",
            depth_score=1.0,
            parallel_score=1.0,
        ),
        Task(
            task_id="demo_test_02",
            question="Who was the first president of the United States?",
            ground_truth="George Washington",
            source_dataset="hotpotqa_style",
            depth_score=2.0,
            parallel_score=1.0,
        ),
        Task(
            task_id="demo_test_03",
            question="Find the author of the book that inspired the movie Blade Runner.",
            ground_truth="Philip K. Dick",
            source_dataset="musique_style",
            depth_score=3.0,
            parallel_score=2.0,
        ),
    ]


def test_mock_mas_orchestrator(sample_demo_tasks: list[Task]) -> None:
    """Validate MockMASOrchestrator selects strategies and enforces token budgets."""
    mock_mas = MockMASOrchestrator(seed=42)
    task = sample_demo_tasks[0]

    ans, tokens = mock_mas(task, budget=500)
    assert isinstance(ans, str)
    assert 0 < tokens <= 500
    assert mock_mas._last_strategy in ("react", "debate", "reflexion")

    # Verify bandit reward update succeeds without error
    mock_mas.update_bandit_reward(
        task=task,
        strategy=mock_mas._last_strategy,
        is_correct=True,
        tokens_spent=tokens,
        budget=500,
    )


def test_get_demo_gate() -> None:
    """Validate gate loading across gbt, rule, and random options."""
    gbt = get_demo_gate("gbt")
    assert gbt is not None

    rule = get_demo_gate("rule")
    assert rule.__class__.__name__ == "RuleBasedGate"

    random_gate = get_demo_gate("random")
    assert random_gate.__class__.__name__ == "RandomGate"


def test_run_demo_mock_mode(sample_demo_tasks: list[Task]) -> None:
    """Validate run_demo executes end-to-end in offline mock mode."""
    records, summary = run_demo(
        sample_demo_tasks,
        gate_name="gbt",
        k=3,
        mode="mock",
    )

    assert len(records) == 3
    assert summary["total_tasks"] == 3
    assert 0.0 <= summary["accuracy_pct"] <= 100.0
    assert 0.0 <= summary["stop_rate_pct"] <= 100.0
    assert summary["total_gateorchestra_tokens"] > 0
    assert summary["total_always_mas_tokens"] > 0

    for r in records:
        assert isinstance(r, DemoTaskRecord)
        assert r.gate_decision in ("STOP", "ESCALATE")
        assert r.total_tokens > 0
        assert r.always_mas_tokens > 0


def test_run_demo_json_export(sample_demo_tasks: list[Task]) -> None:
    """Ensure demo records can be exported and round-tripped through JSON."""
    records, summary = run_demo(
        sample_demo_tasks,
        gate_name="random",
        k=3,
        mode="mock",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "demo_test.json"
        payload = {
            "summary": summary,
            "records": [r.__dict__ for r in records],
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert loaded["summary"]["total_tasks"] == 3
        assert len(loaded["records"]) == 3


def test_api_demo_summary_endpoint() -> None:
    """Test the /demo/summary API endpoint returns complete capstone metrics."""
    client = TestClient(app)
    response = client.get("/demo/summary")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "research_objectives" in data
    assert "benchmark_methods" in data
    assert "taxonomy" in data
    assert "top_features" in data

    methods = [m["method"] for m in data["benchmark_methods"]]
    assert "GateOrchestra" in methods
    assert "Always-MAS" in methods
    assert "RuleBasedGate" in methods
    assert "CoT-SC-only" in methods

    # Verify RQ values
    rq1 = data["research_objectives"]["rq1_token_savings"]
    assert "78.37%" in rq1["achieved"]

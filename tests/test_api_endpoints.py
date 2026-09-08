"""
tests/test_api_endpoints.py
===========================
Unit and integration tests for the FastAPI application in api/main.py.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestDiscoveryAndHealth:
    def test_root_endpoint(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert "GateOrchestra" in data["name"]

    def test_health_endpoint(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["pipeline"] == "available"

    def test_models_endpoint(self, client: TestClient) -> None:
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "methods" in data
        assert len(data["methods"]) >= 7
        method_ids = [m["id"] for m in data["methods"]]
        assert "GateOrchestra" in method_ids
        assert "RuleBasedGate" in method_ids
        assert "RandomGate" in method_ids
        assert "Always-MAS" in method_ids
        assert "CoT-SC" in method_ids
        assert "providers" in data
        assert "mock" in data["providers"]


class TestTaskExploration:
    def test_tasks_val_default(self, client: TestClient) -> None:
        response = client.get("/tasks?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert data["split"] == "val"
        assert len(data["tasks"]) == 5
        assert data["total"] == 30
        first = data["tasks"][0]
        assert "task_id" in first
        assert "question" in first

    def test_tasks_train_split(self, client: TestClient) -> None:
        response = client.get("/tasks?split=train&limit=10")
        assert response.status_code == 200
        data = response.json()
        assert data["split"] == "train"
        assert len(data["tasks"]) == 10
        assert data["total"] == 90

    def test_tasks_search(self, client: TestClient) -> None:
        response = client.get("/tasks?split=val&search=who")
        assert response.status_code == 200
        data = response.json()
        assert all("who" in t["question"].lower() for t in data["tasks"])

    def test_invalid_split_raises(self, client: TestClient) -> None:
        response = client.get("/tasks?split=unsupported")
        assert response.status_code == 400
        assert "Invalid split" in response.json()["detail"]


class TestExecutionAndHistory:
    def test_clear_and_get_history(self, client: TestClient) -> None:
        del_resp = client.delete("/history")
        assert del_resp.status_code == 200
        assert del_resp.json()["status"] == "cleared"

        hist_resp = client.get("/history")
        assert hist_resp.status_code == 200
        assert hist_resp.json()["count"] == 0

    def test_run_rule_based_gate(self, client: TestClient) -> None:
        payload = {
            "task_id": "test-rbg-01",
            "question": "What is the capital of France?",
            "ground_truth": "Paris",
            "method": "RuleBasedGate",
            "force_simulation": True,
        }
        response = client.post("/run", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "test-rbg-01"
        assert data["method"] == "RuleBasedGate"
        assert data["tokens_spent"] > 0
        assert "gate_decision" in data
        assert data["gate_decision"]["decision"] in ("STOP", "ESCALATE")

    def test_run_gateorchestra_trained(self, client: TestClient) -> None:
        payload = {
            "task_id": "test-gto-01",
            "question": "In what year was Python released?",
            "ground_truth": "1991",
            "method": "GateOrchestra",
            "force_simulation": True,
        }
        response = client.post("/run", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "test-gto-01"
        assert data["method"] == "GateOrchestra"
        assert "tokens_spent" in data

    def test_run_cot_sc_baseline(self, client: TestClient) -> None:
        payload = {
            "task_id": "test-cot-01",
            "question": "Is copper a metal?",
            "ground_truth": "yes",
            "method": "CoT-SC",
            "force_simulation": True,
        }
        response = client.post("/run", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "test-cot-01"
        assert data["method"] == "CoT-SC-only"
        assert data["mas_tokens"] is None
        assert data["tokens_spent"] == data["probe_tokens"]

    def test_run_always_mas_baseline(self, client: TestClient) -> None:
        payload = {
            "task_id": "test-mas-01",
            "question": "Solve this multi-hop problem",
            "ground_truth": "Expected consensus",
            "method": "Always-MAS",
            "force_simulation": True,
        }
        response = client.post("/run", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "test-mas-01"
        assert data["method"] == "Always-MAS"
        assert data["probe_tokens"] is None
        assert data["mas_tokens"] is not None

    def test_history_contains_executed_runs(self, client: TestClient) -> None:
        hist_resp = client.get("/history")
        assert hist_resp.status_code == 200
        history_data = hist_resp.json()
        assert history_data["count"] >= 4
        assert history_data["history"][0]["task_id"] == "test-mas-01"

    def test_run_unknown_method_raises_400(self, client: TestClient) -> None:
        payload = {
            "task_id": "test-fail-01",
            "question": "Question?",
            "method": "NonExistentMethod",
            "force_simulation": True,
        }
        response = client.post("/run", json=payload)
        assert response.status_code == 400
        assert "Unknown method" in response.json()["detail"]

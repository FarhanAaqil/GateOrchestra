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

import api.main as api_main
from api.main import app
from gate.classifier import ConstantGate
from shared.data_loader import load_split


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def trained_gate(monkeypatch: pytest.MonkeyPatch) -> ConstantGate:
    """Provide a trained gate without touching configs/models/."""
    gate = ConstantGate("STOP")
    gate.train([], ["STOP"])
    monkeypatch.setattr(api_main, "_CACHED_LEARNED_GATE", gate)
    return gate


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
        method_ids = [m["id"] for m in data["methods"]]
        # Only methods the evaluation can produce; no dummy-trained gates.
        assert set(method_ids) == {
            "GateOrchestra",
            "RuleBasedGate",
            "RandomGate",
            "Always-MAS",
            "CoT-SC",
        }
        assert all(m["name"].isascii() for m in data["methods"])
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
        assert data["total"] == len(load_split("val"))
        first = data["tasks"][0]
        assert "task_id" in first
        assert "question" in first

    def test_tasks_train_split(self, client: TestClient) -> None:
        response = client.get("/tasks?split=train&limit=10")
        assert response.status_code == 200
        data = response.json()
        assert data["split"] == "train"
        assert len(data["tasks"]) == 10
        assert data["total"] == len(load_split("train"))

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

    def test_run_gateorchestra_trained(self, client: TestClient, trained_gate) -> None:
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

    def test_simulation_never_sees_or_grades_the_ground_truth(self, client: TestClient) -> None:
        for method in ("CoT-SC", "Always-MAS", "RuleBasedGate", "RandomGate"):
            payload = {
                "task_id": f"test-leak-{method}",
                "question": "What is the capital of Tanzania's semi-autonomous archipelago?",
                "ground_truth": "Zanzibar",
                "method": method,
                "force_simulation": True,
            }
            data = client.post("/run", json=payload).json()
            assert data["simulated"] is True
            assert data["is_correct"] is None
            assert "zanzibar" not in data["predicted_answer"].lower()

    def test_random_gate_varies_across_requests(self, client: TestClient) -> None:
        decisions = set()
        for i in range(12):
            payload = {
                "task_id": f"test-rand-{i}",
                "question": "Which is larger: the Moon or Mars?",
                "method": "RandomGate",
                "force_simulation": True,
            }
            decisions.add(client.post("/run", json=payload).json()["gate_decision"]["decision"])
        assert decisions == {"STOP", "ESCALATE"}

    def test_gateorchestra_without_trained_gate_returns_503(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setattr(api_main, "_CACHED_LEARNED_GATE", None)
        monkeypatch.setattr(api_main, "_gate_model_candidates", lambda: [tmp_path / "none.pkl"])
        payload = {
            "task_id": "test-no-gate",
            "question": "Question?",
            "method": "GateOrchestra",
            "force_simulation": True,
        }
        response = client.post("/run", json=payload)
        assert response.status_code == 503
        assert "No trained gate" in response.json()["detail"]

    def test_demo_summary_has_no_hardcoded_numbers(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setattr(api_main, "MASTER_RESULTS_PATH", tmp_path / "missing.json")
        data = client.get("/demo/summary").json()
        assert data["status"] == "no_results"
        assert data["aggregated"] == []

    def test_history_contains_executed_runs(self, client: TestClient) -> None:
        hist_resp = client.get("/history")
        assert hist_resp.status_code == 200
        history_data = hist_resp.json()
        assert history_data["count"] >= 4
        assert any(run["task_id"] == "test-mas-01" for run in history_data["history"])

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


class TestGateProvenance:
    def _isolate(self, monkeypatch: pytest.MonkeyPatch, paths) -> None:
        monkeypatch.setattr(api_main, "_CACHED_LEARNED_GATE", None)
        monkeypatch.setattr(api_main, "_CACHED_GATE_INFO", None)
        monkeypatch.setattr(api_main, "_CACHED_GATE_KEY", None)
        monkeypatch.setattr(api_main, "_gate_model_candidates", lambda: paths)

    def test_gate_without_build_info_is_never_served(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        import pickle

        legacy = ConstantGate("STOP")
        legacy.train([], ["STOP"])
        path = tmp_path / "best_gate.pkl"
        path.write_bytes(pickle.dumps(legacy))  # written without save(): no build_info
        self._isolate(monkeypatch, [path])

        assert client.get("/models").json()["trained_gate"] is None
        payload = {
            "task_id": "t",
            "question": "Q?",
            "method": "GateOrchestra",
            "force_simulation": True,
        }
        assert client.post("/run", json=payload).status_code == 503

    def test_newest_benchmark_gate_is_served_with_provenance(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        gate = ConstantGate("STOP")
        gate.train([], ["STOP"])
        path = tmp_path / "best_gate_seed_123.pkl"
        gate.save(path)
        self._isolate(monkeypatch, [path])

        info = client.get("/models").json()["trained_gate"]
        assert info["seed"] == 123
        assert info["classifier"] == "ConstantGate"
        assert info["degenerate"] is True
        assert info["build_info"]["sklearn_version"]
        payload = {
            "task_id": "t2",
            "question": "Q?",
            "method": "GateOrchestra",
            "force_simulation": True,
        }
        assert client.post("/run", json=payload).status_code == 200

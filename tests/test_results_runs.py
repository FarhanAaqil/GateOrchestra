"""The Results page can switch between the main benchmark and the tagged stronger-model runs."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app, available_runs

client = TestClient(app)


def test_runs_list_puts_the_main_benchmark_first() -> None:
    runs = client.get("/results/runs").json()["runs"]
    assert runs and runs[0]["id"] == "final" and runs[0]["main"] is True
    assert [r["id"] for r in runs] == available_runs()
    assert all(r["label"] and isinstance(r["seeds"], list) for r in runs)


def test_summary_and_detail_default_to_the_main_benchmark() -> None:
    default = client.get("/demo/summary").json()
    explicit = client.get("/demo/summary", params={"run": "final"}).json()
    assert default["aggregated"] == explicit["aggregated"]
    assert client.get("/results/detail").json()["status"] == explicit["status"]


def test_a_tagged_run_is_served_and_marked_partial() -> None:
    tagged = [r for r in available_runs() if r != "final"]
    if not tagged:  # nothing but the main run on disk
        return
    summary = client.get("/demo/summary", params={"run": tagged[0]}).json()
    assert summary["status"] == "partial" and summary["seeds"] == [42]
    detail = client.get("/results/detail", params={"run": tagged[0]}).json()
    assert detail["status"] == "partial" and detail["escalation_quality"]


def test_unknown_or_path_like_runs_are_refused() -> None:
    for run in ("does-not-exist", "../final", "archive"):
        assert client.get("/demo/summary", params={"run": run}).status_code == 404
        assert client.get("/results/detail", params={"run": run}).status_code == 404

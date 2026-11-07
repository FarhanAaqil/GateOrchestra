"""Shared pytest fixtures."""

from __future__ import annotations

import urllib.request

import pytest


@pytest.fixture(autouse=True)
def _no_real_llm_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any unmocked HTTP call so the suite never spends real tokens.

    Tests that exercise the providers patch ``urllib.request.urlopen``
    themselves, which takes precedence over this fixture.
    """

    def _blocked(*args: object, **kwargs: object) -> None:
        raise OSError("Network access is disabled in tests; inject an llm_caller instead.")

    monkeypatch.setattr(urllib.request, "urlopen", _blocked)


@pytest.fixture(autouse=True)
def _isolated_api_history(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Keep each test's API run history in its own file, never in logs/."""
    import sys

    api_main = sys.modules.get("api.main")
    if api_main is not None:
        from api.history import RunHistory

        monkeypatch.setattr(api_main, "_HISTORY", RunHistory(tmp_path / "history.jsonl"))

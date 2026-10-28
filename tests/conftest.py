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

"""scripts/make_env.py and scripts/check_deploy.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from assistant.hosted import hosted_config_problems
from scripts import check_deploy, make_env


def generate(tmp_path: Path) -> Path:
    out = tmp_path / ".env"
    assert (
        make_env.main(
            [
                "--domain",
                "https://Gate.Example.com/",
                "--admin-email",
                "Boss@Example.com",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    return out


def test_make_env_generates_every_secret_and_never_overwrites(tmp_path: Path) -> None:
    out = generate(tmp_path)
    text = out.read_text(encoding="utf-8")
    assert "GATE_HOSTED=true" in text and "GATE_DOMAIN=gate.example.com" in text
    assert "GATE_ADMIN_EMAILS=boss@example.com" in text and "GATE_REQUIRE_ADMIN_2FA=true" in text
    values = check_deploy.effective_env(out)
    assert len(values["GATE_ADMIN_TOKEN"]) >= 32 and values["GATE_ENCRYPTION_KEY"].endswith("=")
    assert values["GATE_CORS_ORIGINS"] == "https://gate.example.com"
    # Run twice, the secrets differ; an existing file is refused without --force.
    assert (
        make_env.main(
            ["--domain", "gate.example.com", "--admin-email", "a@example.com", "--out", str(out)]
        )
        == 1
    )
    second = tmp_path / "other.env"
    make_env.main(
        ["--domain", "gate.example.com", "--admin-email", "a@example.com", "--out", str(second)]
    )
    assert check_deploy.effective_env(second)["GATE_ADMIN_TOKEN"] != values["GATE_ADMIN_TOKEN"]
    assert (
        make_env.main(
            [
                "--domain",
                "not a domain",
                "--admin-email",
                "a@example.com",
                "--out",
                str(tmp_path / "x"),
            ]
        )
        == 1
    )


def test_a_generated_env_only_lacks_what_the_operator_must_supply(tmp_path: Path) -> None:
    env = check_deploy.effective_env(generate(tmp_path))
    problems = hosted_config_problems(env)
    assert any("API key is required" in p for p in problems)
    assert any("GATE_SMTP_HOST" in p for p in problems)
    filled = {
        **env,
        "GROQ_API_KEY": "gsk_" + "k" * 40,
        "GATE_SMTP_HOST": "smtp.example.com",
        "GATE_MAIL_FROM": "noreply@example.com",
    }
    assert hosted_config_problems(filled) == []


def test_check_config_reports_problems(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    out = generate(tmp_path)
    assert check_deploy.main(["--env", str(out)]) == 1
    printed = capsys.readouterr().out
    assert "FAIL  A cloud API key is required" in printed
    assert "ok    configs/plans.yaml loads" in printed
    assert check_deploy.main(["--env", str(tmp_path / "missing.env")]) == 1


HEALTHY = {
    "https://gate.example.com/": (
        200,
        dict.fromkeys(check_deploy.PAGE_HEADERS, "x") | {"strict-transport-security": "max-age=1"},
        '<div id="root"></div>',
    ),
    "http://gate.example.com/": (308, {"location": "https://gate.example.com/"}, ""),
    "https://gate.example.com/api/health": (200, {}, '{"status": "ok"}'),
    "https://gate.example.com/api/": (200, {}, '{"hosted": true}'),
    "https://gate.example.com/api/auth/me": (401, {"cache-control": "no-store"}, ""),
    "https://gate.example.com/api/assistant/stats": (403, {}, ""),
}


def test_live_check_passes_a_healthy_site(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check_deploy, "fetch", lambda url, method="GET", follow=True: HEALTHY[url])
    assert check_deploy.main(["--url", "https://gate.example.com"]) == 0


def test_live_check_names_what_is_wrong(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    broken = dict(HEALTHY)
    broken["https://gate.example.com/api/"] = (200, {}, '{"hosted": false}')
    broken["https://gate.example.com/api/assistant/stats"] = (200, {}, "{}")
    broken["https://gate.example.com/"] = (200, {}, '<div id="root"></div>')
    monkeypatch.setattr(check_deploy, "fetch", lambda url, method="GET", follow=True: broken[url])
    assert check_deploy.main(["--url", "https://gate.example.com"]) == 1
    printed = capsys.readouterr().out
    assert "FAIL  the API runs in hosted mode" in printed
    assert "FAIL  operator statistics are admin-only (403)" in printed
    assert "FAIL  the page sends content-security-policy" in printed
    assert (
        check_deploy.main(["--url", "http://gate.example.com"]) == 1
    )  # plain http needs --allow-http

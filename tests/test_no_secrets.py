"""Fail the build if a real credential is committed.

Scans every file git tracks for token formats that providers use. The
documented placeholder (``gsk_your_groq_api_key_here``) does not match because
real tokens are long runs of random characters.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

TOKEN_PATTERNS = {
    "GitHub token": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b"),
    "GitHub fine-grained token": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{60,}\b"),
    "Groq API key": re.compile(r"\bgsk_[A-Za-z0-9]{40,}\b"),
    "OpenAI-style key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{40,}\b"),
    "Anthropic key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{40,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "Private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "Credentials in a URL": re.compile(r"https://[^/\s:@]+:[^/\s@]{20,}@github\.com"),
}


def _tracked_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [ROOT / name for name in out.decode("utf-8").split("\0") if name]


def test_no_credentials_are_committed() -> None:
    findings = []
    for path in _tracked_files():
        if not path.is_file() or path.stat().st_size > 5_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary file
        for kind, pattern in TOKEN_PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append(f"{path.relative_to(ROOT)}:{line}: {kind}")
    assert not findings, "Possible credentials committed:\n" + "\n".join(findings)


@pytest.mark.parametrize(
    "secret",
    [
        "ghp_" + "a1B2" * 9,
        "gsk_" + "Z9y8" * 13,
        "https://someone:" + "x" * 30 + "@github.com/o/r.git",
    ],
)
def test_patterns_catch_real_token_shapes(secret: str) -> None:
    assert any(p.search(secret) for p in TOKEN_PATTERNS.values())


def test_documented_placeholder_is_allowed() -> None:
    assert not any(p.search("gsk_your_groq_api_key_here") for p in TOKEN_PATTERNS.values())

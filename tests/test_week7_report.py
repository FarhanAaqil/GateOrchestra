"""tests/test_week7_report.py
==========================
Unit tests validating Person 3 Capstone Technical Report content,
mathematical formulations, schema references, and data integrity.
"""

from __future__ import annotations

from scripts.verify_week7_report import (
    FIGURE_PATH,
    REPORT_PATH,
    REQUIRED_FEATURES,
    REQUIRED_SCHEMAS,
    REQUIRED_SECTIONS,
    verify_report,
)


def test_report_file_exists_and_structured() -> None:
    """Validate report file existence, non-trivial length, and required section headers."""
    assert REPORT_PATH.exists(), f"Report file missing: {REPORT_PATH}"
    content = REPORT_PATH.read_text(encoding="utf-8")
    assert len(content.strip()) > 10000, "Report is unexpectedly brief (<10,000 characters)"

    for section in REQUIRED_SECTIONS:
        assert section in content, f"Missing required section header: {section}"


def test_mathematical_formulations_present() -> None:
    """Validate presence of key mathematical gating and budget equations."""
    content = REPORT_PATH.read_text(encoding="utf-8")

    # Gating decision rule & conditional probability
    assert "P(\\text{ESCALATE}" in content or "P(ESCALATE" in content
    assert "\\theta" in content or "theta" in content

    # Accuracy difference labeling rule
    assert (
        "\\Delta_{\\text{acc}}" in content
        or "A_{MAS}" in content
        or "\\tau_{\\text{acc}}" in content
    )

    # Dynamic token budget clamping
    assert (
        "B_{\\text{MAS}}" in content
        or "k \\cdot T_{\\text{probe}}" in content
        or "k \\times \\text{probe\\_tokens}" in content
    )


def test_schemas_and_features_documented() -> None:
    """Validate all core schemas and 8 features are thoroughly documented."""
    content = REPORT_PATH.read_text(encoding="utf-8")

    for schema in REQUIRED_SCHEMAS:
        assert schema in content, f"Schema '{schema}' is not referenced in the report"

    for feature in REQUIRED_FEATURES:
        assert feature in content, f"Feature '{feature}' is not documented in the report"


def test_empirical_research_questions_covered() -> None:
    """Validate that RQ1, RQ2, and RQ3 are explicitly addressed with empirical claims."""
    content = REPORT_PATH.read_text(encoding="utf-8")

    # RQ1: Token savings
    assert "RQ1" in content
    assert "78.37%" in content or "78.4%" in content

    # RQ2: Accuracy preservation
    assert "RQ2" in content
    assert "82.29%" in content or "87.5" in content

    # RQ3: Pareto frontier optimality
    assert "RQ3" in content
    assert "Pareto" in content
    assert FIGURE_PATH.exists(), f"Pareto frontier figure file missing at {FIGURE_PATH}"


def test_verification_script_passes() -> None:
    """Ensure the automated verification script returns True."""
    assert verify_report() is True

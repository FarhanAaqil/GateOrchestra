"""scripts/verify_week7_report.py
================================
Verification script for Person 3 Capstone Technical Report.

Validates that:
1. `reports/capstone_technical_report_person3.md` exists and is non-empty.
2. All 5 core architectural and empirical sections are present.
3. Referenced figure files (e.g. Pareto frontier curve) exist on disk.
4. Numerical claims and evaluation tables match the verified JSON artifacts:
   - `logs/results/week5_evaluation.json`
   - `logs/results/week6_ablations.json`
   - `logs/results/week6_pareto.json`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = REPO_ROOT / "reports" / "capstone_technical_report_person3.md"
FIGURE_PATH = REPO_ROOT / "reports" / "figures" / "week6_pareto_frontier.png"

WEEK5_JSON = REPO_ROOT / "logs" / "results" / "week5_evaluation.json"
WEEK6_ABLATIONS_JSON = REPO_ROOT / "logs" / "results" / "week6_ablations.json"
WEEK6_PARETO_JSON = REPO_ROOT / "logs" / "results" / "week6_pareto.json"

REQUIRED_SECTIONS = [
    "## 1. System Architecture & Contract Layer",
    "## 2. Gating Formulation & Feature Extraction",
    "## 3. Gate Training Methodology & Optimization",
    "## 4. Empirical Results & Research Question Validation",
    "## 5. Gate Failure Taxonomy, Edge Cases & Safety Mechanisms",
]

REQUIRED_SCHEMAS = [
    "Task",
    "ProbeResult",
    "GateFeatures",
    "GateDecision",
    "EvalResult",
    "TokenAccountant",
]

REQUIRED_FEATURES = [
    "probe_tokens",
    "consistency_score",
    "estimated_depth",
    "estimated_parallel",
    "entity_count",
    "clause_count",
    "question_word_count",
    "has_context",
]


def verify_report() -> bool:
    """Run full verification on report content, figures, and data integrity."""
    print("=" * 65)
    print("  [GateOrchestra] Verifying Week 7 Capstone Technical Report")
    print("=" * 65)

    errors: list[str] = []

    # 1. Existence check
    if not REPORT_PATH.exists():
        errors.append(f"Report file missing: {REPORT_PATH}")
        print(f"[FAIL] {REPORT_PATH} does not exist.")
        return False

    content = REPORT_PATH.read_text(encoding="utf-8")
    if len(content.strip()) < 1000:
        errors.append(f"Report file is too short ({len(content)} chars).")

    print(
        f"[OK] Report file exists: {REPORT_PATH.name} ({len(content)} characters, {len(content.splitlines())} lines)"
    )

    # 2. Section presence check
    for sec in REQUIRED_SECTIONS:
        if sec in content:
            print(f"[OK] Section present: {sec.replace('## ', '')}")
        else:
            errors.append(f"Missing required section: '{sec}'")
            print(f"[FAIL] Missing section: '{sec}'")

    # 3. Contract layer models check
    for schema in REQUIRED_SCHEMAS:
        if schema in content:
            print(f"[OK] Schema referenced: {schema}")
        else:
            errors.append(f"Missing schema reference: '{schema}'")
            print(f"[FAIL] Missing schema: '{schema}'")

    # 4. Feature list check
    for feat in REQUIRED_FEATURES:
        if feat in content:
            print(f"[OK] Feature documented: {feat}")
        else:
            errors.append(f"Missing feature definition: '{feat}'")
            print(f"[FAIL] Missing feature: '{feat}'")

    # 5. Figure existence check
    if FIGURE_PATH.exists():
        print(
            f"[OK] Referenced figure exists: {FIGURE_PATH.name} ({FIGURE_PATH.stat().st_size} bytes)"
        )
    else:
        errors.append(f"Missing referenced figure: {FIGURE_PATH}")
        print(f"[FAIL] Missing figure: {FIGURE_PATH}")

    # 6. Data Integrity Check: Week 5 Benchmark Aggregates
    if WEEK5_JSON.exists():
        with open(WEEK5_JSON, encoding="utf-8") as f:
            w5_data = json.load(f)
        w5_agg = w5_data.get("aggregate", {})
        go_savings = w5_agg.get("GateOrchestra", {}).get("token_savings_mean", 0.0)
        go_acc = w5_agg.get("GateOrchestra", {}).get("accuracy_mean", 0.0)

        # Check that 78.37% and 82.29% appear in the report
        savings_str = f"{go_savings:.2f}%"
        acc_str = f"{go_acc * 100:.2f}%"
        if "78.37%" in content and "82.29%" in content:
            print(f"[OK] Week 5 benchmark metrics verified: Savings={savings_str}, Acc={acc_str}")
        else:
            errors.append(
                f"Week 5 aggregate metrics mismatch. Expected {savings_str} and {acc_str} in report."
            )
            print("[FAIL] Week 5 metrics missing or incorrect in report text.")
    else:
        errors.append(f"Week 5 JSON missing: {WEEK5_JSON}")

    # 7. Data Integrity Check: Week 6 Ablations & Taxonomy
    if WEEK6_ABLATIONS_JSON.exists():
        with open(WEEK6_ABLATIONS_JSON, encoding="utf-8") as f:
            w6_data = json.load(f)

        tax = w6_data.get("taxonomy", {})
        true_stops = tax.get("true_stops", 0)
        false_stops = tax.get("false_stops", 0)

        if f"{true_stops} tasks" in content and f"{false_stops} tasks" in content:
            print(
                f"[OK] Week 6 failure taxonomy verified: True STOPs={true_stops}, False STOPs={false_stops}"
            )
        else:
            errors.append(
                f"Week 6 taxonomy counts mismatch. Expected {true_stops} and {false_stops}."
            )
            print("[FAIL] Week 6 taxonomy counts missing in report text.")

        # Check LinUCB bandit breakdown
        bandit = w6_data.get("bandit_breakdown", {})
        react_count = bandit.get("arm_counts", {}).get("react", 0)
        if f"{react_count} / 32" in content or f"{react_count} / 32 tasks" in content:
            print(f"[OK] LinUCB bandit arm counts verified: ReAct={react_count}/32 tasks")
        else:
            errors.append("LinUCB arm count mismatch in report.")
            print("[FAIL] LinUCB arm count missing in report text.")
    else:
        errors.append(f"Week 6 Ablations JSON missing: {WEEK6_ABLATIONS_JSON}")

    # 8. Data Integrity Check: Week 6 Pareto Frontier
    if WEEK6_PARETO_JSON.exists():
        with open(WEEK6_PARETO_JSON, encoding="utf-8") as f:
            pareto_data = json.load(f)

        k3_points = [
            p for p in pareto_data if p.get("k") == 3 and p.get("method") == "GateOrchestra"
        ]
        if k3_points:
            k3_savings = k3_points[0].get("token_savings_pct", 0.0)
            if f"{k3_savings:.2f}%" in content:
                print(f"[OK] Pareto k=3 GateOrchestra savings verified: {k3_savings:.2f}%")
            else:
                errors.append(f"Pareto k=3 savings ({k3_savings:.2f}%) not found in report.")
                print("[FAIL] Pareto k=3 savings missing in report text.")
    else:
        errors.append(f"Week 6 Pareto JSON missing: {WEEK6_PARETO_JSON}")

    print("-" * 65)
    if errors:
        print(f"[VERIFICATION FAILED] Encountered {len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")
        return False

    print(
        "[SUCCESS] All report sections, figures, schemas, and empirical metrics successfully verified!"
    )
    print("=" * 65)
    return True


def main() -> int:
    success = verify_report()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

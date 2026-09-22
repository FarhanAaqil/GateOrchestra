"""
dataset/error_labels.py
=======================
Error taxonomy, failure classification, and diagnostic profiling engine.

Provides formal error categorizations for routing and single-agent vs MAS
execution failures, supporting post-hoc evaluation audits and continuous
dataset calibration.

Person 1 owns this file.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from shared.schemas import Task

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Error Taxonomy Definitions
# ─────────────────────────────────────────────────────────────────────────────


class ErrorType(str, Enum):
    """Taxonomy of failure modes in task resolution and agent routing."""

    REASONING_ERROR = "reasoning_error"
    CALCULATION_ERROR = "calculation_error"
    FACTUAL_ERROR = "factual_error"
    CONCURRENCY_ERROR = "concurrency_error"
    FORMAT_ERROR = "format_error"
    CONTEXT_MISALIGNMENT = "context_misalignment"
    UNKNOWN = "unknown"

    @classmethod
    def all_types(cls) -> list[str]:
        return [e.value for e in cls]


ERROR_DESCRIPTIONS: dict[ErrorType, str] = {
    ErrorType.REASONING_ERROR: (
        "Logical breakdown in multi-hop bridge reasoning, invalid deduction step, "
        "or incorrect relational inference between facts."
    ),
    ErrorType.CALCULATION_ERROR: (
        "Arithmetic slip, computational operator error, incorrect formula application, "
        "or unit conversion mismatch."
    ),
    ErrorType.FACTUAL_ERROR: (
        "Entity hallucination, incorrect entity retrieval, wrong historical date, "
        "or attribute confusion."
    ),
    ErrorType.CONCURRENCY_ERROR: (
        "Failure to resolve one of multiple parallel sub-questions, dropped branch, "
        "or incomplete joint aggregation."
    ),
    ErrorType.FORMAT_ERROR: (
        "Failure to produce canonical expected output, answer extraction failure, "
        "or unparseable response structure."
    ),
    ErrorType.CONTEXT_MISALIGNMENT: (
        "Ignoring, misreading, or contradicting premise facts provided in the task context."
    ),
    ErrorType.UNKNOWN: "Uncategorized failure or ambiguous model output.",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures & Models
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ErrorAnnotation:
    """Diagnostic annotation for an individual incorrect task execution."""

    task_id: str
    error_type: ErrorType
    predicted_answer: str
    ground_truth: str
    severity: str = "MAJOR"  # "MINOR", "MAJOR", "CRITICAL"
    rationale: str = ""
    source_dataset: str | None = None
    depth_score: int | None = None
    parallel_score: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        res = asdict(self)
        res["error_type"] = self.error_type.value
        return res


@dataclass
class ErrorBreakdown:
    """Aggregated failure statistics across dimensions."""

    total_errors: int = 0
    by_error_type: dict[str, int] = field(default_factory=dict)
    by_source: dict[str, int] = field(default_factory=dict)
    by_depth: dict[str, int] = field(default_factory=dict)
    by_parallel: dict[str, int] = field(default_factory=dict)
    by_severity: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ErrorAnalysisReport:
    """Comprehensive failure analysis report across evaluated tasks."""

    total_evaluated: int
    total_errors: int
    error_rate_pct: float
    breakdown: ErrorBreakdown
    annotations: list[ErrorAnnotation] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_evaluated": self.total_evaluated,
            "total_errors": self.total_errors,
            "error_rate_pct": self.error_rate_pct,
            "breakdown": self.breakdown.to_dict(),
            "annotations": [a.to_dict() for a in self.annotations],
            "recommendations": self.recommendations,
            "generated_at": self.generated_at,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic Engine
# ─────────────────────────────────────────────────────────────────────────────


class ErrorAnalyzer:
    """Classifies task resolution failures into canonical error taxonomy categories."""

    @staticmethod
    def is_exact_match(prediction: str, ground_truth: str) -> bool:
        """Check if normalized prediction matches normalized ground truth."""
        norm_pred = prediction.strip().lower().rstrip(".").strip("'\"")
        norm_gt = ground_truth.strip().lower().rstrip(".").strip("'\"")
        return norm_pred == norm_gt

    def diagnose_error(
        self,
        task: Task,
        predicted_answer: str,
        error_type_override: ErrorType | None = None,
        severity: str = "MAJOR",
        rationale: str = "",
    ) -> ErrorAnnotation:
        """Diagnose probable failure mode for an incorrect prediction."""
        if error_type_override is not None:
            detected_type = error_type_override
        else:
            detected_type = self._infer_error_type(task, predicted_answer)

        if not rationale:
            rationale = ERROR_DESCRIPTIONS.get(detected_type, "")

        return ErrorAnnotation(
            task_id=task.task_id,
            error_type=detected_type,
            predicted_answer=predicted_answer,
            ground_truth=str(task.ground_truth),
            severity=severity,
            rationale=rationale,
            source_dataset=task.source_dataset,
            depth_score=task.depth_score,
            parallel_score=task.parallel_score,
        )

    def _infer_error_type(self, task: Task, prediction: str) -> ErrorType:
        """Heuristic rule-based inference for failure classification."""
        pred_clean = prediction.strip().lower()
        gt_clean = str(task.ground_truth).strip().lower()
        q_lower = (task.question or "").lower()

        # 1. Format / Extraction Error
        if len(pred_clean) == 0 or pred_clean in {"none", "null", "n/a", "unknown", "error"}:
            return ErrorType.FORMAT_ERROR

        if len(pred_clean) > 300 and len(gt_clean) < 30:
            return ErrorType.FORMAT_ERROR

        # 2. Calculation / Arithmetic Error
        num_pattern = r"^-?\d+(\.\d+)?$"
        is_num_gt = bool(re.match(num_pattern, gt_clean.replace(",", "")))
        is_num_pred = bool(re.match(num_pattern, pred_clean.replace(",", "")))

        if (
            task.source_dataset == "template_arithmetic"
            or "calculate" in q_lower
            or "multiplied" in q_lower
            or (is_num_gt and is_num_pred)
        ):
            return ErrorType.CALCULATION_ERROR

        # 3. Concurrency / Multi-part Error
        if (
            (task.parallel_score and task.parallel_score >= 2)
            or " and " in task.question
            or "both" in q_lower
        ):
            # If ground truth has multiple parts (e.g. comma or semicolon separated)
            if ("," in gt_clean or " and " in gt_clean) and (
                "," not in pred_clean and " and " not in pred_clean
            ):
                return ErrorType.CONCURRENCY_ERROR

        # 4. Context Misalignment
        if task.context:
            ctx_lower = task.context.lower()
            # If ground truth appears in context but prediction doesn't
            if gt_clean in ctx_lower and pred_clean not in ctx_lower:
                return ErrorType.CONTEXT_MISALIGNMENT

        # 5. Reasoning vs Factual
        if task.source_dataset in {"musique_style", "hotpotqa_style"}:
            if task.depth_score and task.depth_score >= 3:
                return ErrorType.REASONING_ERROR
            return ErrorType.FACTUAL_ERROR

        if task.source_dataset == "template_comparison":
            return ErrorType.REASONING_ERROR

        return ErrorType.REASONING_ERROR

    def compute_breakdown(self, annotations: list[ErrorAnnotation]) -> ErrorBreakdown:
        """Compute aggregated error breakdowns across all dimensions."""
        by_type: dict[str, int] = dict.fromkeys(ErrorType.all_types(), 0)
        by_source: dict[str, int] = {}
        by_depth: dict[str, int] = {}
        by_parallel: dict[str, int] = {}
        by_severity: dict[str, int] = {}

        for a in annotations:
            t_val = a.error_type.value
            by_type[t_val] = by_type.get(t_val, 0) + 1

            if a.source_dataset:
                by_source[a.source_dataset] = by_source.get(a.source_dataset, 0) + 1

            if a.depth_score is not None:
                d_key = f"depth_{a.depth_score}"
                by_depth[d_key] = by_depth.get(d_key, 0) + 1

            if a.parallel_score is not None:
                p_key = f"parallel_{a.parallel_score}"
                by_parallel[p_key] = by_parallel.get(p_key, 0) + 1

            by_severity[a.severity] = by_severity.get(a.severity, 0) + 1

        return ErrorBreakdown(
            total_errors=len(annotations),
            by_error_type={k: v for k, v in by_type.items() if v > 0},
            by_source=by_source,
            by_depth=by_depth,
            by_parallel=by_parallel,
            by_severity=by_severity,
        )

    def analyze_predictions(
        self,
        tasks: list[Task],
        predictions: dict[str, str],
    ) -> ErrorAnalysisReport:
        """Analyze a mapping of task_id -> predicted_answer against ground truth."""
        annotations: list[ErrorAnnotation] = []
        total_eval = len(tasks)

        for task in tasks:
            pred = predictions.get(task.task_id, "")
            if not self.is_exact_match(pred, str(task.ground_truth)):
                annotation = self.diagnose_error(task, pred)
                annotations.append(annotation)

        breakdown = self.compute_breakdown(annotations)
        err_rate = round((len(annotations) / total_eval * 100.0), 2) if total_eval > 0 else 0.0

        recommendations = self._generate_recommendations(breakdown, err_rate)

        return ErrorAnalysisReport(
            total_evaluated=total_eval,
            total_errors=len(annotations),
            error_rate_pct=err_rate,
            breakdown=breakdown,
            annotations=annotations,
            recommendations=recommendations,
        )

    def _generate_recommendations(self, breakdown: ErrorBreakdown, error_rate: float) -> list[str]:
        """Generate targeted recommendations based on diagnostic breakdown."""
        recs: list[str] = []
        if breakdown.total_errors == 0:
            return ["No errors detected in evaluated tasks. System performing optimally."]

        by_type = breakdown.by_error_type
        if by_type.get(ErrorType.CALCULATION_ERROR.value, 0) > 0:
            recs.append(
                "Calculation errors detected: Ensure arithmetic tasks route to symbolic calculation sub-agents or use calculator tool bindings."
            )

        if by_type.get(ErrorType.CONCURRENCY_ERROR.value, 0) > 0:
            recs.append(
                "Concurrency errors detected: Multi-part questions should escalate to multi-agent debate or parallel sub-question dispatch."
            )

        if by_type.get(ErrorType.REASONING_ERROR.value, 0) > 0:
            recs.append(
                "Reasoning errors detected: Tasks with depth >= 3 require multi-step chain-of-thought verification or Reflexion agent review."
            )

        if by_type.get(ErrorType.FORMAT_ERROR.value, 0) > 0:
            recs.append(
                "Format errors detected: Strengthen final-answer extraction parser and canonical prompt formatting instructions."
            )

        if error_rate > 30.0:
            recs.append(
                "High overall failure rate (>30%): Lower gate escalation threshold to direct more ambiguous queries to MAS orchestration."
            )

        return recs

    # ─────────────────────────────────────────────────────────────────────────
    # Renderers & Exporters
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def render_ascii_summary(report: ErrorAnalysisReport) -> str:
        """Generate plain-text summary of failure diagnostics."""
        lines = [
            "=" * 70,
            "  GATEORCHESTRA ERROR ANALYSIS & DIAGNOSTICS REPORT",
            "=" * 70,
            f"  Total Evaluated Tasks : {report.total_evaluated}",
            f"  Total Failures        : {report.total_errors}",
            f"  Failure Rate          : {report.error_rate_pct:.1f}%",
            f"  Generated At (UTC)    : {report.generated_at}",
            "",
            "1. ERROR CATEGORY BREAKDOWN",
            "-" * 40,
        ]
        for e_type, count in sorted(report.breakdown.by_error_type.items(), key=lambda x: -x[1]):
            pct = count / report.total_errors * 100.0 if report.total_errors > 0 else 0.0
            lines.append(f"  * {e_type:<24} : {count:>4} ({pct:>5.1f}%)")

        lines.extend(["", "2. FAILURES BY REASONING DEPTH", "-" * 40])
        for d_key, count in sorted(report.breakdown.by_depth.items()):
            lines.append(f"  * {d_key:<24} : {count:>4}")

        lines.extend(["", "3. FAILURES BY SOURCE DATASET", "-" * 40])
        for s_key, count in sorted(report.breakdown.by_source.items()):
            lines.append(f"  * {s_key:<24} : {count:>4}")

        lines.extend(["", "4. RECOMMENDATIONS & MITIGATIONS", "-" * 40])
        for rec in report.recommendations:
            lines.append(f"  -> {rec}")

        lines.append("=" * 70)
        return "\n".join(lines)

    @staticmethod
    def render_markdown_report(report: ErrorAnalysisReport) -> str:
        """Generate GitHub Flavored Markdown failure analysis report."""
        lines = [
            "# GateOrchestra — Error Analysis & Diagnostic Report",
            "",
            f"**Total Evaluated:** `{report.total_evaluated}`  ",
            f"**Total Failures:** `{report.total_errors}`  ",
            f"**Failure Rate:** `{report.error_rate_pct:.1f}%`  ",
            f"**Generated:** `{report.generated_at}`  ",
            "",
            "---",
            "",
            "## 1. Failure Modes by Error Taxonomy",
            "",
            "| Error Category | Count | Percentage of Failures | Description |",
            "|---|---|---|---|",
        ]
        for e_type, count in sorted(report.breakdown.by_error_type.items(), key=lambda x: -x[1]):
            pct = count / report.total_errors * 100.0 if report.total_errors > 0 else 0.0
            desc = ERROR_DESCRIPTIONS.get(ErrorType(e_type), "")
            lines.append(f"| `{e_type}` | {count} | {pct:.1f}% | {desc} |")

        lines.extend(
            [
                "",
                "---",
                "",
                "## 2. Cross-Strata Failure Distribution",
                "",
                "### Failures by Source Dataset",
                "",
                "| Source Dataset | Failures |",
                "|---|---|",
            ]
        )
        for s_key, count in sorted(report.breakdown.by_source.items()):
            lines.append(f"| `{s_key}` | {count} |")

        lines.extend(
            [
                "",
                "### Failures by Reasoning Depth Level",
                "",
                "| Depth Strata | Failures |",
                "|---|---|",
            ]
        )
        for d_key, count in sorted(report.breakdown.by_depth.items()):
            lines.append(f"| `{d_key}` | {count} |")

        lines.extend(
            [
                "",
                "---",
                "",
                "## 3. Sample Annotated Failure Cases",
                "",
                "| Task ID | Source | Depth | Predicted | Ground Truth | Diagnosis |",
                "|---|---|---|---|---|---|",
            ]
        )
        for ann in report.annotations[:10]:
            p_trunc = (
                ann.predicted_answer[:25] + "..."
                if len(ann.predicted_answer) > 25
                else ann.predicted_answer
            )
            gt_trunc = (
                ann.ground_truth[:25] + "..." if len(ann.ground_truth) > 25 else ann.ground_truth
            )
            lines.append(
                f"| `{ann.task_id}` | `{ann.source_dataset}` | {ann.depth_score} | "
                f"`{p_trunc}` | `{gt_trunc}` | `{ann.error_type.value}` |"
            )

        lines.extend(
            [
                "",
                "---",
                "",
                "## 4. Remediation Recommendations",
                "",
            ]
        )
        for rec in report.recommendations:
            lines.extend(
                [
                    "> [!TIP]",
                    f"> **Actionable Insight:** {rec}",
                    "",
                ]
            )

        return "\n".join(lines)

    def export_report(
        self,
        report: ErrorAnalysisReport,
        json_path: Path | str,
        md_path: Path | str | None = None,
    ) -> None:
        """Save report to JSON and Markdown on disk."""
        j_path = Path(json_path)
        j_path.parent.mkdir(parents=True, exist_ok=True)
        with j_path.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved Error JSON report to {j_path}")

        if md_path is not None:
            m_path = Path(md_path)
            m_path.parent.mkdir(parents=True, exist_ok=True)
            with m_path.open("w", encoding="utf-8") as f:
                f.write(self.render_markdown_report(report))
            logger.info(f"Saved Error Markdown report to {m_path}")

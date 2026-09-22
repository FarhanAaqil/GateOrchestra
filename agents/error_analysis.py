"""agents/error_analysis.py
========================
Error Analysis engine for GateOrchestra (Person 2).

Systematically diagnoses failure modes across evaluation traces:
  1. Wrong answers (Exact match / correctness failures)
  2. STOP vs ESCALATE gate routing errors (False STOP vs Failed/Wasteful ESCALATE)
  3. Agent / strategy-specific failures (Probe, ReAct, Debate, Reflexion, Baselines)
  4. Common failure patterns (Premature exit, budget exhaustion, arithmetic slip, hallucination)
  5. Error categories, distributions, and frequency counts
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from agents.agent_analysis import exact_match
from shared.schemas import EvalResult

logger = logging.getLogger(__name__)


class ErrorCategory(str, Enum):
    """Taxonomy of failure modes in GateOrchestra."""

    FALSE_STOP = "false_stop"
    """Gate decided STOP, but single-agent probe was incorrect (missed escalation)."""

    FALSE_ESCALATE_FAILED = "false_escalate_failed"
    """Gate decided ESCALATE, spent extra tokens, but MAS still failed to produce correct answer."""

    FALSE_ESCALATE_WASTE = "false_escalate_waste"
    """Gate escalated unnecessarily when cheap probe was already correct."""

    MAS_OVERTHINKING = "mas_overthinking"
    """Multi-agent communication degraded an initially correct answer into an incorrect one."""

    BUDGET_EXHAUSTION = "budget_exhaustion"
    """Token budget cap exhausted before agent could conclude with a final answer."""

    EMPTY_OR_UNPARSED = "empty_or_unparsed"
    """Answer was empty, 'unable to determine', 'i don't know', or unparseable."""

    ARITHMETIC_ERROR = "arithmetic_error"
    """Numerical calculation discrepancy between predicted and ground truth."""

    REASONING_GAP = "reasoning_gap"
    """Logical reasoning failure or missing intermediate hop."""

    OTHER_INCORRECT = "other_incorrect"
    """General incorrect answer not fitting previous specific patterns."""


@dataclass
class ErrorRecord:
    """Detailed record of a single evaluated error."""

    task_id: str
    method: str
    predicted_answer: str
    ground_truth: str | None
    category: ErrorCategory
    gate_decision: str | None = None  # "STOP" | "ESCALATE" | None
    confidence: float | None = None
    strategy: str | None = None
    tokens_spent: int = 0
    probe_tokens: int | None = None
    mas_tokens: int | None = None
    latency_ms: float | None = None
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        return d


@dataclass
class GateErrorBreakdown:
    """Breakdown of routing decisions and gating classification errors."""

    total_decisions: int = 0
    total_stop: int = 0
    total_escalate: int = 0
    correct_stop: int = 0
    false_stop: int = 0  # STOP chosen, but probe was wrong
    correct_escalate: int = 0  # ESCALATE chosen, and final answer was correct
    failed_escalate: int = 0  # ESCALATE chosen, but answer was wrong
    false_stop_rate: float = 0.0  # false_stop / total_stop
    failed_escalate_rate: float = 0.0  # failed_escalate / total_escalate

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentErrorBreakdown:
    """Failure statistics for a specific agent or baseline method."""

    agent_or_method: str
    total_evaluations: int = 0
    error_count: int = 0
    error_rate: float = 0.0  # Percentage [0.0, 100.0]
    category_counts: dict[str, int] = field(default_factory=dict)
    avg_tokens_on_error: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ErrorAnalysisReport:
    """Comprehensive Error Analysis Report."""

    total_tasks_evaluated: int = 0
    total_errors: int = 0
    overall_error_rate: float = 0.0  # Percentage
    gate_errors: GateErrorBreakdown = field(default_factory=GateErrorBreakdown)
    agent_breakdowns: dict[str, AgentErrorBreakdown] = field(default_factory=dict)
    category_counts: dict[str, int] = field(default_factory=dict)
    category_percentages: dict[str, float] = field(default_factory=dict)
    error_records: list[ErrorRecord] = field(default_factory=list)
    source_info: str = "evaluation_records"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_info": self.source_info,
            "total_tasks_evaluated": self.total_tasks_evaluated,
            "total_errors": self.total_errors,
            "overall_error_rate": self.overall_error_rate,
            "gate_errors": self.gate_errors.to_dict(),
            "agent_breakdowns": {k: v.to_dict() for k, v in self.agent_breakdowns.items()},
            "category_counts": self.category_counts,
            "category_percentages": self.category_percentages,
            "error_samples_count": len(self.error_records),
            "error_records": [e.to_dict() for e in self.error_records],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        lines = [
            "# GateOrchestra -- Error Analysis Report",
            "",
            f"- **Source:** `{self.source_info}`",
            f"- **Evaluations Analyzed:** {self.total_tasks_evaluated}",
            f"- **Total Errors Identified:** {self.total_errors} ({self.overall_error_rate:.1f}% error rate)",
            "",
            "## 1. Gate Routing Error Breakdown (STOP vs ESCALATE)",
            "",
            "| Metric | Count | Rate | Interpretation |",
            "|:-------|:-----:|:----:|:---------------|",
            f"| **Total Gated Decisions** | {self.gate_errors.total_decisions} | 100.0% | Decisions inspected |",
            f"| **Correct STOP** | {self.gate_errors.correct_stop} | "
            f"{((self.gate_errors.correct_stop / self.gate_errors.total_stop * 100.0) if self.gate_errors.total_stop else 0.0):.1f}% of STOP | Probe was correct; tokens saved |",
            f"| **False STOP (Under-routing)** | {self.gate_errors.false_stop} | "
            f"{self.gate_errors.false_stop_rate:.1f}% of STOP | Premature exit; missed escalation |",
            f"| **Correct ESCALATE** | {self.gate_errors.correct_escalate} | "
            f"{((self.gate_errors.correct_escalate / self.gate_errors.total_escalate * 100.0) if self.gate_errors.total_escalate else 0.0):.1f}% of ESCALATE | MAS escalation successful |",
            f"| **Failed ESCALATE** | {self.gate_errors.failed_escalate} | "
            f"{self.gate_errors.failed_escalate_rate:.1f}% of ESCALATE | Escalated, but MAS failed |",
            "",
            "## 2. Agent & Baseline Error Distributions",
            "",
            "| Method / Agent | Evaluations | Errors | Error Rate | Avg Tokens on Error | Dominant Failure Mode |",
            "|:---------------|:-----------:|:------:|:----------:|:-------------------:|:----------------------|",
        ]

        for method, b in sorted(self.agent_breakdowns.items()):
            dominant = (
                max(b.category_counts.items(), key=lambda x: x[1])[0]
                if b.category_counts
                else "None"
            )
            lines.append(
                f"| **{method}** | {b.total_evaluations} | {b.error_count} | {b.error_rate:.1f}% | "
                f"{b.avg_tokens_on_error:.1f} | `{dominant}` |"
            )

        lines.extend(
            [
                "",
                "## 3. Error Categories & Failure Patterns",
                "",
                "| Error Category | Count | Percentage | Description |",
                "|:---------------|:-----:|:----------:|:------------|",
            ]
        )

        descriptions = {
            ErrorCategory.FALSE_STOP.value: "Gate stopped early on query that probe answered incorrectly",
            ErrorCategory.FALSE_ESCALATE_FAILED.value: "Gate escalated to MAS, but multi-agent answer still failed",
            ErrorCategory.FALSE_ESCALATE_WASTE.value: "Unnecessary escalation on simple query already solved by probe",
            ErrorCategory.MAS_OVERTHINKING.value: "Multi-agent interaction degraded a valid candidate answer",
            ErrorCategory.BUDGET_EXHAUSTION.value: "Token budget cap hit before conclusion was reached",
            ErrorCategory.EMPTY_OR_UNPARSED.value: "Empty string, 'unable to determine', or unparsed syntax",
            ErrorCategory.ARITHMETIC_ERROR.value: "Numerical or calculation error in final answer",
            ErrorCategory.REASONING_GAP.value: "Logical deduction failure or missing multi-hop connection",
            ErrorCategory.OTHER_INCORRECT.value: "Other mismatched or erroneous responses",
        }

        for cat, cnt in sorted(self.category_counts.items(), key=lambda x: x[1], reverse=True):
            pct = self.category_percentages.get(cat, 0.0)
            desc = descriptions.get(cat, "Generic failure")
            lines.append(f"| **`{cat}`** | {cnt} | {pct:.1f}% | {desc} |")

        if self.error_records:
            lines.extend(
                [
                    "",
                    "## 4. Sample Diagnostic Error Traces (Top 5)",
                    "",
                    "| Task ID | Method | Decision | Predicted Answer | Ground Truth | Category | Explanation |",
                    "|:--------|:-------|:--------:|:-----------------|:-------------|:---------|:------------|",
                ]
            )
            for err in self.error_records[:5]:
                pred_short = (
                    (err.predicted_answer[:30] + "...")
                    if len(err.predicted_answer) > 30
                    else err.predicted_answer
                )
                gt_short = (
                    (str(err.ground_truth)[:30] + "...")
                    if err.ground_truth and len(str(err.ground_truth)) > 30
                    else str(err.ground_truth)
                )
                dec = err.gate_decision or "N/A"
                lines.append(
                    f"| `{err.task_id}` | {err.method} | {dec} | {pred_short} | "
                    f"{gt_short} | `{err.category.value}` | {err.explanation} |"
                )

        lines.append("")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Classification & Diagnostic Engine
# ─────────────────────────────────────────────────────────────────────────────


def _has_digits(text: str) -> bool:
    """Check if string contains numerical digits."""
    return any(c.isdigit() for c in text)


def _is_unparsed_or_empty(text: str) -> bool:
    """Check if output indicates inability to answer or parsing failure."""
    if not text or not text.strip():
        return True
    t_lower = text.lower().strip()
    patterns = [
        "unable to determine",
        "i don't know",
        "i do not know",
        "cannot determine",
        "unknown",
        "not enough information",
        "no answer",
    ]
    return any(p in t_lower for p in patterns)


def classify_error(
    predicted: str,
    ground_truth: str | None,
    gate_decision: str | None = None,
    confidence: float | None = None,
    tokens_spent: int = 0,
    budget_cap: int | None = None,
    method: str = "Unknown",
) -> tuple[ErrorCategory, str]:
    """Classify the failure mode of an incorrect answer.

    Returns:
        tuple of (ErrorCategory, diagnostic_explanation_string).
    """
    pred_clean = predicted.strip() if predicted else ""

    # 1. Gate Routing Failures (Priority diagnostic for GateOrchestra)
    if gate_decision == "STOP":
        conf_str = f" (confidence: {confidence:.2f})" if confidence is not None else ""
        return (
            ErrorCategory.FALSE_STOP,
            f"Gate issued STOP{conf_str}, but probe produced an incorrect answer.",
        )

    if gate_decision == "ESCALATE":
        if budget_cap and tokens_spent >= budget_cap:
            return (
                ErrorCategory.BUDGET_EXHAUSTION,
                f"ESCALATE consumed full budget cap ({tokens_spent}/{budget_cap} tokens) without finding correct answer.",
            )
        return (
            ErrorCategory.FALSE_ESCALATE_FAILED,
            "Gate correctly ESCALATED, but multi-agent orchestrator failed to produce ground truth.",
        )

    # 2. Token Budget Exhaustion on un-gated MAS
    if budget_cap and tokens_spent >= budget_cap:
        return (
            ErrorCategory.BUDGET_EXHAUSTION,
            f"Tokens spent ({tokens_spent}) reached budget ceiling ({budget_cap}).",
        )

    # 3. Unparsed or Defeatist Responses
    if _is_unparsed_or_empty(pred_clean):
        return (
            ErrorCategory.EMPTY_OR_UNPARSED,
            f"Answer was empty or defeatist: '{pred_clean}'.",
        )

    # 4. Arithmetic discrepancy
    if ground_truth and _has_digits(ground_truth) and _has_digits(pred_clean):
        return (
            ErrorCategory.ARITHMETIC_ERROR,
            f"Numerical mismatch: predicted '{pred_clean}' vs ground truth '{ground_truth}'.",
        )

    # 5. Multi-hop / reasoning gap
    if ground_truth and len(ground_truth.split()) > 2 and len(pred_clean.split()) > 2:
        return (
            ErrorCategory.REASONING_GAP,
            "Syntactic or reasoning gap between multi-word answer and ground truth.",
        )

    # 6. Fallback
    return (
        ErrorCategory.OTHER_INCORRECT,
        f"Mismatched prediction for method {method}.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Analysis Entrypoints
# ─────────────────────────────────────────────────────────────────────────────


def analyze_errors_from_records(
    records: Sequence[dict[str, Any] | EvalResult],
    ground_truths: dict[str, str] | None = None,
    source_name: str = "eval_records",
) -> ErrorAnalysisReport:
    """Analyze errors from an in-memory sequence of evaluation results or records.

    Args:
        records: Sequence of EvalResult objects or dict representations.
        ground_truths: Optional mapping of task_id to ground truth answer.
        source_name: Data source identifier for reporting.

    Returns:
        ErrorAnalysisReport containing full error diagnostics.
    """
    total_evaluations = len(records)
    error_records: list[ErrorRecord] = []

    # Counters for gate breakdown
    gate_total = 0
    gate_stop = 0
    gate_escalate = 0
    correct_stop = 0
    false_stop = 0
    correct_escalate = 0
    failed_escalate = 0

    # Per-method tracking
    method_evals: dict[str, int] = Counter()
    method_errors: dict[str, int] = Counter()
    method_tokens_on_error: dict[str, list[int]] = {}
    method_cat_counts: dict[str, Counter[str]] = {}

    for item in records:
        d = item.model_dump() if isinstance(item, EvalResult) else item
        task_id = str(d.get("task_id", "unknown"))
        method = str(d.get("method", "Unknown"))
        pred = str(d.get("predicted_answer", ""))
        tokens_spent = int(d.get("tokens_spent", 0))
        latency_ms = float(d["latency_ms"]) if d.get("latency_ms") is not None else None
        probe_tokens = int(d["probe_tokens"]) if d.get("probe_tokens") is not None else None
        mas_tokens = int(d["mas_tokens"]) if d.get("mas_tokens") is not None else None
        strategy = d.get("mas_strategy")

        # Extract gate decision details if present
        gate_dec_val: str | None = None
        conf: float | None = None
        budget_cap: int | None = None

        if "gate_decision" in d and isinstance(d["gate_decision"], dict):
            gate_dec_val = d["gate_decision"].get("decision")
            conf = d["gate_decision"].get("confidence")
            budget_cap = d["gate_decision"].get("token_budget_cap")

        # Resolve ground truth
        gt = d.get("ground_truth")
        if gt is None and ground_truths and task_id in ground_truths:
            gt = ground_truths[task_id]

        # Determine correctness
        is_corr = d.get("is_correct")
        if is_corr is None and gt is not None:
            is_corr = exact_match(pred, str(gt))

        method_evals[method] += 1

        # Track gate statistics
        if gate_dec_val:
            gate_total += 1
            if gate_dec_val == "STOP":
                gate_stop += 1
                if is_corr is True:
                    correct_stop += 1
                elif is_corr is False:
                    false_stop += 1
            elif gate_dec_val == "ESCALATE":
                gate_escalate += 1
                if is_corr is True:
                    correct_escalate += 1
                elif is_corr is False:
                    failed_escalate += 1

        # If answer is wrong, diagnose and classify
        if is_corr is False:
            method_errors[method] += 1
            method_tokens_on_error.setdefault(method, []).append(tokens_spent)

            category, explanation = classify_error(
                predicted=pred,
                ground_truth=str(gt) if gt is not None else None,
                gate_decision=gate_dec_val,
                confidence=conf,
                tokens_spent=tokens_spent,
                budget_cap=budget_cap,
                method=method,
            )

            method_cat_counts.setdefault(method, Counter())[category.value] += 1

            error_records.append(
                ErrorRecord(
                    task_id=task_id,
                    method=method,
                    predicted_answer=pred,
                    ground_truth=str(gt) if gt is not None else None,
                    category=category,
                    gate_decision=gate_dec_val,
                    confidence=conf,
                    strategy=strategy,
                    tokens_spent=tokens_spent,
                    probe_tokens=probe_tokens,
                    mas_tokens=mas_tokens,
                    latency_ms=latency_ms,
                    explanation=explanation,
                )
            )

    # Compute overall category counts & percentages
    all_categories = Counter(e.category.value for e in error_records)
    total_errors = len(error_records)
    cat_percentages = (
        {cat: round((cnt / total_errors) * 100.0, 2) for cat, cnt in all_categories.items()}
        if total_errors > 0
        else {}
    )

    # Compute gate error metrics
    false_stop_rate = round((false_stop / gate_stop * 100.0), 2) if gate_stop > 0 else 0.0
    failed_escalate_rate = (
        round((failed_escalate / gate_escalate * 100.0), 2) if gate_escalate > 0 else 0.0
    )

    gate_breakdown = GateErrorBreakdown(
        total_decisions=gate_total,
        total_stop=gate_stop,
        total_escalate=gate_escalate,
        correct_stop=correct_stop,
        false_stop=false_stop,
        correct_escalate=correct_escalate,
        failed_escalate=failed_escalate,
        false_stop_rate=false_stop_rate,
        failed_escalate_rate=failed_escalate_rate,
    )

    # Compute per-agent breakdown
    agent_breakdowns: dict[str, AgentErrorBreakdown] = {}
    for method, n_eval in method_evals.items():
        n_err = method_errors.get(method, 0)
        err_rate = round((n_err / n_eval * 100.0), 2) if n_eval > 0 else 0.0
        tok_list = method_tokens_on_error.get(method, [])
        avg_tok = round(sum(tok_list) / len(tok_list), 2) if tok_list else 0.0
        cats = dict(method_cat_counts.get(method, Counter()))

        agent_breakdowns[method] = AgentErrorBreakdown(
            agent_or_method=method,
            total_evaluations=n_eval,
            error_count=n_err,
            error_rate=err_rate,
            category_counts=cats,
            avg_tokens_on_error=avg_tok,
        )

    overall_err_rate = (
        round((total_errors / total_evaluations * 100.0), 2) if total_evaluations > 0 else 0.0
    )

    return ErrorAnalysisReport(
        total_tasks_evaluated=total_evaluations,
        total_errors=total_errors,
        overall_error_rate=overall_err_rate,
        gate_errors=gate_breakdown,
        agent_breakdowns=agent_breakdowns,
        category_counts=dict(all_categories),
        category_percentages=cat_percentages,
        error_records=error_records,
        source_info=source_name,
    )


def analyze_errors_from_file(
    file_path: str | Path,
    ground_truths: dict[str, str] | None = None,
) -> ErrorAnalysisReport:
    """Load evaluation records from a JSON or JSONL file and run full error analysis.

    Args:
        file_path: Path to evaluation results file.
        ground_truths: Optional mapping of task_id to ground truth answer.

    Returns:
        ErrorAnalysisReport.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation results file not found: {path}")

    records: list[dict[str, Any]] = []
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict) and "records" in data:
                records = data["records"]
            elif isinstance(data, dict) and "results" in data:
                records = data["results"]
            else:
                records = [data]

    return analyze_errors_from_records(records, ground_truths=ground_truths, source_name=str(path))

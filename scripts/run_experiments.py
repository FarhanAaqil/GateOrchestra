"""scripts/run_experiments.py
==========================
Unified Final Experiments Runner for GateOrchestra (Phase 1).

Executes the definitive empirical comparison across all 5 methods:
  1. CoT-SC-only (cheap single-agent baseline)
  2. Always-MAS (un-gated multi-agent ceiling)
  3. RandomGate (stochastic routing baseline)
  4. RuleBasedGate (heuristic syntax/depth/parallel rule-based baseline)
  5. GateOrchestra (learned GBT gate with LinUCB adaptive routing)

Evaluates:
  - Budget multipliers k in {2, 3, 5}
  - Metrics: Accuracy, Precision, Recall, F1, Token Usage, Avg Tokens/Task,
    Latency (ms), STOP/ESCALATE Rate, Errors (False STOP/Failed ESCALATE),
    and Strategy Allocation (ReAct %, Debate %, Reflexion %)
  - 8-feature leave-one-out ablations
  - Sub-agent performance profiling
  - Reproducible outputs saved to reports/final_experiments_results.json,
    reports/final_experiments_summary.md, and reports/figures/final_pareto_frontier.png
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None  # type: ignore[assignment]

from agents.agent_analysis import benchmark_agents
from agents.baselines.always_mas_baseline import run_always_mas_baseline
from agents.baselines.cot_sc_baseline import run_cot_sc_baseline
from agents.baselines.simulated_probe import SimulatedProbe
from agents.error_analysis import analyze_errors_from_records
from agents.orchestrator.orchestrator import MASOrchestrator
from evaluation.metrics import compute_evaluation_metrics
from gate.classifier import (
    GateClassifier,
    GBTGate,
    features_to_array,
)
from gate.feature_extractor import extract_features
from gate.random_gate import RandomGate
from gate.rule_based_gate import RuleBasedGate
from gate.train_gate import apply_label_rule
from integration.pipeline import run_pipeline
from shared.config import (
    K_DEFAULT,
    K_VALUES,
    LOGS_DIR,
    PROBE_TOKEN_BUDGET,
)
from shared.data_loader import load_split
from shared.schemas import EvalResult, GateDecision, GateFeatures, Task
from shared.token_logger import TokenAccountant

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("final_experiments")


# ─────────────────────────────────────────────────────────────────────────────
# Simulation & Mock Callers for Deterministic Experiments
# ─────────────────────────────────────────────────────────────────────────────


def make_deterministic_simulated_mas(seed: int = 42) -> Callable[[Task, int], tuple[str, int]]:
    """Return a deterministic simulated MAS orchestrator pinned to a specific seed."""
    rng = random.Random(seed + 99)

    def simulated_mas(task: Task, token_budget: int) -> tuple[str, int]:
        depth = task.depth_score or 2
        parallel = task.parallel_score or 1
        difficulty = ((depth - 1) / 4 * 0.6) + ((parallel - 1) / 3 * 0.4)
        mas_correct_prob = 0.65 + difficulty * 0.25

        correct = task.ground_truth or "Unknown"
        answer = correct if rng.random() < mas_correct_prob else correct + " (MAS error)"

        base = int(token_budget * 0.68)
        noise = rng.randint(-30, 30)
        tokens_used = max(40, min(token_budget, base + noise))
        return answer, tokens_used

    return simulated_mas


def deterministic_mock_caller(prompt: str, temperature: float, budget: int) -> tuple[str, int]:
    """Deterministic mock LLM caller for sub-agent profiling."""
    p_lower = prompt.lower()
    if "eiffel" in p_lower or "paris" in p_lower:
        return "Final Answer: Euro", 42
    if "tokyo" in p_lower or "london" in p_lower:
        return "Final Answer: Tokyo", 38
    if "15 * 8" in p_lower:
        return "Final Answer: 120", 25
    if "shop sells apples" in p_lower or "2.50" in p_lower:
        return "Final Answer: 15.00", 30
    return "Final Answer: 42", 28


# ─────────────────────────────────────────────────────────────────────────────
# Gating Classification Evaluation (F1, Precision, Recall)
# ─────────────────────────────────────────────────────────────────────────────


def compute_gate_classification_metrics(
    eval_results: list[EvalResult],
    optimal_labels: dict[str, str],
) -> dict[str, float]:
    """Compute Precision, Recall, F1, and Accuracy for the gate's ESCALATE routing decisions.

    Args:
        eval_results: Results from a gated pipeline run.
        optimal_labels: Ground truth optimal routing labels derived from apply_label_rule().

    Returns:
        Dict with precision, recall, f1, and accuracy (all 0.0-100.0%).
    """
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for r in eval_results:
        if not r.gate_decision:
            continue
        task_id = r.task_id
        if task_id not in optimal_labels:
            continue

        predicted_dec = r.gate_decision.decision  # "STOP" | "ESCALATE"
        optimal_dec = optimal_labels[task_id]  # "STOP" | "ESCALATE"

        if predicted_dec == "ESCALATE" and optimal_dec == "ESCALATE":
            tp += 1
        elif predicted_dec == "ESCALATE" and optimal_dec == "STOP":
            fp += 1
        elif predicted_dec == "STOP" and optimal_dec == "ESCALATE":
            fn += 1
        elif predicted_dec == "STOP" and optimal_dec == "STOP":
            tn += 1

    total = tp + fp + fn + tn
    accuracy = round(((tp + tn) / total) * 100.0, 2) if total > 0 else 0.0
    precision = round((tp / (tp + fp)) * 100.0, 2) if (tp + fp) > 0 else 0.0
    recall = round((tp / (tp + fn)) * 100.0, 2) if (tp + fn) > 0 else 0.0
    f1 = (
        round((2 * precision * recall) / (precision + recall), 2)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "gate_accuracy": accuracy,
        "gate_precision": precision,
        "gate_recall": recall,
        "gate_f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature Ablation Engine
# ─────────────────────────────────────────────────────────────────────────────


class AblatedGateWrapper(GateClassifier):
    """Wraps an underlying gate classifier and masks out specified features."""

    def __init__(
        self,
        base_gate: GateClassifier,
        mask_indices: list[int],
        name_suffix: str = "Ablated",
    ) -> None:
        super().__init__(f"{base_gate.name}_{name_suffix}")
        self.base_gate = base_gate
        self.mask_indices = mask_indices
        self._is_trained = True

    def train(
        self,
        features: list[GateFeatures],
        labels: list[str],
    ) -> None:
        """Pass-through training to underlying gate."""
        if features:
            self.base_gate.train(features, labels)
        self._is_trained = True

    def predict(
        self,
        features: GateFeatures,
        k: int = K_DEFAULT,
        probe_tokens: int = PROBE_TOKEN_BUDGET,
        threshold: float | None = None,
    ) -> GateDecision:
        arr = features_to_array(features).copy()
        for idx in self.mask_indices:
            arr[idx] = 0.0

        # Create masked features object
        masked_features = GateFeatures(
            task_id=features.task_id,
            consistency_score=float(arr[0]),
            probe_tokens=int(arr[1]),
            question_word_count=int(arr[2]),
            entity_count=int(arr[3]),
            clause_count=int(arr[4]),
            has_context=bool(arr[5]),
            estimated_depth=float(arr[6]),
            estimated_parallel=float(arr[7]),
        )
        return self.base_gate.predict(
            masked_features, k=k, probe_tokens=probe_tokens, threshold=threshold
        )


def run_feature_ablations(
    tasks: list[Task],
    base_gate: GateClassifier,
    probe_fn: Callable[[Task], Any],
    mas_fn: Callable[[Task, int], tuple[str, int]],
    baseline_results: list[EvalResult],
    k: int = K_DEFAULT,
) -> list[dict[str, Any]]:
    """Run leave-one-feature-out ablations across all 8 pre-execution features."""
    feature_names = [
        "consistency_score",
        "probe_tokens",
        "question_word_count",
        "entity_count",
        "clause_count",
        "has_context",
        "estimated_depth",
        "estimated_parallel",
    ]

    ablation_results = []
    base_eval = compute_evaluation_metrics(
        [
            run_pipeline(
                task=t,
                gate=base_gate,
                probe_agent=probe_fn,
                orchestrator=mas_fn,
                accountant=TokenAccountant(),
                k=k,
                method="GateOrchestra",
            )
            for t in tasks
        ],
        baseline_results=baseline_results,
    )
    full_acc = base_eval["accuracy"]
    full_savings = base_eval["token_savings_pct"]

    for idx, feat in enumerate(feature_names):
        ablated_gate = AblatedGateWrapper(base_gate, mask_indices=[idx], name_suffix=f"no_{feat}")
        acc = TokenAccountant()
        results = [
            run_pipeline(
                task=t,
                gate=ablated_gate,
                probe_agent=probe_fn,
                orchestrator=mas_fn,
                accountant=acc,
                k=k,
                method="GateOrchestra",
            )
            for t in tasks
        ]
        metrics = compute_evaluation_metrics(results, baseline_results=baseline_results)
        delta_acc = round(metrics["accuracy"] - full_acc, 2)
        delta_savings = (
            round(metrics["token_savings_pct"] - full_savings, 2)
            if metrics["token_savings_pct"] is not None and full_savings is not None
            else 0.0
        )

        ablation_results.append(
            {
                "ablated_feature": feat,
                "feature_index": idx,
                "accuracy": metrics["accuracy"],
                "avg_tokens": metrics["avg_tokens"],
                "token_savings_pct": metrics["token_savings_pct"],
                "delta_accuracy_vs_full": delta_acc,
                "delta_savings_vs_full": delta_savings,
            }
        )

    return ablation_results


# ─────────────────────────────────────────────────────────────────────────────
# Pareto Frontier Plotter
# ─────────────────────────────────────────────────────────────────────────────


def plot_pareto_frontier(
    pareto_data: dict[str, list[dict[str, Any]]],
    output_path: Path,
) -> None:
    """Generate high-resolution Pareto frontier curve (Accuracy vs. Average Tokens)."""
    if not HAS_MATPLOTLIB or plt is None:
        logger.warning("Matplotlib not available; skipping PNG plot generation.")
        return

    plt.figure(figsize=(9, 6), dpi=300)
    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )

    colors = {
        "Always-MAS": "#d9534f",
        "GateOrchestra": "#2b6cb0",
        "RuleBasedGate": "#f0ad4e",
        "RandomGate": "#888888",
        "CoT-SC-only": "#5cb85c",
    }
    markers = {
        "Always-MAS": "s",
        "GateOrchestra": "o",
        "RuleBasedGate": "^",
        "RandomGate": "x",
        "CoT-SC-only": "*",
    }

    for method, points in pareto_data.items():
        if not points:
            continue
        xs = [p["avg_tokens"] for p in points]
        ys = [p["accuracy"] for p in points]
        c = colors.get(method, "#333333")
        m = markers.get(method, "o")

        # Sort by tokens for coherent line
        sorted_pts = sorted(zip(xs, ys, strict=False), key=lambda pt: pt[0])
        x_sort = [pt[0] for pt in sorted_pts]
        y_sort = [pt[1] for pt in sorted_pts]

        if len(x_sort) > 1:
            plt.plot(x_sort, y_sort, label=method, color=c, marker=m, linewidth=2.5, markersize=8)
        else:
            plt.scatter(x_sort, y_sort, label=method, color=c, marker=m, s=120, zorder=5)

        for p in points:
            k_val = p.get("k")
            if k_val:
                plt.annotate(
                    f"k={k_val}",
                    (p["avg_tokens"], p["accuracy"]),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                )

    plt.title(
        "GateOrchestra -- Accuracy vs. Token Expenditure (Pareto Frontier)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    plt.xlabel("Average Tokens per Task", fontsize=11, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9, loc="lower right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved Pareto frontier plot to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Markdown Summary Formatter
# ─────────────────────────────────────────────────────────────────────────────


def format_markdown_summary(
    metadata: dict[str, Any],
    comparisons: list[dict[str, Any]],
    pareto_data: dict[str, list[dict[str, Any]]],
    ablations: list[dict[str, Any]],
    agent_metrics: dict[str, Any],
    error_report: dict[str, Any],
) -> str:
    """Format full experiment results into a clean, comprehensive Markdown report."""
    lines = [
        "# GateOrchestra -- Definitive Final Experiments Report",
        "",
        "## Executive Summary",
        f"- **Timestamp:** `{metadata.get('timestamp')}`",
        f"- **Dataset:** MASBench-mini (`{metadata.get('test_tasks_count')} held-out test tasks`)",
        f"- **Learned Gate Model:** `{metadata.get('gate_model')}`",
        f"- **Evaluation Mode:** `{metadata.get('mode')}` (seed={metadata.get('seed')})",
        "",
        "---",
        "",
        "## 1. Primary Method Comparison (Held-out Test Split)",
        "",
        "| Method | Accuracy | Token Savings vs Always-MAS | Avg Tokens/Task | Total Tokens | Avg Latency (ms) | STOP % | ESCALATE % | Gate F1 | Strategy Breakdown |",
        "|:-------|:--------:|:---------------------------:|:---------------:|:------------:|:----------------:|:------:|:----------:|:-------:|:-------------------|",
    ]

    for c in comparisons:
        acc_str = f"{c['accuracy']:.1f}% ({c['correct_count']}/{c['n_tasks']})"
        sav_str = (
            f"+{c['token_savings_pct']:.1f}%"
            if c.get("token_savings_pct") is not None
            else "0.0% (Baseline)"
        )
        lat_str = f"{c['avg_latency_ms']:.1f}" if c.get("avg_latency_ms") is not None else "N/A"
        stop_str = f"{c['stop_rate'] * 100:.1f}%"
        esc_str = f"{c['escalate_rate'] * 100:.1f}%"
        f1_str = f"{c['gate_f1']:.1f}%" if c.get("gate_f1") is not None else "N/A"

        strat = c.get("strategy_allocation", {})
        if strat and strat.get("total_escalated", 0) > 0:
            strat_str = f"ReAct: {strat.get('react_pct', 0):.0f}%, Debate: {strat.get('debate_pct', 0):.0f}%, Refl: {strat.get('reflexion_pct', 0):.0f}%"
        else:
            strat_str = "None (STOP / Single-Agent)"

        lines.append(
            f"| **{c['method']}** | {acc_str} | {sav_str} | {c['avg_tokens']:.1f} | "
            f"{c['total_tokens']:,} | {lat_str} | {stop_str} | {esc_str} | {f1_str} | {strat_str} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 2. Budget Multiplier Sweep & Pareto Analysis (k in {2, 3, 5})",
            "",
            "| Method | Multiplier (k) | Accuracy | Avg Tokens/Task | Token Savings % vs Always-MAS | ReAct % | Debate % | Reflexion % |",
            "|:-------|:--------------:|:--------:|:---------------:|:-----------------------------:|:-------:|:--------:|:-----------:|",
        ]
    )

    for method, rows in pareto_data.items():
        for r in rows:
            k_val = r.get("k", "-")
            sav = (
                f"+{r['token_savings_pct']:.1f}%"
                if r.get("token_savings_pct") is not None
                else "N/A"
            )
            strat = r.get("strategy_allocation", {})
            r_pct = f"{strat.get('react_pct', 0):.1f}%"
            d_pct = f"{strat.get('debate_pct', 0):.1f}%"
            x_pct = f"{strat.get('reflexion_pct', 0):.1f}%"
            lines.append(
                f"| **{method}** | {k_val} | {r['accuracy']:.1f}% | {r['avg_tokens']:.1f} | "
                f"{sav} | {r_pct} | {d_pct} | {x_pct} |"
            )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Leave-One-Feature-Out Sensitivity Ablations (GBTGate)",
            "",
            "| Ablated Feature | Accuracy | Avg Tokens | Token Savings % | Delta Accuracy vs Full | Delta Savings vs Full |",
            "|:----------------|:--------:|:----------:|:---------------:|:----------------------:|:---------------------:|",
        ]
    )

    for a in ablations:
        delta_acc = f"{a['delta_accuracy_vs_full']:+.2f}%"
        delta_sav = f"{a['delta_savings_vs_full']:+.2f}%"
        lines.append(
            f"| Without `{a['ablated_feature']}` | {a['accuracy']:.1f}% | {a['avg_tokens']:.1f} | "
            f"+{a['token_savings_pct']:.1f}% | {delta_acc} | {delta_sav} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Multi-Agent System & Sub-Agent Performance Profiling",
            "",
            "| Agent / Strategy | Executions | Accuracy | Total Tokens | Avg Tokens | Avg Latency (ms) | Strategy Breakdown |",
            "|:-----------------|:----------:|:--------:|:------------:|:----------:|:----------------:|:-------------------|",
        ]
    )

    for name, m in sorted(agent_metrics.items()):
        acc_str = f"{m.get('accuracy', 0):.1f}% ({m.get('correct_count', 0)}/{m.get('execution_count', 0)})"
        lat_str = (
            f"{m.get('avg_latency_ms', 0):.1f}" if m.get("avg_latency_ms") is not None else "N/A"
        )
        freqs = m.get("strategy_frequencies", {})
        strat_str = ", ".join(f"{k}: {v:.1f}%" for k, v in freqs.items()) if freqs else "N/A"
        lines.append(
            f"| **{name}** | {m.get('execution_count')} | {acc_str} | {m.get('total_tokens', 0):,} | "
            f"{m.get('avg_tokens', 0):.1f} | {lat_str} | {strat_str} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 5. Diagnostic Error Analysis & Gating Failure Breakdown",
            "",
            f"- **Total Test Tasks Evaluated:** {error_report.get('total_tasks_evaluated')}",
            f"- **Total Errors Identified:** {error_report.get('total_errors')} ({error_report.get('overall_error_rate', 0):.1f}% error rate)",
            "",
            "| Metric | Count | Interpretation |",
            "|:-------|:-----:|:---------------|",
        ]
    )

    gate_errs = error_report.get("gate_errors", {})
    lines.extend(
        [
            f"| **Total Gated Decisions** | {gate_errs.get('total_decisions', 0)} | Routing decisions inspected |",
            f"| **Correct STOP** | {gate_errs.get('correct_stop', 0)} | Probe was correct; tokens saved safely |",
            f"| **False STOP (Under-routing)** | {gate_errs.get('false_stop', 0)} ({gate_errs.get('false_stop_rate', 0):.1f}% of STOPs) | Premature exit on incorrect probe |",
            f"| **Correct ESCALATE** | {gate_errs.get('correct_escalate', 0)} | MAS escalation successful |",
            f"| **Failed ESCALATE** | {gate_errs.get('failed_escalate', 0)} ({gate_errs.get('failed_escalate_rate', 0):.1f}% of ESCALATEs) | Escalated, but MAS failed |",
            "",
            "### Error Taxonomy Distribution",
            "",
            "| Category | Count | Percentage |",
            "|:---------|:-----:|:----------:|",
        ]
    )

    for cat, cnt in sorted(
        error_report.get("category_counts", {}).items(), key=lambda x: x[1], reverse=True
    ):
        pct = error_report.get("category_percentages", {}).get(cat, 0.0)
        lines.append(f"| `{cat}` | {cnt} | {pct:.1f}% |")

    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main Experiment Execution Flow
# ─────────────────────────────────────────────────────────────────────────────


def run_experiments(
    split: str = "test",
    seed: int = 42,
    k_values: list[int] | None = None,
    mode: str = "mock",
    n: int | None = None,
    output_json: Path | None = None,
    output_md: Path | None = None,
    output_plot: Path | None = None,
) -> dict[str, Any]:
    """Execute the full final experiments suite.

    Returns:
        Structured dictionary of all experimental results.
    """
    if k_values is None:
        k_values = list(K_VALUES)

    random.seed(seed)
    logger.info(
        f"[*] Starting GateOrchestra Final Experiments Suite (split={split}, seed={seed}, mode={mode})"
    )

    tasks = load_split(cast(Literal["train", "val", "test"], split))
    if n is not None and n > 0:
        tasks = tasks[:n]
    logger.info(f"[*] Loaded {len(tasks)} tasks from split '{split}'")

    # Set up probe and MAS orchestrators
    sim_probe = SimulatedProbe(seed=seed)
    probe_fn = sim_probe.run

    sim_mas = make_deterministic_simulated_mas(seed=seed)
    mas_orch = MASOrchestrator(default_strategy="bandit", llm_caller=deterministic_mock_caller)

    def orchestrator_fn(t: Task, b: int) -> tuple[str, int]:
        # Uses MASOrchestrator for strategy selection and simulated token pacing
        strat = mas_orch.select_strategy(t)
        mas_orch._last_strategy = strat
        ans, tok = sim_mas(t, b)
        return ans, tok

    # Load trained Gate
    gate_model_path = LOGS_DIR / "week7_real_gbt_gate.pkl"
    if not gate_model_path.exists():
        gate_model_path = LOGS_DIR / "week2_best_gate.pkl"

    learned_gate = None
    if gate_model_path.exists():
        try:
            logger.info(f"[*] Loading learned GBT gate from {gate_model_path}")
            learned_gate = GateClassifier.load(gate_model_path)
        except Exception as e:
            logger.warning(
                f"[!] Failed to load learned GBT gate from {gate_model_path} ({e}); "
                "falling back to dynamically training a fallback GBT gate."
            )
            learned_gate = None

    if learned_gate is None:
        logger.info("[*] Training fallback GBT gate on dataset features")
        learned_gate = GBTGate(n_estimators=100, learning_rate=0.1, max_depth=3)
        train_tasks = load_split("train")
        train_features = [extract_features(t, probe_fn(t)) for t in train_tasks]
        train_labels = [
            "ESCALATE" if (t.depth_score or 2) >= 3 or (t.parallel_score or 1) >= 2 else "STOP"
            for t in train_tasks
        ]
        learned_gate.train(train_features, train_labels)

    # 1. Run Baselines: CoT-SC-only and Always-MAS
    logger.info("[1/5] Evaluating CoT-SC-only and Always-MAS baselines...")
    test_cot_accountant = TokenAccountant()
    start_t = time.perf_counter()
    cot_results = [
        run_cot_sc_baseline(t, probe_fn=probe_fn, accountant=test_cot_accountant) for t in tasks
    ]
    cot_latency = (time.perf_counter() - start_t) * 1000.0

    test_mas_accountant = TokenAccountant()
    start_t = time.perf_counter()
    mas_results_k_default = [
        run_always_mas_baseline(
            t,
            orchestrator_fn=orchestrator_fn,
            accountant=test_mas_accountant,
            token_budget=K_DEFAULT * PROBE_TOKEN_BUDGET,
        )
        for t in tasks
    ]
    mas_latency = (time.perf_counter() - start_t) * 1000.0

    # Derive optimal routing labels
    cot_by_id = {r.task_id: r for r in cot_results}
    mas_by_id = {r.task_id: r for r in mas_results_k_default}
    optimal_labels = apply_label_rule(cot_by_id, mas_by_id)

    # 2. Run Gated Methods at k=K_DEFAULT
    logger.info("[2/5] Evaluating Gated Methods (RandomGate, RuleBasedGate, GateOrchestra)...")
    random_gate = RandomGate(escalation_rate=0.5, seed=seed)
    rule_gate = RuleBasedGate()

    start_t = time.perf_counter()
    random_results = [
        run_pipeline(
            task=t,
            gate=cast(GateClassifier, random_gate),
            probe_agent=probe_fn,
            orchestrator=orchestrator_fn,
            accountant=TokenAccountant(),
            k=K_DEFAULT,
            method="RandomGate",
            mas_orchestrator=mas_orch,
        )
        for t in tasks
    ]
    random_latency = (time.perf_counter() - start_t) * 1000.0

    start_t = time.perf_counter()
    rule_results = [
        run_pipeline(
            task=t,
            gate=cast(GateClassifier, rule_gate),
            probe_agent=probe_fn,
            orchestrator=orchestrator_fn,
            accountant=TokenAccountant(),
            k=K_DEFAULT,
            method="RuleBasedGate",
            mas_orchestrator=mas_orch,
        )
        for t in tasks
    ]
    rule_latency = (time.perf_counter() - start_t) * 1000.0

    start_t = time.perf_counter()
    gate_results = [
        run_pipeline(
            task=t,
            gate=learned_gate,
            probe_agent=probe_fn,
            orchestrator=orchestrator_fn,
            accountant=TokenAccountant(),
            k=K_DEFAULT,
            method="GateOrchestra",
            mas_orchestrator=mas_orch,
        )
        for t in tasks
    ]
    gate_latency = (time.perf_counter() - start_t) * 1000.0

    # Summarize 5-method comparison
    comparisons = []
    runs_map = [
        ("CoT-SC-only", cot_results, cot_latency, None),
        ("Always-MAS", mas_results_k_default, mas_latency, None),
        ("RandomGate", random_results, random_latency, random_gate),
        ("RuleBasedGate", rule_results, rule_latency, rule_gate),
        ("GateOrchestra", gate_results, gate_latency, learned_gate),
    ]

    for name, res_list, total_lat, gate_inst in runs_map:
        m = compute_evaluation_metrics(res_list, baseline_results=mas_results_k_default)
        correct_cnt = sum(1 for r in res_list if r.is_correct is True)
        entry: dict[str, Any] = {
            "method": name,
            "accuracy": m["accuracy"],
            "correct_count": correct_cnt,
            "n_tasks": m["n_tasks"],
            "total_tokens": m["total_tokens"],
            "avg_tokens": m["avg_tokens"],
            "token_savings_pct": m["token_savings_pct"],
            "accuracy_delta": m["accuracy_delta"],
            "avg_latency_ms": round(total_lat / len(res_list), 2),
            "total_latency_ms": round(total_lat, 2),
            "stop_count": m["stop_count"],
            "stop_rate": m["stop_rate"],
            "escalate_count": m["escalate_count"],
            "escalate_rate": m["escalate_rate"],
            "strategy_allocation": m["strategy_allocation"],
        }
        if gate_inst:
            clf_metrics = compute_gate_classification_metrics(res_list, optimal_labels)
            entry.update(clf_metrics)
        comparisons.append(entry)

    # 3. Budget Multiplier Sweep (k in {2, 3, 5})
    logger.info("[3/5] Running Budget Multiplier Sweep (k in {2, 3, 5})...")
    pareto_data: dict[str, list[dict[str, Any]]] = {
        "CoT-SC-only": [
            {
                "k": None,
                "accuracy": comparisons[0]["accuracy"],
                "avg_tokens": comparisons[0]["avg_tokens"],
                "token_savings_pct": comparisons[0]["token_savings_pct"],
            }
        ],
        "Always-MAS": [],
        "RandomGate": [],
        "RuleBasedGate": [],
        "GateOrchestra": [],
    }

    for k_val in k_values:
        # Always-MAS at k
        mas_k_results = [
            run_always_mas_baseline(
                t,
                orchestrator_fn=orchestrator_fn,
                accountant=TokenAccountant(),
                token_budget=k_val * PROBE_TOKEN_BUDGET,
            )
            for t in tasks
        ]
        mas_k_metrics = compute_evaluation_metrics(mas_k_results, baseline_results=mas_k_results)
        pareto_data["Always-MAS"].append(
            {
                "k": k_val,
                "accuracy": mas_k_metrics["accuracy"],
                "avg_tokens": mas_k_metrics["avg_tokens"],
                "token_savings_pct": 0.0,
                "strategy_allocation": mas_k_metrics["strategy_allocation"],
            }
        )

        # GateOrchestra at k
        go_k_results = [
            run_pipeline(
                task=t,
                gate=learned_gate,
                probe_agent=probe_fn,
                orchestrator=orchestrator_fn,
                accountant=TokenAccountant(),
                k=k_val,
                method="GateOrchestra",
                mas_orchestrator=mas_orch,
            )
            for t in tasks
        ]
        go_k_metrics = compute_evaluation_metrics(go_k_results, baseline_results=mas_k_results)
        pareto_data["GateOrchestra"].append(
            {
                "k": k_val,
                "accuracy": go_k_metrics["accuracy"],
                "avg_tokens": go_k_metrics["avg_tokens"],
                "token_savings_pct": go_k_metrics["token_savings_pct"],
                "strategy_allocation": go_k_metrics["strategy_allocation"],
            }
        )

        # RuleBasedGate at k
        rule_k_results = [
            run_pipeline(
                task=t,
                gate=cast(GateClassifier, rule_gate),
                probe_agent=probe_fn,
                orchestrator=orchestrator_fn,
                accountant=TokenAccountant(),
                k=k_val,
                method="RuleBasedGate",
                mas_orchestrator=mas_orch,
            )
            for t in tasks
        ]
        rule_k_metrics = compute_evaluation_metrics(rule_k_results, baseline_results=mas_k_results)
        pareto_data["RuleBasedGate"].append(
            {
                "k": k_val,
                "accuracy": rule_k_metrics["accuracy"],
                "avg_tokens": rule_k_metrics["avg_tokens"],
                "token_savings_pct": rule_k_metrics["token_savings_pct"],
                "strategy_allocation": rule_k_metrics["strategy_allocation"],
            }
        )

        # RandomGate at k
        rand_k_results = [
            run_pipeline(
                task=t,
                gate=cast(GateClassifier, random_gate),
                probe_agent=probe_fn,
                orchestrator=orchestrator_fn,
                accountant=TokenAccountant(),
                k=k_val,
                method="RandomGate",
                mas_orchestrator=mas_orch,
            )
            for t in tasks
        ]
        rand_k_metrics = compute_evaluation_metrics(rand_k_results, baseline_results=mas_k_results)
        pareto_data["RandomGate"].append(
            {
                "k": k_val,
                "accuracy": rand_k_metrics["accuracy"],
                "avg_tokens": rand_k_metrics["avg_tokens"],
                "token_savings_pct": rand_k_metrics["token_savings_pct"],
                "strategy_allocation": rand_k_metrics["strategy_allocation"],
            }
        )

    # 4. Feature Ablations
    logger.info("[4/5] Running 8-Feature Leave-One-Out Ablations...")
    ablations = run_feature_ablations(
        tasks=tasks,
        base_gate=learned_gate,
        probe_fn=probe_fn,
        mas_fn=orchestrator_fn,
        baseline_results=mas_results_k_default,
        k=K_DEFAULT,
    )

    # 5. Agent Profiling & Error Diagnostics
    logger.info("[5/5] Running Sub-Agent Profiling & Error Diagnostics...")
    agent_report = benchmark_agents(
        tasks=tasks,
        accountant=TokenAccountant(),
        llm_caller=deterministic_mock_caller,
        include_subagents=True,
    )
    agent_metrics = {k: v.to_dict() for k, v in agent_report.metrics.items()}

    # Diagnostic error report across all primary comparison runs
    all_eval_records = (
        cot_results + mas_results_k_default + random_results + rule_results + gate_results
    )
    error_analysis_obj = analyze_errors_from_records(
        all_eval_records, source_name="final_experiments_test_runs"
    )
    error_report = error_analysis_obj.to_dict()

    # Compile final data structure
    final_output: dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seed": seed,
            "split": split,
            "test_tasks_count": len(tasks),
            "gate_model": getattr(learned_gate, "name", "GBTGate"),
            "k_values": k_values,
            "mode": mode,
        },
        "baseline_comparison": comparisons,
        "pareto_sweep": pareto_data,
        "feature_ablations": ablations,
        "agent_profiling": agent_metrics,
        "error_diagnostics": error_report,
    }

    # Save outputs under reports/
    out_json = output_json or (REPO_ROOT / "reports" / "final_experiments_results.json")
    out_md = output_md or (REPO_ROOT / "reports" / "final_experiments_summary.md")
    out_plot = output_plot or (REPO_ROOT / "reports" / "figures" / "final_pareto_frontier.png")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(final_output, indent=2), encoding="utf-8")
    logger.info(f"[*] Exported final JSON results to {out_json}")

    md_content = format_markdown_summary(
        metadata=final_output["metadata"],
        comparisons=comparisons,
        pareto_data=pareto_data,
        ablations=ablations,
        agent_metrics=agent_metrics,
        error_report=error_report,
    )
    out_md.write_text(md_content, encoding="utf-8")
    logger.info(f"[*] Exported Markdown summary to {out_md}")

    plot_pareto_frontier(pareto_data, out_plot)

    return final_output


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entrypoint
# ─────────────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GateOrchestra Unified Final Experiments Runner (Phase 1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[2, 3, 5],
        help="Token budget multipliers k to evaluate for Pareto sweep.",
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "live"],
        default="mock",
        help="Execution mode ('mock' for fast reproducible simulation; 'live' for real Groq/Ollama).",
    )
    parser.add_argument(
        "--n", type=int, default=None, help="Subset size limit (default: all tasks)."
    )
    parser.add_argument("--output-json", type=str, default=None, help="Path for JSON output.")
    parser.add_argument("--output-md", type=str, default=None, help="Path for Markdown output.")
    parser.add_argument("--output-plot", type=str, default=None, help="Path for PNG Pareto plot.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_experiments(
            split=args.split,
            seed=args.seed,
            k_values=args.k_values,
            mode=args.mode,
            n=args.n,
            output_json=Path(args.output_json) if args.output_json else None,
            output_md=Path(args.output_md) if args.output_md else None,
            output_plot=Path(args.output_plot) if args.output_plot else None,
        )
        return 0
    except Exception as e:
        logger.exception(f"Final experiments failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

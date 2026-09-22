"""
scripts/week6_ablations.py
===========================
Week 6 — Gate Analysis & Ablations Suite.

Comprehensive diagnostics, sensitivity studies, and empirical ablations
for the GateOrchestra learned gating framework:

  1. Gate Failure Taxonomy (--mode taxonomy)
     - Quadrant analysis: True STOP, False STOP, True ESCALATE, False ESCALATE
     - Recoverability of False STOPs (did MAS have the answer?)
     - Semantic error categorization via dataset/error_labels.py ErrorAnalyzer
     - Gate classification precision, recall, F1

  2. Feature Ablations (--mode feature_ablation)
     - Leave-one-feature-out for all 8 features
     - Feature subsets: consistency_only, probe_only, text_only, structure_only
     - Retrains classifier and reports delta in accuracy and token savings

  3. CoT-SC Sample Size Sensitivity (--mode n_sweep)
     - Evaluates N in {3, 5, 7} samples per probe run
     - Measures trade-off between probe cost and downstream gating quality

  4. Pareto Frontier Analysis (--mode pareto)
     - Sweeps token multiplier k in {2, 3, 5} across all 5 gating methods
     - Generates token spend vs. accuracy curves
     - Saves high-res plot to reports/figures/week6_pareto_frontier.png
     - Outputs ASCII representation for terminal viewing

  5. LinUCB Bandit Strategy Breakdown (--mode bandit)
     - Tracks arm selection frequencies across react, debate, reflexion
     - Stratifies routing by depth, parallel complexity, and task type
     - Reports online bandit convergence and reward metrics

  6. Full Suite (--mode all, default)
     - Executes all 5 analyses and exports consolidated JSON and markdown reports.

Person 3 owns this script.
"""

from __future__ import annotations

import argparse
import json
import logging
import random as _random
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from agents.baselines.simulated_probe import SimulatedProbe
from agents.orchestrator.bandit_router import LinUCBRouter
from dataset.error_labels import ErrorAnalyzer, ErrorType
from gate.classifier import (
    GateClassifier,
    make_classifier,
)
from gate.feature_extractor import extract_features
from gate.random_gate import RandomGate
from gate.rule_based_gate import RuleBasedGate
from gate.train_gate import apply_label_rule
from integration.pipeline import run_batch
from shared.config import (
    K_DEFAULT,
    LOGS_DIR,
)
from shared.data_loader import load_split
from shared.schemas import EvalResult, GateDecision, GateFeatures, Task
from shared.token_logger import TokenAccountant

logger = logging.getLogger(__name__)

RESULTS_DIR = LOGS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = REPO_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Simulation & Gating Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_simulated_mas(seed: int) -> Callable[[Task, int], tuple[str, int]]:
    """Return a deterministic simulated MAS orchestrator pinned to a specific seed."""
    _rng = _random.Random(seed + 77)

    def simulated_mas(task: Task, token_budget: int) -> tuple[str, int]:
        depth = task.depth_score or 2
        parallel = task.parallel_score or 1
        difficulty = ((depth - 1) / 4 * 0.6) + ((parallel - 1) / 3 * 0.4)
        mas_correct_prob = 0.60 + difficulty * 0.30

        correct = task.ground_truth or "Unknown"
        answer = correct if _rng.random() < mas_correct_prob else correct + " (MAS wrong)"

        base = int(token_budget * 0.7)
        noise = _rng.randint(-50, 50)
        tokens_used = max(50, min(token_budget, base + noise))
        return answer, tokens_used

    return simulated_mas


def _always_mas_baseline(
    tasks: list[Task], mas_fn: Callable[[Task, int], tuple[str, int]], token_budget: int
) -> list[EvalResult]:
    """Run tasks unconditionally through MAS."""
    results: list[EvalResult] = []
    for task in tasks:
        answer, tokens = mas_fn(task, token_budget)
        is_correct: bool | None = None
        if task.ground_truth:
            is_correct = answer.lower().strip() == (task.ground_truth or "").lower().strip()
        results.append(
            EvalResult(
                task_id=task.task_id,
                method="Always-MAS",
                predicted_answer=answer,
                is_correct=is_correct,
                tokens_spent=tokens,
                mas_tokens=tokens,
            )
        )
    return results


def _cot_sc_baseline(tasks: list[Task], probe: SimulatedProbe) -> list[EvalResult]:
    """Run tasks unconditionally through Probe (CoT-SC) with no escalation."""
    results: list[EvalResult] = []
    for task in tasks:
        p_res = probe.run(task)
        is_correct: bool | None = None
        if task.ground_truth:
            is_correct = p_res.answer.lower().strip() == (task.ground_truth or "").lower().strip()
        results.append(
            EvalResult(
                task_id=task.task_id,
                method="CoT-SC-only",
                predicted_answer=p_res.answer,
                is_correct=is_correct,
                tokens_spent=p_res.tokens_used,
                probe_tokens=p_res.tokens_used,
            )
        )
    return results


def mask_features(f: GateFeatures, dropped_features: set[str]) -> GateFeatures:
    """Mask specified features to zero or default for ablation."""
    return GateFeatures(
        task_id=f.task_id,
        consistency_score=(0.0 if "consistency_score" in dropped_features else f.consistency_score),
        probe_tokens=0 if "probe_tokens" in dropped_features else f.probe_tokens,
        question_word_count=(
            0 if "question_word_count" in dropped_features else f.question_word_count
        ),
        entity_count=0 if "entity_count" in dropped_features else f.entity_count,
        clause_count=0 if "clause_count" in dropped_features else f.clause_count,
        has_context=False if "has_context" in dropped_features else f.has_context,
        estimated_depth=(0 if "estimated_depth" in dropped_features else f.estimated_depth),
        estimated_parallel=(
            0 if "estimated_parallel" in dropped_features else f.estimated_parallel
        ),
    )


class AblatedGateWrapper:
    """Wrapper that masks specified features before feeding to the underlying GateClassifier."""

    def __init__(self, base_gate: GateClassifier, dropped_features: set[str]) -> None:
        self.base_gate = base_gate
        self.dropped_features = dropped_features
        self.name = f"{base_gate.name}[ablated:{'+'.join(sorted(dropped_features)) or 'none'}]"

    def predict(
        self,
        features: GateFeatures,
        k: int,
        probe_tokens: int,
        threshold: float | None = None,
    ) -> GateDecision:
        masked = mask_features(features, self.dropped_features)
        return self.base_gate.predict(masked, k=k, probe_tokens=probe_tokens, threshold=threshold)


def _train_gate_with_ablation(
    train_tasks: list[Task],
    val_tasks: list[Task],
    probe: SimulatedProbe,
    mas_fn: Callable[[Task, int], tuple[str, int]],
    dropped_features: set[str] | None = None,
    classifier_type: str = "logreg",
    k: int = K_DEFAULT,
) -> AblatedGateWrapper:
    """Train a gate classifier with specified features ablated from both train and val."""
    dropped = dropped_features or set()

    # 1. Collect training traces and derive labels
    cot_sc_train = {t.task_id: _cot_sc_baseline([t], probe)[0] for t in train_tasks}
    mas_train = {t.task_id: _always_mas_baseline([t], mas_fn, 1000)[0] for t in train_tasks}
    labels_dict = apply_label_rule(cot_sc_train, mas_train)

    train_task_map = {t.task_id: t for t in train_tasks}
    train_features: list[GateFeatures] = []
    train_labels: list[str] = []

    for task_id, label in labels_dict.items():
        task = train_task_map[task_id]
        p_res = probe.run(task)
        raw_feat = extract_features(task, p_res)
        train_features.append(mask_features(raw_feat, dropped))
        train_labels.append(label)

    # Ensure at least 2 classes present for sklearn fit (e.g. tiny test fixtures)
    if len(set(train_labels)) < 2:
        dummy_feat = GateFeatures(
            task_id="dummy_opposite_class",
            consistency_score=0.1,
            probe_tokens=500,
            question_word_count=50,
            entity_count=10,
            clause_count=5,
            has_context=True,
            estimated_depth=5,
            estimated_parallel=4,
        )
        opposite_label = "ESCALATE" if "STOP" in train_labels else "STOP"
        train_features.append(mask_features(dummy_feat, dropped))
        train_labels.append(opposite_label)

    # 2. Fit base classifier
    base_gate = make_classifier(classifier_type)
    base_gate.train(train_features, train_labels)

    return AblatedGateWrapper(base_gate, dropped)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Gate Failure Taxonomy
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FailureTaxonomyResult:
    """Taxonomy metrics for gate routing outcomes."""

    total_tasks: int
    true_stops: int
    false_stops: int
    true_escalates: int
    false_escalates: int
    recoverable_false_stops: int
    unrecoverable_false_stops: int
    overthinking_false_escalates: int
    successful_escalates: int
    exhausted_escalates: int
    precision: float
    recall: float
    f1: float
    error_type_breakdown: dict[str, int] = field(default_factory=dict)
    detailed_cases: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_gate_taxonomy(
    tasks: list[Task],
    gate: Any,
    probe: SimulatedProbe,
    mas_fn: Callable[[Task, int], tuple[str, int]],
    k: int = K_DEFAULT,
) -> FailureTaxonomyResult:
    """Execute gate failure taxonomy analysis across tasks."""
    analyzer = ErrorAnalyzer()

    true_stops = 0
    false_stops = 0
    true_escalates = 0
    false_escalates = 0
    recoverable_false_stops = 0
    unrecoverable_false_stops = 0
    overthinking_false_escalates = 0
    successful_escalates = 0
    exhausted_escalates = 0

    error_counts: dict[str, int] = dict.fromkeys(ErrorType.all_types(), 0)
    detailed_cases: list[dict[str, Any]] = []

    for task in tasks:
        p_res = probe.run(task)
        features = extract_features(task, p_res)
        decision = gate.predict(features=features, k=k, probe_tokens=p_res.tokens_used)

        gt = (task.ground_truth or "").strip()
        probe_correct = p_res.answer.strip().lower() == gt.lower()

        mas_answer, mas_tokens = mas_fn(task, decision.token_budget_cap or 1000)
        mas_correct = mas_answer.strip().lower() == gt.lower()

        quadrant = ""
        sub_type = ""

        if decision.decision == "STOP":
            if probe_correct:
                true_stops += 1
                quadrant = "TRUE_STOP"
            else:
                false_stops += 1
                quadrant = "FALSE_STOP"
                if mas_correct:
                    recoverable_false_stops += 1
                    sub_type = "recoverable_missed_escalation"
                else:
                    unrecoverable_false_stops += 1
                    sub_type = "unrecoverable_error"

                diag = analyzer.diagnose_error(task, p_res.answer)
                err_val = (
                    diag.error_type.value
                    if hasattr(diag.error_type, "value")
                    else str(diag.error_type)
                )
                error_counts[err_val] = error_counts.get(err_val, 0) + 1
        else:  # ESCALATE
            if not probe_correct:
                true_escalates += 1
                quadrant = "TRUE_ESCALATE"
                if mas_correct:
                    successful_escalates += 1
                    sub_type = "successful_escalation"
                else:
                    exhausted_escalates += 1
                    sub_type = "exhausted_escalation"
            else:
                false_escalates += 1
                quadrant = "FALSE_ESCALATE"
                if not mas_correct:
                    overthinking_false_escalates += 1
                    sub_type = "overthinking_corruption"
                else:
                    sub_type = "pure_token_waste"

                diag = analyzer.diagnose_error(task, mas_answer)
                err_val = (
                    diag.error_type.value
                    if hasattr(diag.error_type, "value")
                    else str(diag.error_type)
                )
                error_counts[err_val] = error_counts.get(err_val, 0) + 1

        detailed_cases.append(
            {
                "task_id": task.task_id,
                "question": task.question[:80] + ("..." if len(task.question) > 80 else ""),
                "ground_truth": gt,
                "probe_answer": p_res.answer,
                "probe_correct": probe_correct,
                "gate_decision": decision.decision,
                "gate_confidence": decision.confidence,
                "mas_answer": mas_answer if decision.decision == "ESCALATE" else None,
                "mas_correct": mas_correct if decision.decision == "ESCALATE" else None,
                "quadrant": quadrant,
                "sub_type": sub_type,
            }
        )

    # Precision, Recall, F1 for the ESCALATE decision
    prec = (
        true_escalates / (true_escalates + false_escalates)
        if (true_escalates + false_escalates) > 0
        else 0.0
    )
    rec = (
        true_escalates / (true_escalates + false_stops)
        if (true_escalates + false_stops) > 0
        else 0.0
    )
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return FailureTaxonomyResult(
        total_tasks=len(tasks),
        true_stops=true_stops,
        false_stops=false_stops,
        true_escalates=true_escalates,
        false_escalates=false_escalates,
        recoverable_false_stops=recoverable_false_stops,
        unrecoverable_false_stops=unrecoverable_false_stops,
        overthinking_false_escalates=overthinking_false_escalates,
        successful_escalates=successful_escalates,
        exhausted_escalates=exhausted_escalates,
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        error_type_breakdown={k: v for k, v in error_counts.items() if v > 0},
        detailed_cases=detailed_cases,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature Ablations
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AblationRunMetrics:
    """Metrics for an individual feature ablation configuration."""

    name: str
    dropped_features: list[str]
    accuracy: float
    avg_tokens: float
    token_savings_pct: float
    stop_rate_pct: float
    delta_accuracy: float = 0.0
    delta_savings: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_feature_ablations(
    train_tasks: list[Task],
    test_tasks: list[Task],
    seeds: list[int],
    k: int = K_DEFAULT,
) -> list[AblationRunMetrics]:
    """Evaluate Leave-One-Out and Subset feature ablations across seeds."""
    ablation_configs: list[tuple[str, set[str]]] = [
        ("Full Features (Baseline)", set()),
        ("Leave-One-Out: -consistency_score", {"consistency_score"}),
        ("Leave-One-Out: -probe_tokens", {"probe_tokens"}),
        ("Leave-One-Out: -question_word_count", {"question_word_count"}),
        ("Leave-One-Out: -entity_count", {"entity_count"}),
        ("Leave-One-Out: -clause_count", {"clause_count"}),
        ("Leave-One-Out: -has_context", {"has_context"}),
        ("Leave-One-Out: -estimated_depth", {"estimated_depth"}),
        ("Leave-One-Out: -estimated_parallel", {"estimated_parallel"}),
        (
            "Subset: consistency_only",
            {
                "probe_tokens",
                "question_word_count",
                "entity_count",
                "clause_count",
                "has_context",
                "estimated_depth",
                "estimated_parallel",
            },
        ),
        (
            "Subset: probe_only",
            {
                "question_word_count",
                "entity_count",
                "clause_count",
                "has_context",
                "estimated_depth",
                "estimated_parallel",
            },
        ),
        (
            "Subset: text_only",
            {
                "consistency_score",
                "probe_tokens",
                "estimated_depth",
                "estimated_parallel",
            },
        ),
        (
            "Subset: structure_only",
            {
                "consistency_score",
                "probe_tokens",
                "question_word_count",
                "entity_count",
                "clause_count",
                "has_context",
            },
        ),
    ]

    baseline_acc = 0.0
    baseline_sav = 0.0
    results: list[AblationRunMetrics] = []

    for name, dropped in ablation_configs:
        accs: list[float] = []
        tokens_list: list[float] = []
        savings_list: list[float] = []
        stop_rates: list[float] = []

        for seed in seeds:
            probe = SimulatedProbe(seed=seed)
            mas_fn = _make_simulated_mas(seed)

            # Baseline Always-MAS tokens for savings reference
            always_mas = _always_mas_baseline(test_tasks, mas_fn, 1000)
            mas_tokens_total = sum(r.tokens_spent for r in always_mas)

            gate = _train_gate_with_ablation(
                train_tasks=train_tasks,
                val_tasks=test_tasks,
                probe=probe,
                mas_fn=mas_fn,
                dropped_features=dropped,
                classifier_type="logreg",
                k=k,
            )

            # Evaluate on test set
            eval_results = run_batch(
                tasks=test_tasks,
                gate=gate,
                probe_agent=probe.run,
                orchestrator=mas_fn,
                accountant=TokenAccountant(),
                k=k,
            )

            correct_count = sum(1 for r in eval_results if r.is_correct is True)
            acc = correct_count / len(eval_results) if eval_results else 0.0
            tot_tokens = sum(r.tokens_spent for r in eval_results)
            avg_tok = tot_tokens / len(eval_results) if eval_results else 0.0

            sav = (
                (mas_tokens_total - tot_tokens) / mas_tokens_total * 100.0
                if mas_tokens_total > 0
                else 0.0
            )
            stops = sum(
                1 for r in eval_results if r.gate_decision and r.gate_decision.decision == "STOP"
            )
            stop_rate = stops / len(eval_results) * 100.0 if eval_results else 0.0

            accs.append(acc * 100.0)
            tokens_list.append(avg_tok)
            savings_list.append(sav)
            stop_rates.append(stop_rate)

        mean_acc = round(mean(accs), 2)
        mean_tok = round(mean(tokens_list), 1)
        mean_sav = round(mean(savings_list), 2)
        mean_stop = round(mean(stop_rates), 2)

        if "Baseline" in name:
            baseline_acc = mean_acc
            baseline_sav = mean_sav
            delta_acc = 0.0
            delta_sav = 0.0
        else:
            delta_acc = round(mean_acc - baseline_acc, 2)
            delta_sav = round(mean_sav - baseline_sav, 2)

        metric = AblationRunMetrics(
            name=name,
            dropped_features=sorted(dropped),
            accuracy=mean_acc,
            avg_tokens=mean_tok,
            token_savings_pct=mean_sav,
            stop_rate_pct=mean_stop,
            delta_accuracy=delta_acc,
            delta_savings=delta_sav,
        )
        results.append(metric)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. CoT-SC Sample Size Sensitivity (N-sweep)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class NSweepMetrics:
    """Metrics for CoT-SC sample size sensitivity N in {3, 5, 7}."""

    n_samples: int
    probe_accuracy: float
    probe_avg_tokens: float
    pipeline_accuracy: float
    pipeline_avg_tokens: float
    token_savings_pct: float
    stop_rate_pct: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_n_sweep(
    train_tasks: list[Task],
    test_tasks: list[Task],
    seeds: list[int],
    n_values: list[int] | None = None,
    k: int = K_DEFAULT,
) -> list[NSweepMetrics]:
    """Evaluate performance trade-offs for N in {3, 5, 7}."""
    n_vals = n_values or [3, 5, 7]
    sweep_results: list[NSweepMetrics] = []

    for n in n_vals:
        p_accs: list[float] = []
        p_toks: list[float] = []
        pipe_accs: list[float] = []
        pipe_toks: list[float] = []
        savings_list: list[float] = []
        stop_rates: list[float] = []

        for seed in seeds:
            probe = SimulatedProbe(seed=seed, n_samples=n)
            mas_fn = _make_simulated_mas(seed)

            # Probe-only stats
            cot_results = _cot_sc_baseline(test_tasks, probe)
            p_correct = sum(1 for r in cot_results if r.is_correct is True)
            p_acc = p_correct / len(cot_results) * 100.0 if cot_results else 0.0
            p_tok = (
                sum(r.tokens_spent for r in cot_results) / len(cot_results) if cot_results else 0.0
            )

            # Train gate calibrated with n-sample probe
            gate = _train_gate_with_ablation(
                train_tasks=train_tasks,
                val_tasks=test_tasks,
                probe=probe,
                mas_fn=mas_fn,
                classifier_type="logreg",
                k=k,
            )

            always_mas = _always_mas_baseline(test_tasks, mas_fn, 1000)
            mas_tokens_total = sum(r.tokens_spent for r in always_mas)

            eval_results = run_batch(
                tasks=test_tasks,
                gate=gate,
                probe_agent=probe.run,
                orchestrator=mas_fn,
                accountant=TokenAccountant(),
                k=k,
            )

            correct = sum(1 for r in eval_results if r.is_correct is True)
            pipe_acc = correct / len(eval_results) * 100.0 if eval_results else 0.0
            tot_tokens = sum(r.tokens_spent for r in eval_results)
            pipe_tok = tot_tokens / len(eval_results) if eval_results else 0.0

            sav = (
                (mas_tokens_total - tot_tokens) / mas_tokens_total * 100.0
                if mas_tokens_total > 0
                else 0.0
            )
            stops = sum(
                1 for r in eval_results if r.gate_decision and r.gate_decision.decision == "STOP"
            )
            stop_rate = stops / len(eval_results) * 100.0 if eval_results else 0.0

            p_accs.append(p_acc)
            p_toks.append(p_tok)
            pipe_accs.append(pipe_acc)
            pipe_toks.append(pipe_tok)
            savings_list.append(sav)
            stop_rates.append(stop_rate)

        sweep_results.append(
            NSweepMetrics(
                n_samples=n,
                probe_accuracy=round(mean(p_accs), 2),
                probe_avg_tokens=round(mean(p_toks), 1),
                pipeline_accuracy=round(mean(pipe_accs), 2),
                pipeline_avg_tokens=round(mean(pipe_toks), 1),
                token_savings_pct=round(mean(savings_list), 2),
                stop_rate_pct=round(mean(stop_rates), 2),
            )
        )

    return sweep_results


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pareto Frontier Analysis (k in {2, 3, 5})
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ParetoPoint:
    """Single point on the Pareto frontier curve."""

    method: str
    k: int
    accuracy: float
    avg_tokens: float
    token_savings_pct: float
    stop_rate_pct: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_pareto_analysis(
    train_tasks: list[Task],
    test_tasks: list[Task],
    seeds: list[int],
    k_values: list[int] | None = None,
) -> list[ParetoPoint]:
    """Generate Pareto points across k in {2, 3, 5} for all gating methods."""
    k_vals = k_values or [2, 3, 5]
    points: list[ParetoPoint] = []

    methods = [
        "CoT-SC-only",
        "Always-MAS",
        "RandomGate",
        "RuleBasedGate",
        "GateOrchestra",
    ]

    for k in k_vals:
        method_stats: dict[str, dict[str, list[float]]] = {
            m: {"acc": [], "tokens": [], "savings": [], "stop_rate": []} for m in methods
        }

        for seed in seeds:
            probe = SimulatedProbe(seed=seed)
            mas_fn = _make_simulated_mas(seed)

            # Always-MAS token upper bound for current k
            budget_cap = k * 230  # probe tokens approx 230
            mas_res = _always_mas_baseline(test_tasks, mas_fn, budget_cap)
            mas_tot_tokens = sum(r.tokens_spent for r in mas_res)
            mas_acc = sum(1 for r in mas_res if r.is_correct is True) / len(mas_res) * 100.0
            mas_avg_tok = mas_tot_tokens / len(mas_res)

            method_stats["Always-MAS"]["acc"].append(mas_acc)
            method_stats["Always-MAS"]["tokens"].append(mas_avg_tok)
            method_stats["Always-MAS"]["savings"].append(0.0)
            method_stats["Always-MAS"]["stop_rate"].append(0.0)

            # CoT-SC-only
            cot_res = _cot_sc_baseline(test_tasks, probe)
            cot_tot = sum(r.tokens_spent for r in cot_res)
            cot_acc = sum(1 for r in cot_res if r.is_correct is True) / len(cot_res) * 100.0
            cot_avg = cot_tot / len(cot_res)
            cot_sav = (
                (mas_tot_tokens - cot_tot) / mas_tot_tokens * 100.0 if mas_tot_tokens > 0 else 0.0
            )

            method_stats["CoT-SC-only"]["acc"].append(cot_acc)
            method_stats["CoT-SC-only"]["tokens"].append(cot_avg)
            method_stats["CoT-SC-only"]["savings"].append(cot_sav)
            method_stats["CoT-SC-only"]["stop_rate"].append(100.0)

            # RandomGate
            rand_gate = RandomGate(escalation_rate=0.50, seed=seed)
            rand_res = run_batch(
                tasks=test_tasks,
                gate=rand_gate,
                probe_agent=probe.run,
                orchestrator=mas_fn,
                accountant=TokenAccountant(),
                k=k,
            )
            rand_tot = sum(r.tokens_spent for r in rand_res)
            rand_acc = sum(1 for r in rand_res if r.is_correct is True) / len(rand_res) * 100.0
            rand_avg = rand_tot / len(rand_res)
            rand_sav = (
                (mas_tot_tokens - rand_tot) / mas_tot_tokens * 100.0 if mas_tot_tokens > 0 else 0.0
            )
            rand_stop = (
                sum(1 for r in rand_res if r.gate_decision and r.gate_decision.decision == "STOP")
                / len(rand_res)
                * 100.0
            )

            method_stats["RandomGate"]["acc"].append(rand_acc)
            method_stats["RandomGate"]["tokens"].append(rand_avg)
            method_stats["RandomGate"]["savings"].append(rand_sav)
            method_stats["RandomGate"]["stop_rate"].append(rand_stop)

            # RuleBasedGate
            rule_gate = RuleBasedGate()
            rule_res = run_batch(
                tasks=test_tasks,
                gate=rule_gate,
                probe_agent=probe.run,
                orchestrator=mas_fn,
                accountant=TokenAccountant(),
                k=k,
            )
            rule_tot = sum(r.tokens_spent for r in rule_res)
            rule_acc = sum(1 for r in rule_res if r.is_correct is True) / len(rule_res) * 100.0
            rule_avg = rule_tot / len(rule_res)
            rule_sav = (
                (mas_tot_tokens - rule_tot) / mas_tot_tokens * 100.0 if mas_tot_tokens > 0 else 0.0
            )
            rule_stop = (
                sum(1 for r in rule_res if r.gate_decision and r.gate_decision.decision == "STOP")
                / len(rule_res)
                * 100.0
            )

            method_stats["RuleBasedGate"]["acc"].append(rule_acc)
            method_stats["RuleBasedGate"]["tokens"].append(rule_avg)
            method_stats["RuleBasedGate"]["savings"].append(rule_sav)
            method_stats["RuleBasedGate"]["stop_rate"].append(rule_stop)

            # GateOrchestra
            go_gate = _train_gate_with_ablation(
                train_tasks=train_tasks,
                val_tasks=test_tasks,
                probe=probe,
                mas_fn=mas_fn,
                classifier_type="logreg",
                k=k,
            )
            go_res = run_batch(
                tasks=test_tasks,
                gate=go_gate,
                probe_agent=probe.run,
                orchestrator=mas_fn,
                accountant=TokenAccountant(),
                k=k,
            )
            go_tot = sum(r.tokens_spent for r in go_res)
            go_acc = sum(1 for r in go_res if r.is_correct is True) / len(go_res) * 100.0
            go_avg = go_tot / len(go_res)
            go_sav = (
                (mas_tot_tokens - go_tot) / mas_tot_tokens * 100.0 if mas_tot_tokens > 0 else 0.0
            )
            go_stop = (
                sum(1 for r in go_res if r.gate_decision and r.gate_decision.decision == "STOP")
                / len(go_res)
                * 100.0
            )

            method_stats["GateOrchestra"]["acc"].append(go_acc)
            method_stats["GateOrchestra"]["tokens"].append(go_avg)
            method_stats["GateOrchestra"]["savings"].append(go_sav)
            method_stats["GateOrchestra"]["stop_rate"].append(go_stop)

        for m in methods:
            points.append(
                ParetoPoint(
                    method=m,
                    k=k,
                    accuracy=round(mean(method_stats[m]["acc"]), 2),
                    avg_tokens=round(mean(method_stats[m]["tokens"]), 1),
                    token_savings_pct=round(mean(method_stats[m]["savings"]), 2),
                    stop_rate_pct=round(mean(method_stats[m]["stop_rate"]), 2),
                )
            )

    return points


def plot_pareto_frontier(points: list[ParetoPoint], output_path: Path) -> None:
    """Generate high-resolution Pareto frontier curve using Matplotlib."""
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    color_map = {
        "CoT-SC-only": "#94a3b8",
        "Always-MAS": "#f97316",
        "RandomGate": "#ef4444",
        "RuleBasedGate": "#3b82f6",
        "GateOrchestra": "#10b981",
    }
    marker_map = {
        "CoT-SC-only": "s",
        "Always-MAS": "^",
        "RandomGate": "x",
        "RuleBasedGate": "d",
        "GateOrchestra": "o",
    }

    # Group points by method
    method_data: dict[str, list[ParetoPoint]] = {}
    for pt in points:
        method_data.setdefault(pt.method, []).append(pt)

    for method, p_list in method_data.items():
        p_list.sort(key=lambda p: p.avg_tokens)
        xs = [p.avg_tokens for p in p_list]
        ys = [p.accuracy for p in p_list]

        c = color_map.get(method, "#64748b")
        m = marker_map.get(method, "o")

        ax.plot(
            xs,
            ys,
            label=method,
            color=c,
            marker=m,
            markersize=8,
            linewidth=2 if method == "GateOrchestra" else 1.5,
            linestyle="-" if method == "GateOrchestra" else "--",
        )

        for p in p_list:
            if method in ("GateOrchestra", "RuleBasedGate", "Always-MAS"):
                ax.annotate(
                    f"k={p.k}",
                    (p.avg_tokens, p.accuracy),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=8,
                    color=c,
                    fontweight="bold" if method == "GateOrchestra" else "normal",
                )

    ax.set_title(
        "Pareto Frontier: Accuracy vs. Token Expenditure across Multiplier k",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Average Token Spend Per Task", fontsize=10, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 5. LinUCB Bandit Strategy Breakdown
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class BanditStrategyBreakdown:
    """Breakdown of LinUCB contextual bandit arm selections and rewards."""

    total_tasks: int
    arm_counts: dict[str, int]
    arm_proportions: dict[str, float]
    by_depth: dict[int, dict[str, int]]
    by_parallel: dict[int, dict[str, int]]
    by_task_type: dict[str, dict[str, int]]
    mean_rewards: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_bandit_breakdown(
    tasks: list[Task],
    seed: int = 42,
    alpha: float = 0.5,
) -> BanditStrategyBreakdown:
    """Analyze LinUCB arm selection distribution across task dimensions."""
    _rng = _random.Random(seed)
    router = LinUCBRouter(alpha=alpha)

    arm_counts = dict.fromkeys(router.arms, 0)
    rewards_acc: dict[str, list[float]] = {arm: [] for arm in router.arms}

    by_depth: dict[int, dict[str, int]] = {}
    by_parallel: dict[int, dict[str, int]] = {}
    by_task_type: dict[str, dict[str, int]] = {}

    for task in tasks:
        chosen_arm = router.select_arm(task)
        arm_counts[chosen_arm] += 1

        # Track by depth
        depth = task.depth_score if task.depth_score is not None else 2
        by_depth.setdefault(depth, dict.fromkeys(router.arms, 0))[chosen_arm] += 1

        # Track by parallel
        parallel = task.parallel_score if task.parallel_score is not None else 1
        by_parallel.setdefault(parallel, dict.fromkeys(router.arms, 0))[chosen_arm] += 1

        # Track by task type
        ttype = task.source_dataset or "unknown"
        by_task_type.setdefault(ttype, dict.fromkeys(router.arms, 0))[chosen_arm] += 1

        # Simulate task success reward depending on strategy match
        # react excels on deep multi-hop (depth >= 3)
        # debate excels on parallel reasoning (parallel >= 2)
        # reflexion excels on verification / self-correction
        success_prob = 0.65
        if chosen_arm == "react" and depth >= 3:
            success_prob += 0.20
        elif chosen_arm == "debate" and parallel >= 2:
            success_prob += 0.20
        elif chosen_arm == "reflexion" and depth <= 2:
            success_prob += 0.15

        reward = 1.0 if _rng.random() < min(0.95, success_prob) else 0.0
        router.update(task, chosen_arm, reward)
        rewards_acc[chosen_arm].append(reward)

    total = len(tasks)
    arm_props = {
        arm: round(count / total * 100.0, 1) if total > 0 else 0.0
        for arm, count in arm_counts.items()
    }
    mean_rewards = {
        arm: round(mean(r_list), 3) if r_list else 0.0 for arm, r_list in rewards_acc.items()
    }

    return BanditStrategyBreakdown(
        total_tasks=total,
        arm_counts=arm_counts,
        arm_proportions=arm_props,
        by_depth=by_depth,
        by_parallel=by_parallel,
        by_task_type=by_task_type,
        mean_rewards=mean_rewards,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Markdown Report Generator
# ─────────────────────────────────────────────────────────────────────────────


def generate_markdown_report(
    taxonomy: FailureTaxonomyResult | None,
    ablations: list[AblationRunMetrics] | None,
    n_sweep: list[NSweepMetrics] | None,
    pareto: list[ParetoPoint] | None,
    bandit: BanditStrategyBreakdown | None,
    output_path: Path,
) -> None:
    """Generate a clean, publication-ready markdown report of Week 6 findings."""
    lines: list[str] = [
        "# Week 6 — Gate Analysis & Ablations Report",
        "",
        "> **System:** GateOrchestra | **Role:** Person 3 (Architecture, Gate, Integration)",
        f"> **Generated at:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        "---",
        "",
    ]

    # 1. Taxonomy
    if taxonomy:
        lines.extend(
            [
                "## 1. Gate Failure Taxonomy & Diagnostic Analysis",
                "",
                "The Gate routing decisions were partitioned into four empirical quadrants based on whether",
                "the probe baseline was correct and whether the gate triggered multi-agent escalation (MAS):",
                "",
                "| Quadrant | Count | % of Test | Interpretation |",
                "|---|---|---|---|",
                f"| **True STOP** | {taxonomy.true_stops} | {taxonomy.true_stops/taxonomy.total_tasks*100:.1f}% | Optimal token saving (Probe correct, MAS bypassed) |",
                f"| **False STOP (Type II)** | {taxonomy.false_stops} | {taxonomy.false_stops/taxonomy.total_tasks*100:.1f}% | Uncaught error (Probe incorrect, MAS skipped) |",
                f"| **True ESCALATE** | {taxonomy.true_escalates} | {taxonomy.true_escalates/taxonomy.total_tasks*100:.1f}% | Necessary escalation (Probe incorrect, MAS engaged) |",
                f"| **False ESCALATE (Type I)** | {taxonomy.false_escalates} | {taxonomy.false_escalates/taxonomy.total_tasks*100:.1f}% | Token budget waste (Probe already correct, MAS invoked) |",
                "",
                f"- **Gate Decision Precision (ESCALATE):** `{taxonomy.precision:.3f}`",
                f"- **Gate Decision Recall (ESCALATE):** `{taxonomy.recall:.3f}`",
                f"- **Gate Decision F1 Score:** `{taxonomy.f1:.3f}`",
                f"- **Recoverable False STOPs:** {taxonomy.recoverable_false_stops} (Tasks where MAS would have produced the correct answer)",
                f"- **Unrecoverable False STOPs:** {taxonomy.unrecoverable_false_stops} (Tasks that failed under both probe and MAS)",
                "",
                "### Failure Modes by Error Taxonomy",
                "Cross-referencing failure cases with `dataset/error_labels.py`:",
                "",
                "| Error Category | Occurrences |",
                "|---|---|",
            ]
        )
        for cat, cnt in sorted(taxonomy.error_type_breakdown.items(), key=lambda x: -x[1]):
            lines.append(f"| `{cat}` | {cnt} |")
        lines.append("")

    # 2. Feature Ablations
    if ablations:
        lines.extend(
            [
                "## 2. Feature Ablations (Leave-One-Out & Subsets)",
                "",
                "Evaluating the empirical contribution of the 8 extracted features by training and testing gates",
                "with individual features or entire feature subsets masked out:",
                "",
                "| Configuration | Accuracy (%) | Avg Tokens | Token Savings (%) | STOP Rate (%) | Δ Acc (%) | Δ Savings (%) |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for ab in ablations:
            lines.append(
                f"| {ab.name} | {ab.accuracy:.1f}% | {ab.avg_tokens:.0f} | "
                f"{ab.token_savings_pct:+.1f}% | {ab.stop_rate_pct:.1f}% | "
                f"{ab.delta_accuracy:+.1f}% | {ab.delta_savings:+.1f}% |"
            )
        lines.append("")

    # 3. N-sweep
    if n_sweep:
        lines.extend(
            [
                "## 3. CoT-SC Sample Size Sensitivity ($N \\in \\{3, 5, 7\\}$)",
                "",
                "Evaluating how probe budget scaling affects consistency precision and downstream gate accuracy:",
                "",
                "| $N$ Samples | Probe Acc (%) | Probe Tokens | Pipeline Acc (%) | Total Tokens | Savings vs Always-MAS | STOP Rate (%) |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for ns in n_sweep:
            lines.append(
                f"| $N={ns.n_samples}$ | {ns.probe_accuracy:.1f}% | {ns.probe_avg_tokens:.0f} | "
                f"{ns.pipeline_accuracy:.1f}% | {ns.pipeline_avg_tokens:.0f} | "
                f"{ns.token_savings_pct:+.1f}% | {ns.stop_rate_pct:.1f}% |"
            )
        lines.append("")

    # 4. Pareto Frontier
    if pareto:
        lines.extend(
            [
                "## 4. Pareto Frontier Analysis ($k \\in \\{2, 3, 5\\}$)",
                "",
                "Token spend vs accuracy trade-off across budget multiplier $k$:",
                "",
                "| Method | $k$ | Accuracy (%) | Avg Tokens | Token Savings (%) | STOP Rate (%) |",
                "|---|---|---|---|---|---|",
            ]
        )
        for p in pareto:
            lines.append(
                f"| {p.method} | {p.k} | {p.accuracy:.1f}% | {p.avg_tokens:.0f} | "
                f"{p.token_savings_pct:+.1f}% | {p.stop_rate_pct:.1f}% |"
            )
        lines.append("")
        lines.append("![Pareto Frontier](reports/figures/week6_pareto_frontier.png)")
        lines.append("")

    # 5. Bandit Strategy Breakdown
    if bandit:
        lines.extend(
            [
                "## 5. LinUCB Bandit Strategy Routing Breakdown",
                "",
                "Distribution of sub-agent strategy assignments by the Contextual Multi-Armed Bandit:",
                "",
                f"- **Total Tasks Routed:** {bandit.total_tasks}",
                f"- **ReAct Strategy:** {bandit.arm_counts.get('react', 0)} ({bandit.arm_proportions.get('react', 0.0):.1f}%) | Mean Reward: {bandit.mean_rewards.get('react', 0.0):.3f}",
                f"- **Debate Strategy:** {bandit.arm_counts.get('debate', 0)} ({bandit.arm_proportions.get('debate', 0.0):.1f}%) | Mean Reward: {bandit.mean_rewards.get('debate', 0.0):.3f}",
                f"- **Reflexion Strategy:** {bandit.arm_counts.get('reflexion', 0)} ({bandit.arm_proportions.get('reflexion', 0.0):.1f}%) | Mean Reward: {bandit.mean_rewards.get('reflexion', 0.0):.3f}",
                "",
                "### Strategy Distribution by Reasoning Depth",
                "",
                "| Depth Level | ReAct | Debate | Reflexion |",
                "|---|---|---|---|",
            ]
        )
        for d in sorted(bandit.by_depth.keys()):
            row = bandit.by_depth[d]
            lines.append(
                f"| Depth {d} | {row.get('react', 0)} | {row.get('debate', 0)} | {row.get('reflexion', 0)} |"
            )
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Markdown report written to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GateOrchestra Week 6 Gate Analysis & Ablation Suite."
    )
    parser.add_argument(
        "--mode",
        choices=["all", "taxonomy", "feature_ablation", "n_sweep", "pareto", "bandit"],
        default="all",
        help="Analysis mode to execute (default: all).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 999],
        help="Random seeds for multi-run evaluations (default: 42 123 999).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=K_DEFAULT,
        help=f"Token budget multiplier cap k (default: {K_DEFAULT}).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(RESULTS_DIR / "week6_ablations.json"),
        help="Path to save main ablation JSON results.",
    )
    parser.add_argument(
        "--output-pareto-json",
        type=str,
        default=str(RESULTS_DIR / "week6_pareto.json"),
        help="Path to save Pareto frontier JSON points.",
    )
    parser.add_argument(
        "--output-fig",
        type=str,
        default=str(FIGURES_DIR / "week6_pareto_frontier.png"),
        help="Path to save Pareto frontier figure.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=str(REPO_ROOT / "reports" / "week6_gate_analysis.md"),
        help="Path to save output Markdown summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("=" * 70)
    print(" GATEORCHESTRA WEEK 6 - GATE ANALYSIS & ABLATIONS SUITE")
    print(f" Mode: {args.mode} | Seeds: {args.seeds} | Multiplier k: {args.k}")
    print("=" * 70)

    # 1. Load data
    train_tasks = load_split("train")
    test_tasks = load_split("test")
    print(f"[DATA] Loaded {len(train_tasks)} train tasks and {len(test_tasks)} test tasks.")

    taxonomy_res: FailureTaxonomyResult | None = None
    ablation_res: list[AblationRunMetrics] | None = None
    n_sweep_res: list[NSweepMetrics] | None = None
    pareto_res: list[ParetoPoint] | None = None
    bandit_res: BanditStrategyBreakdown | None = None

    # --- Mode 1: Taxonomy ---
    if args.mode in ("all", "taxonomy"):
        print("\n" + "-" * 70)
        print(" [MODE 1] GATE FAILURE TAXONOMY & DIAGNOSTIC PROFILING")
        print("-" * 70)
        probe = SimulatedProbe(seed=args.seeds[0])
        mas_fn = _make_simulated_mas(args.seeds[0])
        gate = _train_gate_with_ablation(
            train_tasks=train_tasks,
            val_tasks=test_tasks,
            probe=probe,
            mas_fn=mas_fn,
            classifier_type="logreg",
            k=args.k,
        )
        taxonomy_res = run_gate_taxonomy(
            tasks=test_tasks,
            gate=gate,
            probe=probe,
            mas_fn=mas_fn,
            k=args.k,
        )

        print(f" Total Evaluated: {taxonomy_res.total_tasks} tasks")
        print(
            f"   True STOPs (Efficient saves):        {taxonomy_res.true_stops:2d} ({taxonomy_res.true_stops/taxonomy_res.total_tasks*100:.1f}%)"
        )
        print(
            f"   False STOPs (Missed escalations):    {taxonomy_res.false_stops:2d} ({taxonomy_res.false_stops/taxonomy_res.total_tasks*100:.1f}%)"
        )
        print(f"     -> Recoverable by MAS:             {taxonomy_res.recoverable_false_stops:2d}")
        print(
            f"     -> Intractable (both failed):      {taxonomy_res.unrecoverable_false_stops:2d}"
        )
        print(
            f"   True ESCALATEs (Necessary calls):    {taxonomy_res.true_escalates:2d} ({taxonomy_res.true_escalates/taxonomy_res.total_tasks*100:.1f}%)"
        )
        print(f"     -> Rescued by MAS:                 {taxonomy_res.successful_escalates:2d}")
        print(f"     -> Exhausted (both failed):        {taxonomy_res.exhausted_escalates:2d}")
        print(
            f"   False ESCALATEs (Token waste):       {taxonomy_res.false_escalates:2d} ({taxonomy_res.false_escalates/taxonomy_res.total_tasks*100:.1f}%)"
        )
        print(f" Gate Decision Precision: {taxonomy_res.precision:.3f}")
        print(f" Gate Decision Recall:    {taxonomy_res.recall:.3f}")
        print(f" Gate Decision F1 Score:  {taxonomy_res.f1:.3f}")

    # --- Mode 2: Feature Ablations ---
    if args.mode in ("all", "feature_ablation"):
        print("\n" + "-" * 70)
        print(" [MODE 2] FEATURE ABLATIONS (LEAVE-ONE-OUT & SUBSETS)")
        print("-" * 70)
        ablation_res = run_feature_ablations(
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            seeds=args.seeds,
            k=args.k,
        )
        print(
            f"{'Configuration':<38} | {'Acc (%)':<7} | {'Tokens':<6} | {'Savings':<8} | {'dAcc':<6} | {'dSav':<6}"
        )
        print("-" * 78)
        for ab in ablation_res:
            print(
                f"{ab.name:<38} | {ab.accuracy:>6.1f}% | {ab.avg_tokens:>6.0f} | "
                f"{ab.token_savings_pct:>+7.1f}% | {ab.delta_accuracy:>+5.1f}% | {ab.delta_savings:>+5.1f}%"
            )

    # --- Mode 3: N-sweep ---
    if args.mode in ("all", "n_sweep"):
        print("\n" + "-" * 70)
        print(" [MODE 3] CoT-SC SAMPLE SIZE SENSITIVITY (N in {3, 5, 7})")
        print("-" * 70)
        n_sweep_res = run_n_sweep(
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            seeds=args.seeds,
            n_values=[3, 5, 7],
            k=args.k,
        )
        print(
            f"{'N':<4} | {'Probe Acc':<10} | {'Probe Tok':<10} | {'Pipeline Acc':<12} | {'Total Tok':<10} | {'Savings':<8} | {'STOP Rate':<10}"
        )
        print("-" * 76)
        for ns in n_sweep_res:
            print(
                f"N={ns.n_samples:<2} | {ns.probe_accuracy:>8.1f}% | {ns.probe_avg_tokens:>9.0f} | "
                f"{ns.pipeline_accuracy:>10.1f}% | {ns.pipeline_avg_tokens:>9.0f} | "
                f"{ns.token_savings_pct:>+7.1f}% | {ns.stop_rate_pct:>8.1f}%"
            )

    # --- Mode 4: Pareto Frontier ---
    if args.mode in ("all", "pareto"):
        print("\n" + "-" * 70)
        print(" [MODE 4] PARETO FRONTIER ANALYSIS (k in {2, 3, 5})")
        print("-" * 70)
        pareto_res = run_pareto_analysis(
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            seeds=args.seeds,
            k_values=[2, 3, 5],
        )
        print(
            f"{'Method':<16} | {'k':<2} | {'Accuracy':<9} | {'Avg Tokens':<10} | {'Savings':<8} | {'STOP Rate':<9}"
        )
        print("-" * 65)
        for p in pareto_res:
            print(
                f"{p.method:<16} | {p.k:<2} | {p.accuracy:>7.1f}% | {p.avg_tokens:>9.1f} | "
                f"{p.token_savings_pct:>+7.1f}% | {p.stop_rate_pct:>8.1f}%"
            )

        # Plot figure
        fig_path = Path(args.output_fig)
        plot_pareto_frontier(pareto_res, fig_path)
        print(f"[FIGURE] Saved Pareto curve visualization to {fig_path}")

        # Save Pareto JSON
        pareto_json_path = Path(args.output_pareto_json)
        pareto_json_path.parent.mkdir(parents=True, exist_ok=True)
        with pareto_json_path.open("w", encoding="utf-8") as f:
            json.dump([p.to_dict() for p in pareto_res], f, indent=2)
        print(f"[DATA] Saved Pareto metrics JSON to {pareto_json_path}")

    # --- Mode 5: Bandit Breakdown ---
    if args.mode in ("all", "bandit"):
        print("\n" + "-" * 70)
        print(" [MODE 5] LinUCB BANDIT STRATEGY ROUTING BREAKDOWN")
        print("-" * 70)
        bandit_res = run_bandit_breakdown(test_tasks, seed=args.seeds[0])
        print(f" Total Tasks Evaluated: {bandit_res.total_tasks}")
        for arm, cnt in bandit_res.arm_counts.items():
            prop = bandit_res.arm_proportions[arm]
            r = bandit_res.mean_rewards[arm]
            print(f"   Arm: {arm:<10} | Selections: {cnt:2d} ({prop:4.1f}%) | Mean Reward: {r:.3f}")

        print("\n Stratified by Reasoning Depth:")
        for d, arm_dict in sorted(bandit_res.by_depth.items()):
            print(f"   Depth {d}: " + " | ".join(f"{a}: {c}" for a, c in arm_dict.items()))

    # --- Consolidated JSON Output ---
    consolidated_data = {
        "metadata": {
            "sprint": "Week 6 — Gate Analysis & Ablations",
            "owner": "Person 3 (Farhan Aaqil Durrani)",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seeds": args.seeds,
            "k_default": args.k,
            "train_tasks": len(train_tasks),
            "test_tasks": len(test_tasks),
        },
        "taxonomy": taxonomy_res.to_dict() if taxonomy_res else None,
        "feature_ablations": [ab.to_dict() for ab in ablation_res] if ablation_res else None,
        "n_sweep": [ns.to_dict() for ns in n_sweep_res] if n_sweep_res else None,
        "pareto_frontier": [p.to_dict() for p in pareto_res] if pareto_res else None,
        "bandit_breakdown": bandit_res.to_dict() if bandit_res else None,
    }

    out_json_path = Path(args.output_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(consolidated_data, f, indent=2)
    print(f"\n[DATA] Consolidated results written to {out_json_path}")

    # --- Markdown Summary Output ---
    md_path = Path(args.output_md)
    generate_markdown_report(
        taxonomy=taxonomy_res,
        ablations=ablation_res,
        n_sweep=n_sweep_res,
        pareto=pareto_res,
        bandit=bandit_res,
        output_path=md_path,
    )

    print("\n" + "=" * 70)
    print(" WEEK 6 ABLATIONS & GATE ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""agents/agent_analysis.py
========================
Agent Analysis engine for GateOrchestra (Person 2).

Provides quantitative profiling across:
  - ProbeAgent (CoT-SC)
  - ReActAgent (Decomposition & Multi-hop)
  - DebateAgent (Proposer-Critic Debate)
  - ReflexionAgent (Self-Reflection & Refinement)
  - MASOrchestrator (Heuristic 'auto' and LinUCB 'bandit' routers)

Measures:
  - Accuracy / correctness (exact match against ground truth)
  - Token usage (total, average, min, max)
  - Latency (total, average, min, max in ms)
  - Execution count
  - Strategy / agent selection frequencies
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agents.orchestrator.orchestrator import MASOrchestrator
from agents.orchestrator.sub_agents import (
    DebateAgent,
    ReActAgent,
    ReflexionAgent,
)
from agents.probe_agent import ProbeAgent
from shared.config import K_DEFAULT, PROBE_TOKEN_BUDGET
from shared.schemas import EvalResult, Task
from shared.token_logger import TokenAccountant

logger = logging.getLogger(__name__)

LLMCallerFn = Callable[[str, float, int], tuple[str, int]]


def exact_match(predicted: str, ground_truth: str) -> bool:
    """Normalized exact match comparison between predicted and ground truth."""
    if not predicted or not ground_truth:
        return False

    def normalize(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"[^\w\s]", "", s)
        return " ".join(s.split())

    return normalize(predicted) == normalize(ground_truth)


@dataclass
class AgentPerformanceMetrics:
    """Aggregated performance metrics for a specific agent or routing strategy."""

    name: str
    execution_count: int = 0
    correct_count: int = 0
    accuracy: float = 0.0  # Percentage [0.0, 100.0]
    total_tokens: int = 0
    avg_tokens: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0
    total_latency_ms: float | None = None
    avg_latency_ms: float | None = None
    min_latency_ms: float | None = None
    max_latency_ms: float | None = None
    strategy_counts: dict[str, int] = field(default_factory=dict)
    strategy_frequencies: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


@dataclass
class AgentAnalysisReport:
    """Consolidated agent performance report."""

    metrics: dict[str, AgentPerformanceMetrics] = field(default_factory=dict)
    source_info: str = "evaluation_data"
    total_tasks_analyzed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert full report to dictionary."""
        return {
            "source_info": self.source_info,
            "total_tasks_analyzed": self.total_tasks_analyzed,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Render report as a formatted Markdown table."""
        lines = [
            "# GateOrchestra -- Agent Performance Analysis Report",
            "",
            f"- **Source:** `{self.source_info}`",
            f"- **Total Tasks Analyzed:** {self.total_tasks_analyzed}",
            "",
            "| Agent / Strategy | Executions | Accuracy | Total Tokens | Avg Tokens | Avg Latency (ms) | Strategy Breakdown |",
            "|:-----------------|:----------:|:--------:|:------------:|:----------:|:----------------:|:-------------------|",
        ]

        for name, m in sorted(self.metrics.items()):
            acc_str = f"{m.accuracy:.1f}% ({m.correct_count}/{m.execution_count})"
            lat_str = f"{m.avg_latency_ms:.1f}" if m.avg_latency_ms is not None else "N/A"
            strat_str = (
                ", ".join(f"{k}: {v:.1f}%" for k, v in m.strategy_frequencies.items())
                if m.strategy_frequencies
                else "N/A"
            )
            lines.append(
                f"| **{name}** | {m.execution_count} | {acc_str} | {m.total_tokens:,} | "
                f"{m.avg_tokens:.1f} | {lat_str} | {strat_str} |"
            )

        lines.append("")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Existing Evaluation Data Analysis (Traces / Baseline Results)
# ─────────────────────────────────────────────────────────────────────────────


def analyze_eval_records(
    records: Sequence[dict[str, Any] | EvalResult],
    source_name: str = "trace_records",
) -> AgentAnalysisReport:
    """Compute agent metrics from existing evaluation result records.

    Args:
        records: List of EvalResult instances or dicts matching EvalResult schema.
        source_name: Identifier for the data source.

    Returns:
        AgentAnalysisReport containing aggregated metrics per method/strategy.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}

    for item in records:
        d = item.model_dump() if isinstance(item, EvalResult) else item
        method = str(d.get("method") or "Unknown")
        grouped.setdefault(method, []).append(d)

    metrics_map: dict[str, AgentPerformanceMetrics] = {}
    total_tasks = len(records)

    for method, rows in grouped.items():
        n = len(rows)
        if n == 0:
            continue

        tokens_list = [int(r.get("tokens_spent", 0)) for r in rows]
        total_tokens = sum(tokens_list)
        avg_tokens = round(total_tokens / n, 2)
        min_tokens = min(tokens_list) if tokens_list else 0
        max_tokens = max(tokens_list) if tokens_list else 0

        # Correctness evaluation
        correct_count = 0
        for r in rows:
            is_corr = r.get("is_correct")
            if is_corr is True:
                correct_count += 1
            elif is_corr is None and "predicted_answer" in r and "ground_truth" in r:
                if exact_match(str(r["predicted_answer"]), str(r["ground_truth"])):
                    correct_count += 1

        accuracy = round((correct_count / n) * 100.0, 2) if n > 0 else 0.0

        # Latency evaluation
        latencies = [
            float(r["latency_ms"])
            for r in rows
            if r.get("latency_ms") is not None and float(r["latency_ms"]) >= 0
        ]
        if latencies:
            total_lat = sum(latencies)
            avg_lat = round(total_lat / len(latencies), 2)
            min_lat = round(min(latencies), 2)
            max_lat = round(max(latencies), 2)
        else:
            total_lat, avg_lat, min_lat, max_lat = None, None, None, None

        # Strategy allocation tracking (e.g. from mas_strategy or gate decisions)
        strategy_counts: dict[str, int] = Counter()
        for r in rows:
            strat = r.get("mas_strategy")
            if strat:
                strategy_counts[str(strat)] += 1
            elif "gate_decision" in r and isinstance(r["gate_decision"], dict):
                decision_val = r["gate_decision"].get("decision")
                if decision_val:
                    strategy_counts[str(decision_val)] += 1

        total_strat = sum(strategy_counts.values())
        strategy_frequencies = (
            {k: round((v / total_strat) * 100.0, 2) for k, v in sorted(strategy_counts.items())}
            if total_strat > 0
            else {}
        )

        metrics_map[method] = AgentPerformanceMetrics(
            name=method,
            execution_count=n,
            correct_count=correct_count,
            accuracy=accuracy,
            total_tokens=total_tokens,
            avg_tokens=avg_tokens,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            total_latency_ms=total_lat,
            avg_latency_ms=avg_lat,
            min_latency_ms=min_lat,
            max_latency_ms=max_lat,
            strategy_counts=dict(strategy_counts),
            strategy_frequencies=strategy_frequencies,
        )

    return AgentAnalysisReport(
        metrics=metrics_map,
        source_info=source_name,
        total_tasks_analyzed=total_tasks,
    )


def analyze_trace_file(file_path: str | Path) -> AgentAnalysisReport:
    """Load evaluation records from a JSON or JSONL file and compute metrics.

    Args:
        file_path: Path to .json or .jsonl evaluation results file.

    Returns:
        AgentAnalysisReport.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

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

    return analyze_eval_records(records, source_name=str(path))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Live / Benchmark Agent Profiling Suite
# ─────────────────────────────────────────────────────────────────────────────


def benchmark_agents(
    tasks: list[Task],
    accountant: TokenAccountant | None = None,
    llm_caller: LLMCallerFn | None = None,
    k: int = K_DEFAULT,
    probe_budget: int = PROBE_TOKEN_BUDGET,
    include_subagents: bool = True,
) -> AgentAnalysisReport:
    """Benchmark ProbeAgent, MAS Orchestrator, and individual sub-agents on tasks.

    Runs each agent on the provided tasks and tracks:
      - exact accuracy vs ground truth
      - token consumption via TokenAccountant
      - wall-clock latency per call
      - routing strategy distribution

    Args:
        tasks: Tasks to evaluate.
        accountant: TokenAccountant instance for tracking spend.
        llm_caller: Pluggable caller function for deterministic testing or custom models.
        k: Token budget multiplier for MAS.
        probe_budget: Token budget for probe agent.
        include_subagents: Whether to benchmark ReAct, Debate, Reflexion individually.

    Returns:
        AgentAnalysisReport with comparative metrics across all evaluated agents.
    """
    if accountant is None:
        accountant = TokenAccountant()

    # Instantiate agents with the provided caller
    probe = ProbeAgent(n_samples=3, token_budget=probe_budget, llm_caller=llm_caller)
    orch_auto = MASOrchestrator(default_strategy="auto", llm_caller=llm_caller)
    orch_bandit = MASOrchestrator(default_strategy="bandit", llm_caller=llm_caller)

    def _run_probe(t: Task) -> tuple[str, int]:
        res = probe.run(t)
        return res.answer, res.tokens_used

    agents_to_run: dict[str, Callable[[Task], tuple[str, int]]] = {
        "ProbeAgent (CoT-SC)": _run_probe,
        "MASOrchestrator (Auto)": lambda t: orch_auto.run(t, k * probe_budget),
        "MASOrchestrator (Bandit)": lambda t: orch_bandit.run(t, k * probe_budget),
    }

    if include_subagents:
        react = ReActAgent(llm_caller=llm_caller)
        debate = DebateAgent(llm_caller=llm_caller)
        reflexion = ReflexionAgent(llm_caller=llm_caller)

        agents_to_run["ReActAgent"] = lambda t: react.run(t, k * probe_budget)
        agents_to_run["DebateAgent"] = lambda t: debate.run(t, k * probe_budget)
        agents_to_run["ReflexionAgent"] = lambda t: reflexion.run(t, k * probe_budget)

    results_by_agent: dict[str, list[dict[str, Any]]] = {name: [] for name in agents_to_run}

    for task in tasks:
        gt = task.ground_truth

        for name, agent_fn in agents_to_run.items():
            start_t = time.perf_counter()
            try:
                answer, tokens = agent_fn(task)
            except Exception as e:
                logger.warning(f"Error evaluating {name} on {task.task_id}: {e}")
                answer, tokens = "Error", 0

            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            is_correct = exact_match(answer, gt) if gt else None

            # Log to accountant
            accountant.log(
                task_id=task.task_id,
                method=name,
                stage=(
                    "mas" if "MAS" in name or "Agent" in name and "Probe" not in name else "probe"
                ),
                tokens=tokens,
                path="ESCALATE" if "MAS" in name else "N/A",
            )

            # Record strategy chosen for orchestrators
            strategy: str | None = None
            if name == "MASOrchestrator (Auto)":
                strategy = orch_auto.select_strategy(task)
            elif name == "MASOrchestrator (Bandit)":
                strategy = orch_bandit.select_strategy(task)

            results_by_agent[name].append(
                {
                    "method": name,
                    "task_id": task.task_id,
                    "predicted_answer": answer,
                    "ground_truth": gt,
                    "is_correct": is_correct,
                    "tokens_spent": tokens,
                    "latency_ms": elapsed_ms,
                    "mas_strategy": strategy,
                }
            )

    flat_records = [r for sublist in results_by_agent.values() for r in sublist]
    return analyze_eval_records(flat_records, source_name=f"benchmark_{len(tasks)}_tasks")

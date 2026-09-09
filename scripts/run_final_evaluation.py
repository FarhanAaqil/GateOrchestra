"""Final GateOrchestra evaluation runner.

This script performs the full train/validation/test evaluation flow using the
existing repository APIs without modifying any baseline or gate implementation.

Flow:
  1. Load train/val/test splits.
  2. Run CoT-SC-only and Always-MAS on train and val.
  3. Generate STOP/ESCALATE labels via apply_label_rule().
  4. Extract features and train the learned gate via train_gate().
  5. Evaluate five methods on the held-out test split.
  6. Print a single comparison table and save metrics as JSON.

Important:
  - The existing apply_label_rule() is preserved unchanged.
  - Real probe and real orchestrator are used by default.
  - No simulated_mas() fallback is silently used.
  - If the real endpoints are unavailable, the script exits with a clear blocker.
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
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.baselines.always_mas_baseline import run_always_mas_baseline
from agents.baselines.cot_sc_baseline import run_cot_sc_baseline
from agents.orchestrator.orchestrator import orchestrator as default_orchestrator
from agents.probe_agent import probe_agent as default_probe_agent
from gate.classifier import GateClassifier
from gate.feature_extractor import extract_features
from gate.random_gate import RandomGate
from gate.rule_based_gate import RuleBasedGate
from gate.train_gate import apply_label_rule, train_gate
from integration.pipeline import run_pipeline
from shared.config import (
    COT_SC_N_SAMPLES,
    COT_SC_TEMPERATURE,
    K_DEFAULT,
    LLM_PROVIDER,
    LLM_REQUEST_TIMEOUT_SECONDS,
    LOGS_DIR,
    MODEL_API_BASE,
    MODEL_NAME,
    MODELS_DIR,
    PROBE_TOKEN_BUDGET,
)
from shared.data_loader import load_split
from shared.schemas import EvalResult, GateDecision, ProbeResult, Task
from shared.token_logger import TokenAccountant

RESULTS_DIR = LOGS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_VERSION = 2
MAX_TASK_ATTEMPTS = 3
METHOD_ORDER = [
    "CoT-SC-only",
    "Always-MAS",
    "RandomGate",
    "RuleBasedGate",
    "GateOrchestra",
]


def _checkpoint_path(seed: int, split_limit: int | None) -> Path:
    """Return the checkpoint path for one deterministic evaluation identity."""
    n_label = split_limit if split_limit is not None else "all"
    return CHECKPOINTS_DIR / f"final_eval_seed_{seed}_n_{n_label}.json"


def _run_metadata(
    *, train_tasks: list[Task], val_tasks: list[Task], test_tasks: list[Task], seed: int, n: int | None
) -> dict[str, Any]:
    """Build the immutable identity used to validate a resume checkpoint."""
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "seed": seed,
        "n": n,
        "task_ids": {
            "train": [task.task_id for task in train_tasks],
            "val": [task.task_id for task in val_tasks],
            "test": [task.task_id for task in test_tasks],
        },
        "config": {
            "model_name": MODEL_NAME,
            "model_api_base": MODEL_API_BASE,
            "llm_provider": LLM_PROVIDER,
            "llm_timeout_seconds": LLM_REQUEST_TIMEOUT_SECONDS,
            "probe_token_budget": PROBE_TOKEN_BUDGET,
            "cot_sc_n_samples": COT_SC_N_SAMPLES,
            "cot_sc_temperature": COT_SC_TEMPERATURE,
            "k_default": K_DEFAULT,
            "classifier_names": ["logreg", "gbt", "mlp"],
            "k_values": [2, 3, 5],
        },
        "method_order": METHOD_ORDER,
    }


def _new_checkpoint(metadata: dict[str, Any]) -> dict[str, Any]:
    """Create an empty checkpoint state."""
    return {
        "metadata": metadata,
        "probes": {},
        "results": {},
        "gate_training": None,
        "preflight_complete": False,
        "pending_random_decisions": {},
    }


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    """Write JSON through a same-directory temporary file and atomic replace."""
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _save_checkpoint(state: dict[str, Any], path: Path) -> None:
    """Persist checkpoint state without exposing a partially written JSON file."""
    _atomic_write_json(state, path)


def _load_checkpoint(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Load and strictly validate a checkpoint before using any saved result."""
    if not path.exists():
        raise RuntimeError(f"Resume checkpoint not found: {path}")
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to read resume checkpoint {path}: {exc}") from exc

    if state.get("metadata") != metadata:
        raise RuntimeError(
            "Resume checkpoint identity does not match this run. "
            "Use the same --seed, --n, dataset ordering, configuration, and method order."
        )
    if not isinstance(state.get("probes"), dict) or not isinstance(state.get("results"), dict):
        raise RuntimeError(f"Invalid checkpoint structure: {path}")
    if not isinstance(state.get("pending_random_decisions"), dict):
        raise RuntimeError(f"Invalid RandomGate checkpoint state: {path}")
    return state


def _result_key(phase: str, method: str, task_id: str) -> str:
    return f"{phase}:{method}:{task_id}"


def _validate_probe(probe: ProbeResult, task: Task) -> ProbeResult:
    """Reject provider fallbacks so failed probes are retried, not checkpointed."""
    if probe.task_id != task.task_id:
        raise RuntimeError(
            f"Probe task mismatch: expected {task.task_id!r}, got {probe.task_id!r}"
        )
    if probe.tokens_used <= 0 or any(output.startswith("Error:") for output in probe.raw_outputs):
        raise RuntimeError(f"Probe failed for task_id={task.task_id!r}")
    return probe


def _validate_eval_result(result: EvalResult, task: Task, method: str) -> EvalResult:
    """Reject invalid or swallowed provider failures before checkpointing."""
    if result.task_id != task.task_id or result.method != method:
        raise RuntimeError(
            f"Result mismatch for task_id={task.task_id!r}: "
            f"expected method={method!r}, got task_id={result.task_id!r}, method={result.method!r}"
        )
    if method == "CoT-SC-only" and (result.probe_tokens or 0) <= 0:
        raise RuntimeError(f"Probe evaluation failed for task_id={task.task_id!r}")
    if method == "Always-MAS" and (result.mas_tokens or 0) <= 0:
        raise RuntimeError(f"MAS evaluation failed for task_id={task.task_id!r}")
    if method in {"RandomGate", "RuleBasedGate", "GateOrchestra"}:
        if (result.probe_tokens or 0) <= 0:
            raise RuntimeError(f"Probe stage failed for task_id={task.task_id!r}")
        if result.gate_decision and result.gate_decision.decision == "ESCALATE":
            if (result.mas_tokens or 0) <= 0:
                raise RuntimeError(f"MAS stage failed for task_id={task.task_id!r}")
    return result


class _ProviderFailureHandler(logging.Handler):
    """Capture swallowed provider failures emitted by the existing sub-agents."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if (
            record.name == "agents.orchestrator.sub_agents"
            and "LLM call error" in record.getMessage()
        ):
            self.messages.append(record.getMessage())


def _checked_orchestrator(task: Task, token_budget: int) -> tuple[str, int]:
    """Run MAS and surface provider failures swallowed by the existing agents."""
    logger = logging.getLogger("agents.orchestrator.sub_agents")
    handler = _ProviderFailureHandler()
    logger.addHandler(handler)
    try:
        result = default_orchestrator(task, token_budget)
    finally:
        logger.removeHandler(handler)
    if handler.messages:
        raise RuntimeError(
            f"MAS provider failure for task_id={task.task_id!r}: {handler.messages[-1]}"
        )
    return result


class _FixedGate:
    """Expose one already-selected decision without advancing a stateful gate."""

    def __init__(self, decision: Any) -> None:
        self.decision = decision

    def predict(self, features: Any, k: int, probe_tokens: int) -> Any:
        return self.decision


def _checkpoint_probe(
    task: Task,
    state: dict[str, Any],
    checkpoint_path: Path,
    probe_cache: dict[str, ProbeResult],
    probe_fn: Callable[[Task], ProbeResult],
) -> ProbeResult:
    """Load one probe from memory/checkpoint or run and persist it once."""
    if task.task_id in probe_cache:
        return probe_cache[task.task_id]
    saved_probe = state["probes"].get(task.task_id)
    if saved_probe is not None:
        probe = _validate_probe(ProbeResult.model_validate(saved_probe), task)
        probe_cache[task.task_id] = probe
        return probe

    probe = _validate_probe(probe_fn(task), task)
    state["probes"][task.task_id] = probe.model_dump(mode="json")
    probe_cache[task.task_id] = probe
    _save_checkpoint(state, checkpoint_path)
    return probe


def _run_checkpointed_baseline(
    *,
    tasks: list[Task],
    phase: str,
    method: str,
    state: dict[str, Any],
    checkpoint_path: Path,
    accountant: TokenAccountant,
    runner: Callable[[Task], EvalResult],
) -> list[EvalResult]:
    """Run one baseline task-by-task, checkpointing only successful results."""
    results: list[EvalResult] = []
    for index, task in enumerate(tasks, start=1):
        key = _result_key(phase, method, task.task_id)
        saved_result = state["results"].get(key)
        if saved_result is not None:
            result = _validate_eval_result(EvalResult.model_validate(saved_result), task, method)
            results.append(result)
            print(f"[{phase}/{method}] task {index}/{len(tasks)} resumed: {task.task_id}", flush=True)
            continue

        for attempt in range(1, MAX_TASK_ATTEMPTS + 1):
            print(
                f"[{phase}/{method}] task {index}/{len(tasks)} "
                f"starting: {task.task_id} (attempt {attempt}/{MAX_TASK_ATTEMPTS})",
                flush=True,
            )
            try:
                result = _validate_eval_result(runner(task), task, method)
            except Exception as exc:
                print(
                    f"[{phase}/{method}] task {index}/{len(tasks)} failed: "
                    f"{task.task_id}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                if attempt == MAX_TASK_ATTEMPTS:
                    raise
                continue
            state["results"][key] = result.model_dump(mode="json")
            _save_checkpoint(state, checkpoint_path)
            results.append(result)
            print(f"[{phase}/{method}] task {index}/{len(tasks)} completed: {task.task_id}", flush=True)
            break
    return results


def _run_checkpointed_pipeline(
    *,
    tasks: list[Task],
    phase: str,
    method: str,
    gate: Any,
    state: dict[str, Any],
    checkpoint_path: Path,
    get_probe: Callable[[Task], ProbeResult],
    accountant: TokenAccountant,
) -> list[EvalResult]:
    """Run one gated method task-by-task while preserving gate ordering/state."""
    results: list[EvalResult] = []
    for index, task in enumerate(tasks, start=1):
        key = _result_key(phase, method, task.task_id)
        saved_result = state["results"].get(key)
        if saved_result is not None:
            if method == "RandomGate":
                features = extract_features(task, get_probe(task))
                replayed_decision = gate.predict(
                    features, k=K_DEFAULT, probe_tokens=features.probe_tokens
                )
                saved_decision = EvalResult.model_validate(saved_result).gate_decision
                if saved_decision is None or replayed_decision.model_dump() != saved_decision.model_dump():
                    raise RuntimeError(
                        f"RandomGate decision mismatch for checkpointed task_id={task.task_id!r}"
                    )
            result = _validate_eval_result(EvalResult.model_validate(saved_result), task, method)
            results.append(result)
            print(f"[{phase}/{method}] task {index}/{len(tasks)} resumed: {task.task_id}", flush=True)
            continue

        for attempt in range(1, MAX_TASK_ATTEMPTS + 1):
            print(
                f"[{phase}/{method}] task {index}/{len(tasks)} "
                f"starting: {task.task_id} (attempt {attempt}/{MAX_TASK_ATTEMPTS})",
                flush=True,
            )
            try:
                pipeline_gate = gate
                if method == "RandomGate":
                    features = extract_features(task, get_probe(task))
                    pending_decision = state["pending_random_decisions"].get(key)
                    if pending_decision is None:
                        decision = gate.predict(
                            features, k=K_DEFAULT, probe_tokens=features.probe_tokens
                        )
                        state["pending_random_decisions"][key] = decision.model_dump(mode="json")
                        _save_checkpoint(state, checkpoint_path)
                    else:
                        decision = GateDecision.model_validate(pending_decision)
                    pipeline_gate = _FixedGate(decision)
                result = _validate_eval_result(
                    run_pipeline(
                        task,
                        pipeline_gate,
                        get_probe,
                        _checked_orchestrator,
                        accountant,
                        k=K_DEFAULT,
                        method=method,
                    ),
                    task,
                    method,
                )
            except Exception as exc:
                print(
                    f"[{phase}/{method}] task {index}/{len(tasks)} failed: "
                    f"{task.task_id}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                if attempt == MAX_TASK_ATTEMPTS:
                    raise
                continue
            if method == "RandomGate":
                state["pending_random_decisions"].pop(key, None)
            state["results"][key] = result.model_dump(mode="json")
            _save_checkpoint(state, checkpoint_path)
            results.append(result)
            print(f"[{phase}/{method}] task {index}/{len(tasks)} completed: {task.task_id}", flush=True)
            break
    return results


def _subset_tasks(tasks: list[Task], limit: int | None, seed: int) -> list[Task]:
    """Subset tasks in a reproducible way when --n is provided."""
    if limit is None:
        return tasks
    if not tasks:
        return tasks
    rng = random.Random(seed)
    idx = rng.sample(range(len(tasks)), min(limit, len(tasks)))
    return [tasks[i] for i in sorted(idx)]


def _probe_map(tasks: list[Task]) -> dict[str, object]:
    """Run the real probe agent on each task and return a lookup by task_id."""
    return {task.task_id: default_probe_agent(task) for task in tasks}


def _task_features(tasks: list[Task], probe_lookup: dict[str, object]) -> list:
    """Extract GateFeatures for the provided tasks."""
    features: list = []
    for task in tasks:
        probe = probe_lookup.get(task.task_id)
        if probe is None:
            raise ValueError(f"Missing ProbeResult for task_id={task.task_id!r}")
        features.append(extract_features(task, probe))
    return features


def _preflight_real_components(tasks: list[Task]) -> None:
    """Fail fast if the real probe/orchestrator cannot be used."""
    if not tasks:
        raise RuntimeError("No tasks available for preflight validation.")
    task = tasks[0]

    try:
        probe = default_probe_agent(task)
        _validate_probe(probe, task)
    except Exception as exc:  # pragma: no cover - depends on external service
        raise RuntimeError(
            "Real probe blocker: default_probe_agent(task) failed before evaluation. "
            f"Exact exception: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        _, tokens = _checked_orchestrator(task, max(1, K_DEFAULT * PROBE_TOKEN_BUDGET))
        if tokens <= 0:
            raise RuntimeError(f"MAS preflight returned no tokens for task_id={task.task_id!r}")
    except Exception as exc:  # pragma: no cover - depends on external service
        raise RuntimeError(
            "Real orchestrator blocker: default_orchestrator(task, token_budget) failed before evaluation. "
            f"Exact exception: {type(exc).__name__}: {exc}"
        ) from exc


def _summarize_results(results: list[EvalResult], baseline_avg_tokens: float | None = None) -> dict:
    """Compute metrics for a single method."""
    n = len(results)
    if n == 0:
        return {
            "accuracy": 0.0,
            "total_tokens": 0,
            "avg_tokens": 0.0,
            "token_savings_pct": 0.0,
            "stop_rate": 0.0,
            "escalate_rate": 0.0,
            "avg_latency_ms": 0.0,
        }

    correct = sum(1 for r in results if r.is_correct is True)
    total_tokens = sum(int(r.tokens_spent) for r in results)
    avg_tokens = total_tokens / n

    stop_count = sum(
        1 for r in results if r.gate_decision is not None and r.gate_decision.decision == "STOP"
    )
    escalate_count = sum(
        1 for r in results if r.gate_decision is not None and r.gate_decision.decision == "ESCALATE"
    )

    token_savings_pct = 0.0
    if baseline_avg_tokens is not None and baseline_avg_tokens > 0:
        token_savings_pct = (1.0 - (avg_tokens / baseline_avg_tokens)) * 100.0

    avg_latency_ms = sum(float(r.latency_ms or 0.0) for r in results) / n

    return {
        "accuracy": correct / n,
        "total_tokens": total_tokens,
        "avg_tokens": avg_tokens,
        "token_savings_pct": token_savings_pct,
        "stop_rate": stop_count / n,
        "escalate_rate": escalate_count / n,
        "avg_latency_ms": avg_latency_ms,
    }


def _print_table(rows: list[dict]) -> None:
    """Print a single comparison table."""
    headers = [
        "Method",
        "Accuracy",
        "Total Tokens",
        "Avg Tokens",
        "Token Savings",
        "STOP %",
        "ESCALATE %",
        "Avg Latency (ms)",
    ]

    print(f"\n{'=' * 130}")
    print("GateOrchestra Final Evaluation Comparison")
    print(f"{'=' * 130}")
    print(
        f"{headers[0]:<18}  {headers[1]:>9}  {headers[2]:>12}  {headers[3]:>10}  "
        f"{headers[4]:>14}  {headers[5]:>8}  {headers[6]:>12}  {headers[7]:>16}"
    )
    print("-" * 130)

    for row in rows:
        print(
            f"{row['method']:<18}  "
            f"{row['accuracy'] * 100:>8.1f}%  "
            f"{row['total_tokens']:>12d}  "
            f"{row['avg_tokens']:>9.1f}  "
            f"{row['token_savings_pct']:>+12.1f}%  "
            f"{row['stop_rate'] * 100:>7.1f}%  "
            f"{row['escalate_rate'] * 100:>10.1f}%  "
            f"{row['avg_latency_ms']:>14.1f}"
        )

    print(f"{'=' * 130}\n")


def _save_json(
    results: list[dict], output_path: Path, *, seed: int, n: int | None, dry_run: bool
) -> None:
    """Persist metrics as JSON under logs/results."""
    payload = {
        "seed": seed,
        "n": n,
        "dry_run": dry_run,
        "results": results,
    }
    _atomic_write_json(payload, output_path)


def run_evaluation(
    *, split_limit: int | None, seed: int, dry_run: bool, resume: bool = False
) -> None:
    """Run the full evaluation using existing repository APIs."""
    if dry_run:
        if resume:
            raise ValueError("--resume cannot be combined with --dry-run.")
        print("[DRY-RUN] Skipping all LLM calls and evaluation execution.")
        output_path = RESULTS_DIR / f"dry_run_seed_{seed}.json"
        _save_json([], output_path, seed=seed, n=split_limit, dry_run=True)
        print(f"[DRY-RUN] Wrote placeholder results to {output_path}")
        return

    train_tasks = _subset_tasks(load_split("train"), split_limit, seed)
    val_tasks = _subset_tasks(load_split("val"), split_limit, seed)
    test_tasks = _subset_tasks(load_split("test"), split_limit, seed)

    if not train_tasks or not val_tasks or not test_tasks:
        raise RuntimeError("One or more dataset splits is empty; cannot evaluate.")

    metadata = _run_metadata(
        train_tasks=train_tasks,
        val_tasks=val_tasks,
        test_tasks=test_tasks,
        seed=seed,
        n=split_limit,
    )
    checkpoint_path = _checkpoint_path(seed, split_limit)
    if resume:
        state = _load_checkpoint(checkpoint_path, metadata)
        print(f"[RESUME] Loaded checkpoint: {checkpoint_path}", flush=True)
    else:
        state = _new_checkpoint(metadata)

    if state["preflight_complete"]:
        print("[preflight] resumed: previously completed", flush=True)
    else:
        for attempt in range(1, MAX_TASK_ATTEMPTS + 1):
            print(
                f"[preflight] starting task {train_tasks[0].task_id} "
                f"(attempt {attempt}/{MAX_TASK_ATTEMPTS})",
                flush=True,
            )
            try:
                _preflight_real_components(train_tasks)
            except Exception as exc:
                print(
                    f"[preflight] failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                if attempt == MAX_TASK_ATTEMPTS:
                    raise
                continue
            state["preflight_complete"] = True
            _save_checkpoint(state, checkpoint_path)
            print("[preflight] completed", flush=True)
            break

    probe_cache: dict[str, ProbeResult] = {}

    def get_probe(task: Task) -> ProbeResult:
        """Return a single ProbeResult per task_id for this evaluation run/seed."""
        return _checkpoint_probe(
            task,
            state,
            checkpoint_path,
            probe_cache,
            default_probe_agent,
        )

    # Train a learned gate using the repository's existing validation logic.
    train_cot_accountant = TokenAccountant()
    train_cot_results = _run_checkpointed_baseline(
        tasks=train_tasks,
        phase="train",
        method="CoT-SC-only",
        state=state,
        checkpoint_path=checkpoint_path,
        accountant=train_cot_accountant,
        runner=lambda task: run_cot_sc_baseline(
            task, probe_fn=get_probe, accountant=train_cot_accountant
        ),
    )
    train_mas_accountant = TokenAccountant()
    train_mas_results = _run_checkpointed_baseline(
        tasks=train_tasks,
        phase="train",
        method="Always-MAS",
        state=state,
        checkpoint_path=checkpoint_path,
        accountant=train_mas_accountant,
        runner=lambda task: run_always_mas_baseline(
            task,
            orchestrator_fn=_checked_orchestrator,
            accountant=train_mas_accountant,
            token_budget=K_DEFAULT * PROBE_TOKEN_BUDGET,
        ),
    )
    train_labels = apply_label_rule(
        {r.task_id: r for r in train_cot_results},
        {r.task_id: r for r in train_mas_results},
    )
    train_probe_lookup = {task.task_id: get_probe(task) for task in train_tasks}
    train_features = [
        extract_features(task, train_probe_lookup[task.task_id])
        for task in train_tasks
        if task.task_id in train_labels
    ]
    train_label_list = [
        train_labels[task.task_id] for task in train_tasks if task.task_id in train_labels
    ]

    unique_train_labels = set(train_label_list)
    if "STOP" not in unique_train_labels or "ESCALATE" not in unique_train_labels:
        label_dist = {label: train_label_list.count(label) for label in sorted(unique_train_labels)}
        raise ValueError(
            f"Binary gate training requires both 'STOP' and 'ESCALATE' classes, "
            f"but found label distribution: {label_dist}."
        )

    val_cot_accountant = TokenAccountant()
    val_cot_results = _run_checkpointed_baseline(
        tasks=val_tasks,
        phase="val",
        method="CoT-SC-only",
        state=state,
        checkpoint_path=checkpoint_path,
        accountant=val_cot_accountant,
        runner=lambda task: run_cot_sc_baseline(
            task, probe_fn=get_probe, accountant=val_cot_accountant
        ),
    )
    val_mas_accountant = TokenAccountant()
    val_mas_results = _run_checkpointed_baseline(
        tasks=val_tasks,
        phase="val",
        method="Always-MAS",
        state=state,
        checkpoint_path=checkpoint_path,
        accountant=val_mas_accountant,
        runner=lambda task: run_always_mas_baseline(
            task,
            orchestrator_fn=_checked_orchestrator,
            accountant=val_mas_accountant,
            token_budget=K_DEFAULT * PROBE_TOKEN_BUDGET,
        ),
    )
    val_labels = apply_label_rule(
        {r.task_id: r for r in val_cot_results},
        {r.task_id: r for r in val_mas_results},
    )
    val_probe_lookup = {task.task_id: get_probe(task) for task in val_tasks}
    val_features = [
        extract_features(task, val_probe_lookup[task.task_id])
        for task in val_tasks
        if task.task_id in val_labels
    ]
    val_label_list = [val_labels[task.task_id] for task in val_tasks if task.task_id in val_labels]

    if len(train_features) == 0 or len(val_features) == 0:
        raise ValueError("No labeled train/val examples were produced; check label generation.")

    gate_model_path = MODELS_DIR / f"best_gate_seed_{seed}.pkl"
    saved_gate_training = state.get("gate_training")
    if saved_gate_training and gate_model_path.exists():
        best_gate = GateClassifier.load(gate_model_path)
        best_metrics = saved_gate_training["best_metrics"]
        print(f"[gate-training] resumed: {gate_model_path}", flush=True)
    else:
        print("[gate-training] starting", flush=True)
        best_gate, best_metrics = train_gate(
            train_features,
            train_label_list,
            val_features,
            val_label_list,
            classifier_names=["logreg", "gbt", "mlp"],
            k_values=[2, 3, 5],
            save_path=gate_model_path,
        )
        state["gate_training"] = {
            "model_path": str(gate_model_path),
            "best_metrics": best_metrics,
        }
        _save_checkpoint(state, checkpoint_path)
        print("[gate-training] completed", flush=True)

    # Evaluate the test set across all requested methods.
    start_time = time.perf_counter()
    test_cot_accountant = TokenAccountant()
    test_cot_results = _run_checkpointed_baseline(
        tasks=test_tasks,
        phase="test",
        method="CoT-SC-only",
        state=state,
        checkpoint_path=checkpoint_path,
        accountant=test_cot_accountant,
        runner=lambda task: run_cot_sc_baseline(
            task, probe_fn=get_probe, accountant=test_cot_accountant
        ),
    )
    test_mas_accountant = TokenAccountant()
    test_mas_results = _run_checkpointed_baseline(
        tasks=test_tasks,
        phase="test",
        method="Always-MAS",
        state=state,
        checkpoint_path=checkpoint_path,
        accountant=test_mas_accountant,
        runner=lambda task: run_always_mas_baseline(
            task,
            orchestrator_fn=_checked_orchestrator,
            accountant=test_mas_accountant,
            token_budget=K_DEFAULT * PROBE_TOKEN_BUDGET,
        ),
    )
    baseline_avg_tokens = sum(r.tokens_spent for r in test_mas_results) / len(test_mas_results)

    method_results: list[dict] = []
    results_by_method: dict[str, list[EvalResult]] = {
        "CoT-SC-only": test_cot_results,
        "Always-MAS": test_mas_results,
    }

    random_gate = RandomGate(escalation_rate=0.5, seed=seed)
    rule_gate = RuleBasedGate()
    test_accountant_random = TokenAccountant()
    results_by_method["RandomGate"] = _run_checkpointed_pipeline(
        tasks=test_tasks,
        phase="test",
        method="RandomGate",
        gate=random_gate,
        state=state,
        checkpoint_path=checkpoint_path,
        get_probe=get_probe,
        accountant=test_accountant_random,
    )

    test_accountant_rule = TokenAccountant()
    results_by_method["RuleBasedGate"] = _run_checkpointed_pipeline(
        tasks=test_tasks,
        phase="test",
        method="RuleBasedGate",
        gate=rule_gate,
        state=state,
        checkpoint_path=checkpoint_path,
        get_probe=get_probe,
        accountant=test_accountant_rule,
    )

    test_accountant_gate = TokenAccountant()
    results_by_method["GateOrchestra"] = _run_checkpointed_pipeline(
        tasks=test_tasks,
        phase="test",
        method="GateOrchestra",
        gate=best_gate,
        state=state,
        checkpoint_path=checkpoint_path,
        get_probe=get_probe,
        accountant=test_accountant_gate,
    )

    for method_name, results in results_by_method.items():
        summary = _summarize_results(results, baseline_avg_tokens=baseline_avg_tokens)
        method_results.append({"method": method_name, **summary})

    _print_table(method_results)

    output_path = RESULTS_DIR / f"final_eval_seed_{seed}_n_{split_limit or 'all'}.json"
    _save_json(method_results, output_path, seed=seed, n=split_limit, dry_run=False)
    elapsed = time.perf_counter() - start_time
    print(f"[OK] Complete evaluation finished in {elapsed:.2f}s")
    print(f"[OK] Results written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Final GateOrchestra evaluation runner")
    parser.add_argument(
        "--n", type=int, default=None, help="Subset size per split; defaults to full split"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible subsets and random gate"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running the real LLM-backed evaluation.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the checkpoint for the selected seed and split size.",
    )
    args = parser.parse_args()

    run_evaluation(split_limit=args.n, seed=args.seed, dry_run=args.dry_run, resume=args.resume)


if __name__ == "__main__":
    main()

"""
shared/config.py
================
Single source of truth for all GateOrchestra hyperparameters and paths.

RULE: No magic numbers anywhere else in the codebase.
      Import from here. If you need a new constant, add it here first.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# Paths

ROOT_DIR: Path = Path(__file__).parent.parent
DATASET_DIR: Path = ROOT_DIR / "dataset" / "masbench_mini"
LOGS_DIR: Path = ROOT_DIR / "logs"
CONFIGS_DIR: Path = ROOT_DIR / "configs"
MODELS_DIR: Path = CONFIGS_DIR / "models"
WEEK8_BANDIT_STATE_PATH: Path = LOGS_DIR / "week8_bandit_state.json"

# Auto-create runtime dirs (non-code artifacts)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

# LLM / Providers (Ollama & Groq)

LLM_PROVIDER: str = os.getenv("GATE_LLM_PROVIDER", "ollama").lower()

# Ollama / Local settings (Default)
MODEL_NAME: str = os.getenv("GATE_MODEL_NAME", "qwen2.5:3b")
"""Ollama model. qwen2.5:3b fits a 4 GB GPU; the Blueprint names qwen2.5:7b-instruct."""
MODEL_API_BASE: str = os.getenv("GATE_API_BASE", "http://localhost:11434")  # Ollama default

# Groq Cloud settings (Optional)
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_NAME: str = os.getenv("GROQ_MODEL_NAME", "qwen/qwen3.8-27b")
GROQ_API_BASE: str = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
LLM_REQUEST_TIMEOUT_SECONDS: float = float(os.getenv("GATE_LLM_TIMEOUT", "300"))
ACTIVE_MODEL_NAME: str = GROQ_MODEL_NAME if LLM_PROVIDER == "groq" else MODEL_NAME
"""The model the benchmark actually calls, so run records name the right one."""
ACTIVE_API_BASE: str = GROQ_API_BASE if LLM_PROVIDER == "groq" else MODEL_API_BASE
PROBE_TOKEN_BUDGET: int = 1000
"""Completion-token budget for one whole CoT-SC probe (all samples together).

Split evenly across samples: 1000 // 3 = 333 completion tokens per sample,
which is what every recorded real run used. Reported probe tokens are the
provider's true totals (prompt + completion), so they exceed this figure.
"""

COT_SC_N_SAMPLES: int = 3  # Number of CoT-SC samples per probe run
COT_SC_TEMPERATURE: float = 0.7  # Sampling temperature for CoT-SC

# Dataset Splits

TRAIN_SPLIT: float = 0.60  # ~90 tasks (out of 150)
VAL_SPLIT: float = 0.20  # ~30 tasks
TEST_SPLIT: float = 0.20  # ~30 tasks  ← held-out, do not touch until Week 10
RANDOM_SEED: int = 42

FINAL_SEEDS: tuple[int, ...] = (42, 123)
"""Seeds of the final benchmark. The first one calibrates the gate and router; every seed
repeats the stochastic test evaluation. Both scripts/final_benchmark.py and
scripts/verify_final_results.py read this, so they cannot disagree."""

RUN_TAG: str = re.sub(r"[^A-Za-z0-9_.-]", "", os.getenv("GATE_RUN_TAG", ""))
"""Names a benchmark run on another model (for example ``groq-gpt-oss-120b``). With a tag
set, the run writes its results, checkpoints, gate, router and reports into folders of
its own and never touches the default (untagged) Qwen2.5-3B artifacts, and the API and
assistant never load its gate. Empty by default."""

FINAL_RESULTS_DIR: Path = ROOT_DIR / "results" / "real" / (RUN_TAG or "final")
FINAL_CHECKPOINTS_DIR: Path = LOGS_DIR / "results" / "checkpoints" / RUN_TAG
FINAL_MODELS_DIR: Path = MODELS_DIR / RUN_TAG
FINAL_REPORTS_DIR: Path = ROOT_DIR / "reports" / RUN_TAG

# Gate Hyperparameters

GATE_ESCALATE_THRESHOLD: float = float(os.getenv("GATE_ESCALATE_THRESHOLD", "0.15"))
"""Probability threshold for ESCALATE decision to handle class imbalance."""

TAU_ACC: float = 0.05
"""Accuracy threshold for gate labeling.

Label = ESCALATE  iff  MAS_correct AND NOT CoT-SC_correct
                  (i.e. MAS wins on this task where CoT-SC fails)
Label = STOP      otherwise

τ_acc is swept during validation. Default = 0.05 per the blueprint.
"""

TAU_ACC_SWEEP: list[float] = [0.03, 0.05, 0.08]
"""Values of τ_acc to sweep during validation (Week 8–9)."""

MAS_ROUTER: str = os.getenv("GATE_MAS_ROUTER", "bandit").lower()
"""How the MAS picks a sub-agent: "bandit" (LinUCB, learned during benchmark
calibration; falls back to "auto" until a trained state exists), "auto"
(text heuristic) or a fixed agent name ("react", "debate", "reflexion")."""

K_DEFAULT: int = 3
"""Default token budget multiplier: MAS budget = k × probe_tokens."""

K_VALUES: list[int] = [2, 3, 5]
"""Values of k to sweep during validation (Week 8–9)."""

N_REPEATS: int = int(os.getenv("GATE_N_REPEATS", "3"))
"""Number of independent train/validation repeats used for label aggregation."""

# Rule-Based Gate Thresholds (Week 2 / Day 10)

RULE_CONSISTENCY_STOP_THRESHOLD: float = 0.8
"""If consistency_score ≥ this AND entity_count < RULE_ENTITY_STOP_THRESHOLD → STOP."""

RULE_ENTITY_STOP_THRESHOLD: int = 3
"""Max entity count for the STOP rule."""

RULE_DEPTH_ESCALATE_THRESHOLD: float = 3.0
"""If estimated_depth ≥ this → ESCALATE."""

RULE_PARALLEL_ESCALATE_THRESHOLD: float = 2.0
"""If estimated_parallel ≥ this → ESCALATE."""

# Feature Extraction

SPACY_MODEL: str = "en_core_web_sm"
USE_SPACY: bool = True  # Set False to use regex fallback (slower but no dep)

# Training

CLASSIFIER_NAMES: list[str] = ["logreg", "gbt", "mlp"]
DEFAULT_CLASSIFIER: str = "gbt"

BEST_MODEL_PATH: Path = MODELS_DIR / "best_gate.pkl"
EXPERIMENT_LOG_PATH: Path = LOGS_DIR / "experiment_log.jsonl"
TOKEN_LOG_PATH: Path = LOGS_DIR / "token_log.json"

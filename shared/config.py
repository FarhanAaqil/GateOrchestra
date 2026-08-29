"""
shared/config.py
================
Single source of truth for all GateOrchestra hyperparameters and paths.

RULE: No magic numbers anywhere else in the codebase.
      Import from here. If you need a new constant, add it here first.
"""

from __future__ import annotations

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

ROOT_DIR: Path = Path(__file__).parent.parent
DATASET_DIR: Path = ROOT_DIR / "dataset" / "masbench_mini"
LOGS_DIR: Path = ROOT_DIR / "logs"
CONFIGS_DIR: Path = ROOT_DIR / "configs"
MODELS_DIR: Path = CONFIGS_DIR / "models"

# Auto-create runtime dirs (non-code artifacts)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# LLM / Providers (Ollama & Groq)
# ─────────────────────────────────────────────────────────────────────────────

LLM_PROVIDER: str = os.getenv("GATE_LLM_PROVIDER", "ollama").lower()

# Ollama / Local settings (Default)
MODEL_NAME: str = os.getenv("GATE_MODEL_NAME", "Qwen2.5-7B-Instruct")
MODEL_API_BASE: str = os.getenv("GATE_API_BASE", "http://localhost:11434")  # Ollama default

# Groq Cloud settings (Optional)
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_NAME: str = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
GROQ_API_BASE: str = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")

PROBE_TOKEN_BUDGET: int = 500   # Max tokens the probe may spend per task
COT_SC_N_SAMPLES: int = 5       # Number of CoT-SC samples per probe run
COT_SC_TEMPERATURE: float = 0.7  # Sampling temperature for CoT-SC

# ─────────────────────────────────────────────────────────────────────────────
# Dataset Splits
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_SPLIT: float = 0.60   # ~90 tasks (out of 150)
VAL_SPLIT: float = 0.20     # ~30 tasks
TEST_SPLIT: float = 0.20    # ~30 tasks  ← held-out, do not touch until Week 10
RANDOM_SEED: int = 42

# ─────────────────────────────────────────────────────────────────────────────
# Gate Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

TAU_ACC: float = 0.05
"""Accuracy threshold for gate labeling.

Label = ESCALATE  iff  MAS_correct AND NOT CoT-SC_correct
                  (i.e. MAS wins on this task where CoT-SC fails)
Label = STOP      otherwise

τ_acc is swept during validation. Default = 0.05 per the blueprint.
"""

TAU_ACC_SWEEP: list[float] = [0.03, 0.05, 0.08]
"""Values of τ_acc to sweep during validation (Week 8–9)."""

K_DEFAULT: int = 3
"""Default token budget multiplier: MAS budget = k × probe_tokens."""

K_VALUES: list[int] = [2, 3, 5]
"""Values of k to sweep during validation (Week 8–9)."""

# ─────────────────────────────────────────────────────────────────────────────
# Rule-Based Gate Thresholds (Week 2 / Day 10)
# ─────────────────────────────────────────────────────────────────────────────

RULE_CONSISTENCY_STOP_THRESHOLD: float = 0.8
"""If consistency_score ≥ this AND entity_count < RULE_ENTITY_STOP_THRESHOLD → STOP."""

RULE_ENTITY_STOP_THRESHOLD: int = 3
"""Max entity count for the STOP rule."""

RULE_DEPTH_ESCALATE_THRESHOLD: float = 3.0
"""If estimated_depth ≥ this → ESCALATE."""

RULE_PARALLEL_ESCALATE_THRESHOLD: float = 2.0
"""If estimated_parallel ≥ this → ESCALATE."""

# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

SPACY_MODEL: str = "en_core_web_sm"
USE_SPACY: bool = True  # Set False to use regex fallback (slower but no dep)

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

CLASSIFIER_NAMES: list[str] = ["logreg", "gbt", "mlp"]
DEFAULT_CLASSIFIER: str = "gbt"

BEST_MODEL_PATH: Path = MODELS_DIR / "best_gate.pkl"
EXPERIMENT_LOG_PATH: Path = LOGS_DIR / "experiment_log.jsonl"
TOKEN_LOG_PATH: Path = LOGS_DIR / "token_log.json"

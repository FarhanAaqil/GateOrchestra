"""configs/default.yaml must mirror shared/config.py exactly."""

from __future__ import annotations

import os

import pytest
import yaml

from shared import config


@pytest.fixture(scope="module")
def default_yaml() -> dict:
    return yaml.safe_load((config.CONFIGS_DIR / "default.yaml").read_text(encoding="utf-8"))


def test_yaml_matches_config(default_yaml: dict) -> None:
    probe = default_yaml["probe"]
    gate = default_yaml["gate"]
    assert probe["token_budget"] == config.PROBE_TOKEN_BUDGET
    assert probe["n_samples"] == config.COT_SC_N_SAMPLES
    assert probe["temperature"] == config.COT_SC_TEMPERATURE
    assert gate["tau_acc"] == config.TAU_ACC
    assert gate["k"] == config.K_DEFAULT
    assert gate["classifier"] == config.DEFAULT_CLASSIFIER
    assert gate["tau_acc_sweep"] == config.TAU_ACC_SWEEP
    assert gate["k_sweep"] == config.K_VALUES
    assert gate["rule_consistency_stop"] == config.RULE_CONSISTENCY_STOP_THRESHOLD
    assert gate["rule_entity_stop"] == config.RULE_ENTITY_STOP_THRESHOLD
    assert gate["rule_depth_escalate"] == config.RULE_DEPTH_ESCALATE_THRESHOLD
    assert gate["rule_parallel_escalate"] == config.RULE_PARALLEL_ESCALATE_THRESHOLD
    assert default_yaml["dataset"]["random_seed"] == config.RANDOM_SEED
    assert default_yaml["features"]["spacy_model"] == config.SPACY_MODEL


def test_yaml_model_defaults_match_config_defaults(default_yaml: dict) -> None:
    # Only meaningful when the environment does not override the defaults.
    model = default_yaml["model"]
    if "GATE_MODEL_NAME" not in os.environ:
        assert model["name"] == config.MODEL_NAME
    if "GROQ_MODEL_NAME" not in os.environ:
        assert model["groq_name"] == config.GROQ_MODEL_NAME


def test_env_example_names_the_same_models(default_yaml: dict) -> None:
    env_example = (config.ROOT_DIR / ".env.example").read_text(encoding="utf-8")
    values = dict(
        line.split("=", 1) for line in env_example.splitlines() if "=" in line and line[0] != "#"
    )
    assert values["GATE_MODEL_NAME"] == default_yaml["model"]["name"]
    assert values["GROQ_MODEL_NAME"] == default_yaml["model"]["groq_name"]

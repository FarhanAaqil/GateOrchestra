"""A tagged benchmark run writes into folders of its own and never touches the default run."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _paths(**env: str) -> dict[str, str]:
    code = (
        "import json; from shared import config as c; "
        "print(json.dumps({k: str(getattr(c, k)) for k in "
        "['RUN_TAG','FINAL_RESULTS_DIR','FINAL_CHECKPOINTS_DIR','FINAL_MODELS_DIR',"
        "'FINAL_REPORTS_DIR','ACTIVE_MODEL_NAME']}))"
    )
    clean = {k: v for k, v in os.environ.items() if not k.startswith("GATE_")}
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={**clean, "GATE_RUN_TAG": "", "GATE_LLM_PROVIDER": "ollama", **env},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out)


def test_untagged_run_keeps_the_original_locations() -> None:
    paths = _paths()
    assert paths["RUN_TAG"] == ""
    assert Path(paths["FINAL_RESULTS_DIR"]) == ROOT / "results" / "real" / "final"
    assert Path(paths["FINAL_CHECKPOINTS_DIR"]) == ROOT / "logs" / "results" / "checkpoints"
    assert Path(paths["FINAL_MODELS_DIR"]) == ROOT / "configs" / "models"
    assert Path(paths["FINAL_REPORTS_DIR"]) == ROOT / "reports"


def test_tagged_run_gets_folders_of_its_own() -> None:
    paths = _paths(GATE_RUN_TAG="groq-gpt-oss-120b")
    assert Path(paths["FINAL_RESULTS_DIR"]) == ROOT / "results" / "real" / "groq-gpt-oss-120b"
    for key in ("FINAL_CHECKPOINTS_DIR", "FINAL_MODELS_DIR", "FINAL_REPORTS_DIR"):
        assert Path(paths[key]).name == "groq-gpt-oss-120b"


def test_tag_is_sanitised_so_it_cannot_escape_the_project() -> None:
    paths = _paths(GATE_RUN_TAG="../../evil/tag")
    assert paths["RUN_TAG"] == "....eviltag"
    assert ROOT in Path(paths["FINAL_RESULTS_DIR"]).resolve().parents


def test_groq_run_records_the_groq_model_not_the_local_one() -> None:
    paths = _paths(GATE_LLM_PROVIDER="groq", GROQ_MODEL_NAME="openai/gpt-oss-120b")
    assert paths["ACTIVE_MODEL_NAME"] == "openai/gpt-oss-120b"
    assert _paths(GATE_MODEL_NAME="qwen2.5:3b")["ACTIVE_MODEL_NAME"] == "qwen2.5:3b"

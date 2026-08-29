# GateOrchestra

> **Token-Budget-Calibrated Gating Layer for Learned Multi-Agent Orchestration**
> Final-year AIML Capstone — 4-person team

---

## What It Does

GateOrchestra places a lightweight, trainable **gate** in front of a Multi-Agent System (MAS) orchestrator. Before spending expensive multi-agent compute, the gate decides — using only pre-execution signals — whether to:

- **STOP** → return the cheap CoT-SC probe answer (single agent), or
- **ESCALATE** → invoke the full MAS orchestrator, capped at `k × probe_tokens`

**Research goal:** ≥40% token savings vs. Always-MAS, at ≤2-point accuracy cost.

---

## Architecture

```
Task
 │
 ▼
[Probe Agent]  ──── CoT-SC (N samples) ──► ProbeResult
 │                                          (consistency_score, tokens_used)
 ▼
[Feature Extractor]  ──► GateFeatures
 │                        (entity_count, clause_count, consistency, ...)
 ▼
[Gate Classifier]  ──► GateDecision (STOP | ESCALATE)
 │
 ├── STOP     ──► return probe answer         ─┐
 │                                              ├──► EvalResult ──► TokenLog
 └── ESCALATE ──► [MAS Orchestrator]           ─┘
                   (budget = k × probe_tokens)
```

---

## Repo Structure

```
gateorchestra/
├── shared/          # ← Person 3: frozen contract layer (import this everywhere)
│   ├── schemas.py   #   Pydantic models: Task, ProbeResult, GateFeatures, GateDecision, EvalResult
│   ├── config.py    #   All hyperparameters — single source of truth
│   └── token_logger.py  # Thread-safe token accounting
│
├── dataset/         # ← Person 1
├── agents/          # ← Person 2
├── gate/            # ← Person 3
│   ├── feature_extractor.py
│   ├── classifier.py       # LogReg / GBT / MLP
│   ├── rule_based_gate.py
│   ├── random_gate.py
│   └── train_gate.py
├── evaluation/      # ← Person 4
├── integration/     # ← Person 3: wires all modules together
│   └── pipeline.py
│
├── tests/
│   ├── mocks/       # Mock P1/P2/P4 modules for offline testing
│   ├── test_shared_schemas.py
│   ├── test_gate.py
│   └── test_pipeline.py
├── scripts/
│   └── demo_run.py  # Quick end-to-end demo
└── configs/
    └── default.yaml
```

---

## Quickstart

```bash
# 1. Clone and install
git clone <repo-url>
cd gateorchestra
pip install -e ".[dev]"
python -m spacy download en_core_web_sm

# 2. Run tests
pytest

# 3. Run demo (no LLM needed — uses mocks)
python scripts/demo_run.py
```

---

## Team Interface Contract

All 4 team members code against `shared/schemas.py`. **Do not change these models without a team PR review.**

| Model | Produced by | Consumed by |
|---|---|---|
| `Task` | Person 1 (`dataset/`) | Everyone |
| `ProbeResult` | Person 2 (`agents/probe_agent.py`) | Person 3 |
| `GateFeatures` | Person 3 (`gate/feature_extractor.py`) | Person 3 |
| `GateDecision` | Person 3 (`gate/`) | Person 3, Person 4 |
| `EvalResult` | Person 3 (`integration/pipeline.py`) | Person 4 (`evaluation/`) |

---

## Git Workflow

- `main` — protected, PRs only, must pass CI
- Branches: `person1/dataset`, `person2/agents`, `person3/gate`, `person4/eval`
- Every PR must pass: `pytest` + `ruff` + `mypy`

---

## LLM Provider Configuration (Local Ollama vs Cloud Groq)

GateOrchestra supports both local open-weight models via **Ollama** and cloud fast inference via **Groq**.

### Option A: Local Ollama (Default)
Runs locally with zero external API fees.
```bash
# 1. Start Ollama and pull Qwen2.5
ollama serve
ollama pull qwen2.5:7b-instruct

# 2. Environment variables (Optional — these are defaults)
export GATE_LLM_PROVIDER=ollama
export GATE_MODEL_NAME=Qwen2.5-7B-Instruct
export GATE_API_BASE=http://localhost:11434
```

### Option B: Cloud Groq API (Optional)
Runs cloud inference using Groq's high-speed LPU endpoints.
```bash
# 1. Set your Groq API key (never commit this key to git)
export GROQ_API_KEY="gsk_your_groq_api_key_here"

# 2. Switch provider to Groq
export GATE_LLM_PROVIDER=groq
export GROQ_MODEL_NAME=llama-3.3-70b-versatile    # or llama-3.1-8b-instant
export GROQ_API_BASE=https://api.groq.com/openai/v1
```

On Windows PowerShell:
```powershell
$env:GROQ_API_KEY = "gsk_your_groq_api_key_here"
$env:GATE_LLM_PROVIDER = "groq"
$env:GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
```

---

## Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `TAU_ACC` | 0.05 | Accuracy threshold for ESCALATE labeling |
| `K` | 3 | MAS token budget = k × probe_tokens |
| `PROBE_TOKEN_BUDGET` | 500 | Max tokens per CoT-SC probe |
| `COT_SC_N_SAMPLES` | 5 | Number of CoT-SC samples |

All in `shared/config.py` and `configs/default.yaml`.

# GateOrchestra

> **Token-Budget-Calibrated Gating Layer for Learned Multi-Agent Orchestration**
> Final-year AIML Capstone â€” 4-person team

---

## What It Does

GateOrchestra places a lightweight, trainable **gate** in front of a Multi-Agent System (MAS) orchestrator. Before spending expensive multi-agent compute, the gate decides â€” using only pre-execution signals â€” whether to:

- **STOP** â†’ return the cheap CoT-SC probe answer (single agent), or
- **ESCALATE** â†’ invoke the full MAS orchestrator, capped at `k Ã— probe_tokens`

**Research goal:** â‰¥40% token savings vs. Always-MAS, at â‰¤2-point accuracy cost.

---

## Architecture

```
Task
 â”‚
 â–¼
[Probe Agent]  â”€â”€â”€â”€ CoT-SC (N samples) â”€â”€â–º ProbeResult
 â”‚                                          (consistency_score, tokens_used)
 â–¼
[Feature Extractor]  â”€â”€â–º GateFeatures
 â”‚                        (entity_count, clause_count, consistency, ...)
 â–¼
[Gate Classifier]  â”€â”€â–º GateDecision (STOP | ESCALATE)
 â”‚
 â”œâ”€â”€ STOP     â”€â”€â–º return probe answer         â”€â”
 â”‚                                              â”œâ”€â”€â–º EvalResult â”€â”€â–º TokenLog
 â””â”€â”€ ESCALATE â”€â”€â–º [MAS Orchestrator]           â”€â”˜
                   (budget = k Ã— probe_tokens)
```

---

## Repo Structure

```
gateorchestra/
â”œâ”€â”€ shared/          # â† Person 3: frozen contract layer (import this everywhere)
â”‚   â”œâ”€â”€ schemas.py   #   Pydantic models: Task, ProbeResult, GateFeatures, GateDecision, EvalResult
â”‚   â”œâ”€â”€ config.py    #   All hyperparameters â€” single source of truth
â”‚   â””â”€â”€ token_logger.py  # Thread-safe token accounting
â”‚
â”œâ”€â”€ dataset/         # â† Person 1
â”œâ”€â”€ agents/          # â† Person 2
â”œâ”€â”€ gate/            # â† Person 3
â”‚   â”œâ”€â”€ feature_extractor.py
â”‚   â”œâ”€â”€ classifier.py       # LogReg / GBT / MLP
â”‚   â”œâ”€â”€ rule_based_gate.py
â”‚   â”œâ”€â”€ random_gate.py
â”‚   â””â”€â”€ train_gate.py
â”œâ”€â”€ evaluation/      # â† Person 4
â”œâ”€â”€ integration/     # â† Person 3: wires all modules together
â”‚   â””â”€â”€ pipeline.py
â”‚
â”œâ”€â”€ tests/
â”‚   â”œâ”€â”€ mocks/       # Mock P1/P2/P4 modules for offline testing
â”‚   â”œâ”€â”€ test_shared_schemas.py
â”‚   â”œâ”€â”€ test_gate.py
â”‚   â””â”€â”€ test_pipeline.py
â”œâ”€â”€ scripts/
â”‚   â””â”€â”€ demo_run.py  # Quick end-to-end demo
â””â”€â”€ configs/
    â””â”€â”€ default.yaml
```

---

## Quickstart

```bash
# 1. Clone and install
git clone <repo-url>
cd gateorchestra
pip install -e ".[dev]"
python -m spacy download en_core_web_sm

# 2. Run test suite (422+ unit & integration tests)
pytest

# 3. Run Capstone Live Demo (100% deterministic offline fail-safe mode)
python scripts/week8_demo.py --mode mock --n 5

# 4. Or launch the FastAPI Backend + React Dashboard
uvicorn api.main:app --reload --port 8000
# In a separate terminal:
cd frontend-react && npm run dev
```

---

## Final Capstone Empirical Results (Test Split)

> **⚠️ Status: Simulation-Derived Numbers — Real Evaluation Pending**
>
> The numbers below were produced under `execution_mode = "mock"` (simulated LLM calls).
> They reflect system logic but **are not empirical results from real Groq/Ollama API calls**.
> Real evaluation results will replace this table once the full benchmark pipeline is run.
> Do not cite these numbers in the final report or presentation.
> See `results/simulated/` for the source artifact.

Evaluated across 3 seeds (`42`, `123`, `999`) on the held-out `masbench_mini` test split (32 tasks, balanced across four strata: `hotpotqa_style`, `musique_style`, `template_arithmetic`, `template_comparison`).

| Method | Type | Accuracy (%) | Token Savings vs Always-MAS | Avg Tokens / Task | STOP Rate (%) |
|---|---|---|---|---|---|
| **GateOrchestra** | **Learned GBT Gate** | **82.29% Â± 7.86%** | **78.37% Â± 1.79%** | **226.0** | **96.9%** |
| RuleBasedGate | Heuristic Gate | 81.25% Â± 13.62% | 66.88% Â± 0.49% | 345.9 | 76.0% |
| CoT-SC-only | Single-Agent Baseline | 79.17% Â± 1.80% | 78.20% Â± 1.43% | 227.9 | 100.0% |
| RandomGate | Random Baseline ($p=0.5$) | 71.88% Â± 8.27% | 51.50% Â± 6.72% | 506.7 | 46.9% |
| Always-MAS | Un-gated Multi-Agent System | 67.71% Â± 9.55% | 0.00% Â± 0.00% | 1,044.8 | 0.0% |

- **RQ1 (Token Savings):** **78.37% Â± 1.79%** token reduction vs Always-MAS (**target $\ge 40\%$ comfortably exceeded**).
- **RQ2 (Accuracy Preservation):** **+14.58 percentage points** higher accuracy than Always-MAS (82.29% vs 67.71%) by avoiding multi-agent overthinking.
- **RQ3 (Pareto Dominance):** Complete Pareto frontier dominance across budget multipliers $k \in \{2, 3, 5\}$ and sample counts $N \in \{3, 5, 7\}$.

ðŸ“– **Full Report:** See [Capstone Technical Report](reports/capstone_technical_report_person3.md) and [Presentation & Release Guide](docs/WEEK8_RELEASE_GUIDE.md).

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

- `main` â€” protected, PRs only, must pass CI
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

# 2. Environment variables (Optional â€” these are defaults)
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
| `K` | 3 | MAS token budget = k Ã— probe_tokens |
| `PROBE_TOKEN_BUDGET` | 500 | Max tokens per CoT-SC probe |
| `COT_SC_N_SAMPLES` | 5 | Number of CoT-SC samples |

All in `shared/config.py` and `configs/default.yaml`.



---

## Canonical Commands (Rescue & Defense)

### 1. Canonical Empirical Benchmark & Verification
```bash
# Run real LLM evaluation against Groq and generate master_results.json
python scripts/final_benchmark.py --n 5

# Automated result integrity audit (strict validation)
python scripts/verify_final_results.py
```

### 2. Live & Offline Demonstration Suite
```bash
# Live demonstration querying Groq with live token accounting
python scripts/demo.py --mode live --scenario easy
python scripts/demo.py --mode live --scenario hard
python scripts/demo.py --mode live --scenario recovery

# Offline presentation fallback (100% deterministic, no external API calls)
python scripts/demo.py --mode simulation
```

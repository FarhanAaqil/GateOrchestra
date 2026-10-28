# GateOrchestra

> **Token-Budget-Calibrated Gating Layer for Learned Multi-Agent Orchestration**
> AIML Capstone Project

---

## What It Does

GateOrchestra places a lightweight, trainable **gate** in front of a Multi-Agent System (MAS) orchestrator. Before spending expensive multi-agent compute, the gate decides — using only pre-execution signals — whether to:

- **STOP** → return the inexpensive CoT-SC probe answer (single agent), or
- **ESCALATE** → invoke the full MAS orchestrator, capped at `k × probe_tokens`.

**Research Goal:** ≥40% token savings vs. Always-MAS at ≤2 points accuracy cost, measured under equal token budgets (see the [Blueprint](Blueprint) hypotheses H1/RQ1–RQ3).

---

## Results Status

> **There is no verified real-LLM result yet.** Treat any other figure you find in older reports as simulation.

| Evidence | What it shows |
| :--- | :--- |
| **Simulation** (`results/simulated/`, `mode: mock`) | Not evidence. The simulator builds probe samples from the ground truth and models Always-MAS as always correct. Figures such as "78.37% savings / 82.29% accuracy" come from here. |
| **Real-LLM smoke run** (`results/real/final/master_results.json`, Groq `qwen/qwen3.8-27b`, **5 test tasks, 1 seed**) | No method is distinguishable: every 95% Wilson interval spans about 12–77%, so "40% vs 20%" is a difference of one task. The learned gate stopped on 100% of tasks, so its row is identical to CoT-SC-only. It was recorded before the measurement fixes below and **fails** `scripts/verify_final_results.py`. |
| **Final benchmark** (32 test tasks × seeds 42, 123, 999) | **Not run yet.** Run `python scripts/final_benchmark.py`, then `python scripts/verify_final_results.py`. |

Smoke-run numbers, kept for transparency only:

| Method | Accuracy | Wilson 95% CI | Avg Tokens / Task | STOP Rate |
| :--- | :--- | :--- | :--- | :--- |
| CoT-SC-only | 40.0% (2/5) | [11.8%, 76.9%] | 782.4 | — |
| Always-MAS | 20.0% (1/5) | [3.6%, 62.4%] | 788.2 | — |
| RandomGate | 20.0% (1/5) | [3.6%, 62.4%] | 1,354.2 | 40.0% |
| RuleBasedGate | 40.0% (2/5) | [11.8%, 76.9%] | 935.8 | 80.0% |
| GateOrchestra | 40.0% (2/5) | [11.8%, 76.9%] | 782.4 | 100.0% |

Two things to know before reading any future result:
- **The savings premise needs MAS to be much more expensive than the probe.** In the smoke run, Always-MAS (788 tokens) cost about the same as the probe (782), so no gate could save much.
- **"The gate learned to always stop" is a legitimate, reportable negative result.** Calibration reports a degenerate constant gate rather than substituting another one.

### Fixes made before the final benchmark

| Area | Before | Now |
| :--- | :--- | :--- |
| Token accounting | Spend clipped with `min(tokens, budget)`, hiding prompt tokens | True provider totals recorded; budget enforced *before* each call |
| Failed LLM calls | Sub-agents returned a fake answer with 0 tokens; failed probe samples voted with consistency 1.0 | `SubAgentError` / `ProbeError` are raised; failed samples lower consistency |
| Majority vote | Jaccard ≥ 0.5 merged "30 / 60 / 45 square centimeters" | Uses the grader's equivalence (numbers compared by value) |
| Grading | Two graders; "$500" correct for "$5.00", "15" correct for "5" | One grader (`shared/grading.py`) with numeric/unit handling and curated aliases |
| Dataset | 14 tasks wrong, stale or unanswerable (e.g. the U.S. Capitol dome is not gold) | Corrected by `scripts/apply_dataset_corrections.py`; 82 tasks have answer aliases |
| MAS prompts | Debate/Reflexion dropped the context and previous draft after round 1 | Context and previous draft carried through every step |
| Routing | "auto" router read dataset depth/parallel labels | Text-only heuristic (LinUCB optional), no labels |
| Budgets | Always-MAS capped at 1,500; escalations at k × probe tokens | Every MAS call capped at k × the paired probe's tokens |
| RandomGate | Fixed 50%; the API rebuilt it per request, so it always stopped | Matched to the learned gate's validation escalation rate; one persistent gate in the API |
| Gate training | Single-class labels → loaded a gate trained on *simulated* data and reported hardcoded metrics | Degenerate constant gate, scored on real validation data and flagged |
| Verifier | Hashed no files; seed and "no mock" checks missing | Checks seeds, dataset SHA-256, per-task records, mock markers, and recomputes aggregates |
| Statistics | Accuracy only | Wilson CIs per method, exact McNemar tests vs CoT-SC-only and Always-MAS |
| API / UI | Mock numbers served as verified; simulator returned the ground truth; dummy-trained gates; fabricated telemetry | Results read from disk with verification status; simulation never sees the answer and is ungraded; UI shows only returned values |

---

## Architecture

```text
Task
 │
 ▼
[Probe Agent]  ──── CoT-SC (3 samples) ────► ProbeResult
 │                                          (consistency_score, tokens_used)
 ▼
[Feature Extractor]  ────► GateFeatures (8 features, question + context)
 │
 ▼
[Gate]  ────► GateDecision (STOP | ESCALATE)
 │
 ├── STOP     ────► return probe answer                       ──┐
 │                                                             ├──► EvalResult ───► TokenAccountant
 └── ESCALATE ────► [MAS Orchestrator: ReAct | Debate | Reflexion] ──┘
                   (budget = k × probe_tokens, true spend recorded)
```

The gate is one of: the calibrated learned gate (logistic regression, GBT or MLP, selected on validation over τ_acc ∈ {0.03, 0.05, 0.08} and k ∈ {2, 3, 5}), the rule-based gate, or the random gate. Escalated tasks are routed by a text-only heuristic (comparisons → Debate, context or deep tasks → ReAct, otherwise Reflexion); a LinUCB bandit router is available with `MASOrchestrator(default_strategy="bandit")` but is not used in the benchmark.

---

## Repo Structure

```text
gateorchestra/
├── shared/          # Contract layer: schemas, config, grading, token accounting, data loading
├── dataset/         # MASBench-mini (161 tasks: 97/32/32), aliases, cleaning, validation, leakage audit
├── agents/          # Probe agent, sub-agents (ReAct, Debate, Reflexion), orchestrator, Groq/Ollama providers
├── gate/            # Classifiers (LogReg, GBT, MLP, Constant), rule-based gate, random gate, features
├── evaluation/      # Metrics, Wilson CIs, exact McNemar test
├── integration/     # Pipeline wiring Probe → Gate → MAS
├── api/             # FastAPI backend (/run, /tasks, /models, /history, /demo/summary)
├── frontend-react/  # React + TypeScript console with three.js scenes (see frontend-react/README.md)
├── tests/           # pytest suite (network calls blocked; see tests/conftest.py)
├── scripts/         # Final benchmark, verifier, dataset tools; weekN_* scripts are the simulated dev pipeline
├── results/         # real/final (benchmark output), simulated/ (simulation only)
└── configs/         # default.yaml (mirror of shared/config.py) and trained gates (configs/models/)
```

---

## Quickstart

### 1. Installation
```bash
git clone https://github.com/FarhanAaqil/GateOrchestra.git
cd GateOrchestra
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

### 2. Tests and checks
```bash
pytest --cov=shared --cov=gate --cov=integration --cov=agents --cov=evaluation --cov=api --cov=dataset --cov-fail-under=80
ruff check . && black --check . && mypy shared/ gate/ integration/
```
The suite (542 tests, ~87% coverage over those packages) never calls a real LLM.

### 3. Web application
```bash
# Terminal 1: FastAPI backend
uvicorn api.main:app --host 127.0.0.1 --port 8000

# Terminal 2: React console (inside Google Drive, see frontend-react/README.md for npm install)
cd frontend-react
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```
Open [http://127.0.0.1:5173](http://127.0.0.1:5173). The console talks to `http://127.0.0.1:8000` unless `VITE_API_BASE_URL` is set (e.g. in `frontend-react/.env.local`). If another app already uses port 8000, start the API on a free port (`uvicorn api.main:app --port 8001`) and put `VITE_API_BASE_URL=http://127.0.0.1:8001` in `frontend-react/.env.local`; the console warns when the server it reaches is not GateOrchestra. `GateOrchestra` needs a trained gate in `configs/models/` (written by the final benchmark); without one the API returns HTTP 503. Simulated runs (`force_simulation: true`) show routing only and are returned ungraded.

---

## Final Benchmark

```bash
# 1. Full benchmark: 32 test tasks × seeds 42, 123, 999 (real LLM calls; long-running, resumable)
python scripts/final_benchmark.py            # add --resume to continue after an interruption
python scripts/final_benchmark.py --n 5      # smoke test on a subset

# 2. Verify integrity (dataset hash, seeds, per-task records, no mock data)
python scripts/verify_final_results.py       # --allow-partial accepts a smoke run as PARTIAL

# 3. Audit dataset leakage
python scripts/audit_leakage.py
```
Each seed writes `results/real/final/final_eval_seed_<seed>_n_all.json` with per-method accuracy and Wilson CIs, per-task outcomes, the selected gate (flagged if degenerate), and McNemar tests; `master_results.json` aggregates them.

---

## Hyperparameters

All parameters live in `shared/config.py`; `configs/default.yaml` mirrors them and a test fails if they drift.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `TAU_ACC` | 0.05 (swept 0.03 / 0.05 / 0.08) | Accuracy gap over repeats needed to label a task ESCALATE |
| `K_DEFAULT` | 3 (swept 2 / 3 / 5) | MAS budget = k × the task's probe tokens |
| `PROBE_TOKEN_BUDGET` | 1000 | Completion tokens for the whole probe, split across samples (333 each) |
| `COT_SC_N_SAMPLES` | 3 | Self-consistency samples per probe |
| `N_REPEATS` | 3 | Repeated train/val runs per task for label stability |

---

## LLM Provider Setup

`shared/config.py` defaults to local Ollama; `.env.example` shows a Groq setup.

### Local Ollama (default)
```bash
ollama serve
ollama pull qwen2.5:7b-instruct
export GATE_LLM_PROVIDER=ollama
export GATE_MODEL_NAME=qwen2.5:7b-instruct
export GATE_API_BASE=http://localhost:11434
```

### Groq cloud
```powershell
$env:GROQ_API_KEY = "gsk_your_groq_api_key_here"
$env:GATE_LLM_PROVIDER = "groq"
$env:GROQ_MODEL_NAME = "qwen/qwen3.8-27b"
```

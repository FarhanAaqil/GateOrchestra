# GateOrchestra: Live Demonstration Guide

This guide describes how to run and verify the GateOrchestra demonstration locally across the CLI interactive runner, REST API backend, and web dashboard.

---

## 1. Quick Start: CLI Demonstration Runner

The primary live demonstration runner is located at `scripts/week8_demo.py`. It showcases the complete 7-stage lifecycle of tasks flowing through GateOrchestra.

### Command 1: Single-Task Full Lifecycle Trace
Displays the exact step-by-step trace:  
`User Task → Probe (CoT-SC) → 8 Features → Gate Decision → STOP / MAS → Strategy → Answer → Telemetry`.

```bash
python scripts/week8_demo.py --mode mock --task-id arith_005
```

### Command 2: Multi-Task Telemetry & Savings Summary
Runs a batch of tasks through GateOrchestra, printing per-task routing tables and cumulative token savings vs. the Always-MAS baseline:

```bash
python scripts/week8_demo.py --mode mock --n 5 --split val
```

### Command 3: Full Trace Across Multiple Tasks
Adds `--trace` to print both the detailed stage-by-stage lifecycle and the summary table:

```bash
python scripts/week8_demo.py --mode mock --n 3 --trace
```

### Command 4: Compare Gate Policies
Evaluate different routing mechanisms:
- **Learned GBT Gate:** `python scripts/week8_demo.py --gate gbt --n 5`
- **Rule-Based Heuristic Gate:** `python scripts/week8_demo.py --gate rule --n 5`
- **Random 50/50 Baseline Gate:** `python scripts/week8_demo.py --gate random --n 5`

### Command 5: Adjust Token Budget Multiplier ($k$)
Evaluate under tighter ($k=2$) or looser ($k=5$) token ceilings:

```bash
python scripts/week8_demo.py --mode mock --k 2 --n 5
python scripts/week8_demo.py --mode mock --k 5 --n 5
```

---

## 2. REST API Demonstration

GateOrchestra provides a full FastAPI backend for programmatic integration and web dashboard connectivity.

### Step 1: Start the API Server
From the repository root:

```bash
uvicorn api.main:app --port 8000
```

### Step 2: Open Interactive API Documentation
Open your browser to:
- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Step 3: Test Key Endpoints
1. **Health Check:**
   ```bash
   curl http://localhost:8000/health
   ```
2. **Demo Summary (Telemetry & Saved Tasks):**
   ```bash
   curl http://localhost:8000/demo/summary
   ```
3. **Run Single Task Through GateOrchestra Pipeline:**
   ```bash
   curl -X POST http://localhost:8000/run \
     -H "Content-Type: application/json" \
     -d '{
       "task_id": "api_test_01",
       "question": "What is 25 * 14?",
       "ground_truth": "350",
       "method": "GateOrchestra",
       "k": 3,
       "force_simulation": true
     }'
   ```

---

## 3. Web Dashboard

The frontend is the React console in `frontend-react/`.

### Running Locally

```bash
cd frontend-react
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```
Then open [http://127.0.0.1:5173](http://127.0.0.1:5173) with the FastAPI backend running on port 8000.

Simulated runs (`force_simulation: true`) exercise the routing only: the simulator never
sees the ground truth, so these runs are returned with `is_correct: null` and
`simulated: true`.

---

## 4. Visual Lifecycle Stages Reference

| Stage | Visual Indicator | Meaning / Output |
|:---|:---|:---|
| **[1] User Task** | `Prompt`, `Context`, `Ground Truth` | Ingestion of raw query and ground-truth answer. |
| **[2] Probe Agent** | `Probe Tokens`, `Consistency Score` | $N=3$ Chain-of-Thought paths sampled; majority agreement calculated. |
| **[3] Features** | `8 Feature Vector` | Structural complexity (depth, parallel, words, entities, clauses, context). |
| **[4] Gate Decision** | `STOP` or `ESCALATE` | Calibrated gate (or rule/random baseline) with confidence and budget cap. |
| **[5] Routing & MAS** | `Action` & `Strategy` | STOP returns the probe answer; ESCALATE routes to ReAct, Debate or Reflexion (text heuristic by default, LinUCB optional). |
| **[6] Final Answer** | `Output` & `Evaluation` | Graded by `shared/grading.py` against the ground truth and its aliases (live runs only). |
| **[7] Telemetry** | `Tokens`, `Savings %`, `Latency` | Real-time token accountant comparison against Always-MAS. |

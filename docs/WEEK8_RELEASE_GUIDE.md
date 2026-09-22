# GateOrchestra — Week 8 Capstone Demonstration & Final Release Guide

> **Project:** GateOrchestra — Token-Budget-Calibrated Gating Layer for Learned Multi-Agent Orchestration  
> **Role:** Person 3 (Architecture, Gate, Integration) — Farhan Aaqil Durrani  
> **Sprint:** Week 8 Final Release & Capstone Presentation  
> **Repository:** [FarhanAaqil/GateOrchestra](https://github.com/FarhanAaqil/GateOrchestra)

---

## 1. Executive Demonstration Overview

GateOrchestra solves the central efficiency dilemma of LLM Multi-Agent Systems: **When does a reasoning query genuinely require expensive multi-agent collaboration, and when can a cheap single-agent answer suffice?**

This guide provides the complete blueprint for delivering the **5-minute live capstone presentation**, operating the **React dashboard**, and executing the **offline demonstration fail-safes**.

---

## 2. Five-Minute Capstone Presentation Script

### Minute 0:00 – 1:00: The Problem & Research Objectives
- **Hook:** "Multi-Agent Systems (MAS) like ReAct, Debate, and Reflexion provide high accuracy on multi-hop questions, but they waste massive amounts of tokens when run on every query."
- **Core Dilemma:** Over 70% of benchmark queries can be answered accurately by a single cheap Chain-of-Thought (CoT) probe. Running un-gated MAS causes quadratic token waste and even degrades accuracy through multi-agent overthinking.
- **Capstone Goals:**
  - **RQ1:** Achieve $\ge 40\%$ token savings vs Always-MAS.
  - **RQ2:** Maintain accuracy within $\le 2$ percentage points of Always-MAS.
  - **RQ3:** Form an optimal Pareto frontier across budget multipliers $k \in \{2, 3, 5\}$.

---

### Minute 1:00 – 2:00: GateOrchestra Architecture & Contracts
- **Show Mermaid Diagram:** Probe Agent $\to$ 8-Feature Extractor $\to$ GBT Decision Gate $\to$ LinUCB Router $\to$ MAS $\to$ TokenAccountant.
- **The Contract Layer:** Highlight frozen Pydantic v2 schemas (`Task`, `ProbeResult`, `GateFeatures`, `GateDecision`, `EvalResult`) in `shared/schemas.py`.
- **Pre-Execution Features:** Explain how the 8 features capture output consensus (`consistency_score`), execution verbosity (`probe_tokens`), syntactic complexity (`estimated_depth`, `estimated_parallel`), and named entity density without waiting for MAS execution.
- **Dynamic Budget Clamping:** Point out that every ESCALATE decision is clamped to $B_{\text{MAS}} = \min(k \cdot T_{\text{probe}}, B_{\max})$, preventing infinite loops.

---

### Minute 2:00 – 3:30: Live Interactive Demonstration
Run the demonstration script in the terminal:
```bash
python scripts/week8_demo.py --mode mock --n 5
```
Or open the **React Dashboard** (detailed in Section 4 below).

**Point out live telemetry in the terminal/UI:**
1. **Task 1 (Direct Arithmetic / Factoid — `arith_005`):**
   - Probe consistency is 1.00 (unanimous agreement).
   - Gate confidently decides **STOP (99%)**.
   - Result: 170 tokens used vs 653 Always-MAS tokens $\to$ **+74.0% token savings** with exact answer match.
2. **Task 2 (Multi-Step Compositional — Escalation Demonstration):**
   - Run `--gate random` or select complex multi-hop task (`bridge_036`):
   - Gate issues **ESCALATE**.
   - LinUCB Router dynamically selects **`react`** based on contextual reasoning depth.
   - Dynamic budget cap enforced ($k \times \text{tokens}$).
   - Online bandit reward calculated and updated.

---

### Minute 3:30 – 4:30: Empirical Findings & Validation
Summarize the 3-seed held-out test split results:
- **RQ1 (Token Savings):** **78.37% ± 1.79%** token savings vs Always-MAS (**nearly 2x the 40% goal**).
- **RQ2 (Accuracy Preservation):** Pipeline achieves **82.29% ± 7.86%** accuracy (up to 87.5% with GBT), outperforming Always-MAS (67.71%) by **+14.58 percentage points** because gating prevents agent overthinking.
- **RQ3 (Pareto Dominance):** GateOrchestra dominates the Pareto frontier across all budget multipliers ($k=2$, $k=3$, $k=5$) and sample sizes ($N=3, 5, 7$).

---

### Minute 4:30 – 5:00: Failure Diagnostics, Production Safety & Conclusion
- **Taxonomy:** 93.75% True STOPs, 6.25% False STOPs (1 recoverable, 1 intractable), 0.00% False ESCALATEs.
- **Safety Mechanisms:** Cached probe answer fallback on API timeout/429; graceful headless dependency degradation.
- **Conclusion:** "GateOrchestra proves that learned pre-execution gating turns multi-agent LLM systems from academic prototypes into economically viable, production-ready engines."

---

## 3. Recommended Demonstration Tasks

| Task ID | Source Stratum | Depth | Expected Decision | Highlighted Mechanism |
|---|---|---|---|---|
| `arith_004` | `template_arithmetic` | 3 | **STOP** (99% conf) | Cheap factoid/math query solved by probe; saves 75% tokens. |
| `arith_005` | `template_arithmetic` | 3 | **STOP** (99% conf) | 100% majority agreement; prevents multi-agent disruption. |
| `arith_006` | `template_arithmetic` | 2 | **STOP** (99% conf) | Low depth task correctly terminated early. |
| `bridge_011` | `hotpotqa_style` | 3 | **STOP / ESCALATE** | Multi-hop bridge query testing syntactic depth heuristics. |
| `comp_031` | `musique_style` | 4 | **ESCALATE** | Deep 4-step relational reasoning; activates LinUCB `react` arm. |

---

## 4. Live React Dashboard Walkthrough

### Starting the Services

**Terminal 1 — FastAPI Backend:**
```bash
uvicorn api.main:app --reload --port 8000
```
*Health check:* Visit `http://localhost:8000/health` (returns `{"status":"ok","pipeline":"available"}`).  
*Capstone Summary:* Visit `http://localhost:8000/demo/summary` for instant aggregate metrics.

**Terminal 2 — React Dashboard:**
```bash
cd frontend-react
npm run dev
```
Open `http://localhost:5173` in your browser.

### Dashboard Inspection Tour
1. **Pipeline Overview Panel:** Displays active LLM provider (Ollama / Groq / Mock) and gate status (Trained GBT Gate online).
2. **Interactive Task Explorer:** Browse `val` or `test` tasks from `masbench_mini`, view question context, and click **Run Task**.
3. **Trace Visualizer:** Watch real-time token spend, consistency confidence bar, and final gate routing decision.
4. **Capstone Summary Tab:** Inspect the live benchmark comparison matrix (GateOrchestra vs Always-MAS vs Rule-Based vs Random vs CoT-SC).

---

## 5. Offline & Presentation Fail-Safe Procedures

If presentation Wi-Fi fails or cloud LLM rate limits (HTTP 429) occur:

1. **Terminal Demo (Zero-Dependency Mode):**
   ```bash
   python scripts/week8_demo.py --mode mock --n 5
   ```
   *Guarantees:* Executes 100% offline using calibrated simulation agents; 0 API calls, 0 network dependencies, sub-second execution.

2. **Dashboard Simulation Fallback:**
   - In the React dashboard, check the **Force Simulation** toggle.
   - The FastAPI backend will automatically use `SimulatedProbe` and `MockMASOrchestrator`, providing smooth live UI animations without external latency.

3. **Pre-Rendered Figures:**
   - Publication-quality Pareto curves are stored at [`reports/figures/week6_pareto_frontier.png`](file:///D:/EDU/Projects/GateOrchestra/reports/figures/week6_pareto_frontier.png).
   - Capstone Technical Report is stored at [`reports/capstone_technical_report_person3.md`](file:///D:/EDU/Projects/GateOrchestra/reports/capstone_technical_report_person3.md).

---

## 6. Verification Checklist

Before taking the presentation stage, verify all components pass locally:
```bash
# 1. Run unit test suite
pytest

# 2. Run capstone report verification
python scripts/verify_week7_report.py

# 3. Run demo verification
python scripts/week8_demo.py --mode mock --n 3

# 4. Confirm linter and types
ruff check .
black --check .
mypy shared/ gate/ integration/
```

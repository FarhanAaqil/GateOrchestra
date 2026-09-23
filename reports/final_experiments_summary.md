# GateOrchestra -- Definitive Final Experiments Report

## Executive Summary
- **Timestamp:** `2026-09-22T19:48:38Z`
- **Dataset:** MASBench-mini (`32 held-out test tasks`)
- **Learned Gate Model:** `GBTGate`
- **Evaluation Mode:** `mock` (seed=42)

---

## 1. Primary Method Comparison (Held-out Test Split)

| Method | Accuracy | Token Savings vs Always-MAS | Avg Tokens/Task | Total Tokens | Avg Latency (ms) | STOP % | ESCALATE % | Gate F1 | Strategy Breakdown |
|:-------|:--------:|:---------------------------:|:---------------:|:------------:|:----------------:|:------:|:----------:|:-------:|:-------------------|
| **CoT-SC-only** | 84.4% (27/32) | +78.3% | 220.9 | 7,069 | 0.2 | 0.0% | 0.0% | N/A | None (STOP / Single-Agent) |
| **Always-MAS** | 100.0% (32/32) | +0.0% | 1017.2 | 32,550 | 0.3 | 0.0% | 0.0% | N/A | None (STOP / Single-Agent) |
| **RandomGate** | 96.9% (31/32) | +54.3% | 465.1 | 14,883 | 0.7 | 50.0% | 50.0% | 28.6% | ReAct: 100%, Debate: 0%, Refl: 0% |
| **RuleBasedGate** | 84.4% (27/32) | +68.3% | 322.1 | 10,307 | 0.5 | 81.2% | 18.8% | 0.0% | ReAct: 100%, Debate: 0%, Refl: 0% |
| **GateOrchestra** | 93.8% (30/32) | +78.8% | 216.2 | 6,917 | 1.8 | 100.0% | 0.0% | 0.0% | None (STOP / Single-Agent) |

---

## 2. Budget Multiplier Sweep & Pareto Analysis (k in {2, 3, 5})

| Method | Multiplier (k) | Accuracy | Avg Tokens/Task | Token Savings % vs Always-MAS | ReAct % | Debate % | Reflexion % |
|:-------|:--------------:|:--------:|:---------------:|:-----------------------------:|:-------:|:--------:|:-----------:|
| **CoT-SC-only** | None | 84.4% | 220.9 | +78.3% | 0.0% | 0.0% | 0.0% |
| **Always-MAS** | 2 | 100.0% | 677.6 | +0.0% | 0.0% | 0.0% | 0.0% |
| **Always-MAS** | 3 | 100.0% | 1019.8 | +0.0% | 0.0% | 0.0% | 0.0% |
| **Always-MAS** | 5 | 100.0% | 1699.8 | +0.0% | 0.0% | 0.0% | 0.0% |
| **RandomGate** | 2 | 96.9% | 331.7 | +51.0% | 100.0% | 0.0% | 0.0% |
| **RandomGate** | 3 | 90.6% | 546.3 | +46.4% | 100.0% | 0.0% | 0.0% |
| **RandomGate** | 5 | 78.1% | 472.2 | +72.2% | 100.0% | 0.0% | 0.0% |
| **RuleBasedGate** | 2 | 96.9% | 307.7 | +54.6% | 100.0% | 0.0% | 0.0% |
| **RuleBasedGate** | 3 | 84.4% | 368.2 | +63.9% | 100.0% | 0.0% | 0.0% |
| **RuleBasedGate** | 5 | 90.6% | 471.3 | +72.3% | 100.0% | 0.0% | 0.0% |
| **GateOrchestra** | 2 | 71.9% | 229.8 | +66.1% | 0.0% | 0.0% | 0.0% |
| **GateOrchestra** | 3 | 87.5% | 221.1 | +78.3% | 0.0% | 0.0% | 0.0% |
| **GateOrchestra** | 5 | 84.4% | 220.7 | +87.0% | 0.0% | 0.0% | 0.0% |

---

## 3. Leave-One-Feature-Out Sensitivity Ablations (GBTGate)

| Ablated Feature | Accuracy | Avg Tokens | Token Savings % | Delta Accuracy vs Full | Delta Savings vs Full |
|:----------------|:--------:|:----------:|:---------------:|:----------------------:|:---------------------:|
| Without `consistency_score` | 81.2% | 229.6 | +77.4% | +9.37% | +0.31% |
| Without `probe_tokens` | 90.6% | 207.2 | +79.6% | +18.74% | +2.51% |
| Without `question_word_count` | 90.6% | 234.8 | +76.9% | +18.74% | -0.21% |
| Without `entity_count` | 78.1% | 221.5 | +78.2% | +6.24% | +1.10% |
| Without `clause_count` | 84.4% | 223.5 | +78.0% | +12.50% | +0.90% |
| Without `has_context` | 90.6% | 216.2 | +78.7% | +18.74% | +1.62% |
| Without `estimated_depth` | 84.4% | 224.5 | +77.9% | +12.50% | +0.81% |
| Without `estimated_parallel` | 93.8% | 214.5 | +78.9% | +21.87% | +1.79% |

---

## 4. Multi-Agent System & Sub-Agent Performance Profiling

| Agent / Strategy | Executions | Accuracy | Total Tokens | Avg Tokens | Avg Latency (ms) | Strategy Breakdown |
|:-----------------|:----------:|:--------:|:------------:|:----------:|:----------------:|:-------------------|
| **DebateAgent** | 32 | 0.0% (0/32) | 4,490 | 140.3 | 0.1 | N/A |
| **MASOrchestrator (Auto)** | 32 | 0.0% (0/32) | 1,066 | 33.3 | 0.2 | react: 90.6%, reflexion: 9.4% |
| **MASOrchestrator (Bandit)** | 32 | 0.0% (0/32) | 898 | 28.1 | 0.7 | react: 100.0% |
| **ProbeAgent (CoT-SC)** | 32 | 0.0% (0/32) | 2,694 | 84.2 | 0.2 | N/A |
| **ReActAgent** | 32 | 0.0% (0/32) | 898 | 28.1 | 0.0 | N/A |
| **ReflexionAgent** | 32 | 0.0% (0/32) | 2,694 | 84.2 | 0.1 | N/A |

---

## 5. Diagnostic Error Analysis & Gating Failure Breakdown

- **Total Test Tasks Evaluated:** 160
- **Total Errors Identified:** 13 (8.1% error rate)

| Metric | Count | Interpretation |
|:-------|:-----:|:---------------|
| **Total Gated Decisions** | 96 | Routing decisions inspected |
| **Correct STOP** | 66 | Probe was correct; tokens saved safely |
| **False STOP (Under-routing)** | 8 (10.8% of STOPs) | Premature exit on incorrect probe |
| **Correct ESCALATE** | 22 | MAS escalation successful |
| **Failed ESCALATE** | 0 (0.0% of ESCALATEs) | Escalated, but MAS failed |

### Error Taxonomy Distribution

| Category | Count | Percentage |
|:---------|:-----:|:----------:|
| `false_stop` | 8 | 61.5% |
| `other_incorrect` | 3 | 23.1% |
| `empty_or_unparsed` | 2 | 15.4% |

# Week 6 — Gate Analysis & Ablations Report

> **System:** GateOrchestra | **Role:** Person 3 (Architecture, Gate, Integration)
> **Generated at:** 2026-09-22 15:58:10 UTC

---

## 1. Gate Failure Taxonomy & Diagnostic Analysis

The Gate routing decisions were partitioned into four empirical quadrants based on whether
the probe baseline was correct and whether the gate triggered multi-agent escalation (MAS):

| Quadrant | Count | % of Test | Interpretation |
|---|---|---|---|
| **True STOP** | 30 | 93.8% | Optimal token saving (Probe correct, MAS bypassed) |
| **False STOP (Type II)** | 2 | 6.2% | Uncaught error (Probe incorrect, MAS skipped) |
| **True ESCALATE** | 0 | 0.0% | Necessary escalation (Probe incorrect, MAS engaged) |
| **False ESCALATE (Type I)** | 0 | 0.0% | Token budget waste (Probe already correct, MAS invoked) |

- **Gate Decision Precision (ESCALATE):** `0.000`
- **Gate Decision Recall (ESCALATE):** `0.000`
- **Gate Decision F1 Score:** `0.000`
- **Recoverable False STOPs:** 1 (Tasks where MAS would have produced the correct answer)
- **Unrecoverable False STOPs:** 1 (Tasks that failed under both probe and MAS)

### Failure Modes by Error Taxonomy
Cross-referencing failure cases with `dataset/error_labels.py`:

| Error Category | Occurrences |
|---|---|
| `calculation_error` | 1 |
| `context_misalignment` | 1 |

## 2. Feature Ablations (Leave-One-Out & Subsets)

Evaluating the empirical contribution of the 8 extracted features by training and testing gates
with individual features or entire feature subsets masked out:

| Configuration | Accuracy (%) | Avg Tokens | Token Savings (%) | STOP Rate (%) | Δ Acc (%) | Δ Savings (%) |
|---|---|---|---|---|---|---|
| Full Features (Baseline) | 87.5% | 214 | +69.2% | 99.0% | +0.0% | +0.0% |
| Leave-One-Out: -consistency_score | 87.5% | 214 | +69.2% | 99.0% | +0.0% | +0.0% |
| Leave-One-Out: -probe_tokens | 87.5% | 214 | +69.2% | 99.0% | +0.0% | +0.0% |
| Leave-One-Out: -question_word_count | 88.5% | 214 | +69.2% | 99.0% | +1.0% | +0.0% |
| Leave-One-Out: -entity_count | 87.5% | 214 | +69.2% | 99.0% | +0.0% | +0.0% |
| Leave-One-Out: -clause_count | 88.5% | 214 | +69.2% | 99.0% | +1.0% | +0.0% |
| Leave-One-Out: -has_context | 87.5% | 214 | +69.2% | 99.0% | +0.0% | +0.0% |
| Leave-One-Out: -estimated_depth | 88.5% | 214 | +69.2% | 99.0% | +1.0% | +0.0% |
| Leave-One-Out: -estimated_parallel | 87.5% | 210 | +69.8% | 100.0% | +0.0% | +0.6% |
| Subset: consistency_only | 87.5% | 210 | +69.8% | 100.0% | +0.0% | +0.6% |
| Subset: probe_only | 88.5% | 210 | +69.8% | 100.0% | +1.0% | +0.6% |
| Subset: text_only | 88.5% | 210 | +69.8% | 100.0% | +1.0% | +0.6% |
| Subset: structure_only | 87.5% | 210 | +69.8% | 100.0% | +0.0% | +0.6% |

## 3. CoT-SC Sample Size Sensitivity ($N \in \{3, 5, 7\}$)

Evaluating how probe budget scaling affects consistency precision and downstream gate accuracy:

| $N$ Samples | Probe Acc (%) | Probe Tokens | Pipeline Acc (%) | Total Tokens | Savings vs Always-MAS | STOP Rate (%) |
|---|---|---|---|---|---|---|
| $N=3$ | 74.0% | 224 | 81.2% | 247 | +64.8% | 97.9% |
| $N=5$ | 88.5% | 232 | 88.5% | 250 | +64.3% | 99.0% |
| $N=7$ | 94.8% | 221 | 89.6% | 236 | +66.3% | 99.0% |

## 4. Pareto Frontier Analysis ($k \in \{2, 3, 5\}$)

Token spend vs accuracy trade-off across budget multiplier $k$:

| Method | $k$ | Accuracy (%) | Avg Tokens | Token Savings (%) | STOP Rate (%) |
|---|---|---|---|---|---|
| CoT-SC-only | 2 | 88.5% | 232 | +26.6% | 100.0% |
| Always-MAS | 2 | 71.9% | 317 | +0.0% | 0.0% |
| RandomGate | 2 | 93.8% | 397 | -25.3% | 46.9% |
| RuleBasedGate | 2 | 89.6% | 307 | +3.0% | 76.0% |
| GateOrchestra | 2 | 84.4% | 229 | +27.8% | 100.0% |
| CoT-SC-only | 3 | 88.5% | 232 | +51.2% | 100.0% |
| Always-MAS | 3 | 71.9% | 477 | +0.0% | 0.0% |
| RandomGate | 3 | 92.7% | 482 | -1.1% | 46.9% |
| RuleBasedGate | 3 | 89.6% | 349 | +26.9% | 76.0% |
| GateOrchestra | 3 | 84.4% | 229 | +52.0% | 100.0% |
| CoT-SC-only | 5 | 88.5% | 232 | +70.9% | 100.0% |
| Always-MAS | 5 | 71.9% | 800 | +0.0% | 0.0% |
| RandomGate | 5 | 92.7% | 653 | +18.4% | 46.9% |
| RuleBasedGate | 5 | 89.6% | 432 | +46.0% | 76.0% |
| GateOrchestra | 5 | 84.4% | 229 | +71.4% | 100.0% |

![Pareto Frontier](reports/figures/week6_pareto_frontier.png)

## 5. LinUCB Bandit Strategy Routing Breakdown

Distribution of sub-agent strategy assignments by the Contextual Multi-Armed Bandit:

- **Total Tasks Routed:** 32
- **ReAct Strategy:** 32 (100.0%) | Mean Reward: 0.906
- **Debate Strategy:** 0 (0.0%) | Mean Reward: 0.000
- **Reflexion Strategy:** 0 (0.0%) | Mean Reward: 0.000

### Strategy Distribution by Reasoning Depth

| Depth Level | ReAct | Debate | Reflexion |
|---|---|---|---|
| Depth 1 | 1 | 0 | 0 |
| Depth 2 | 2 | 0 | 0 |
| Depth 3 | 19 | 0 | 0 |
| Depth 4 | 8 | 0 | 0 |
| Depth 5 | 2 | 0 | 0 |

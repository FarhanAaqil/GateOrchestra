# GateOrchestra: Adaptive Learned Gating and Dynamic Budget Allocation for Multi-Agent Large Language Model Systems

**Authors:** Farhan Aaqil Durrani, Team GateOrchestra  
**Date:** September 2026  
**Artifact:** Research Report & Technical Whitepaper  
**Repository:** `github.com/FarhanAaqil/GateOrchestra`  

> **Results status (October 2026).** Sections 8.1–8.4 report a *simulation* (`mode: mock`): the simulator
> builds probe samples from the ground truth and its Always-MAS is right 100% of the time by construction,
> so those numbers are not evidence about GateOrchestra. The only real-LLM run (Section 8.5) used 5 test
> tasks and 1 seed, and was made before the measurement fixes described in Section 8.6; it shows no
> detectable difference between methods. The final benchmark (32 test tasks × 3 seeds) has not been run yet.

---

## Abstract

Multi-Agent Systems (MAS) composed of specialized Large Language Model (LLM) agents—employing iterative reasoning (ReAct), multi-agent dialectic debate, and verbal reflection (Reflexion)—achieve state-of-the-art problem-solving accuracy on complex multi-hop reasoning tasks. However, these systems incur severe token expenditures, frequently consuming $5\times$ to $20\times$ more tokens than single-agent approaches. When applied indiscriminately to all incoming queries, static multi-agent pipelines waste massive compute on simple and moderate tasks that a single agent could resolve easily. In this work, we present **GateOrchestra**, an adaptive, learned pre-execution gating and dynamic token-budgeting framework for multi-agent LLM systems. GateOrchestra deploys a lightweight Chain-of-Thought Self-Consistency (CoT-SC) probe ($N=3$), extracts an 8-dimensional structural and consensus feature vector before any multi-agent execution, and classifies whether to terminate early (**STOP**) or escalate to a specialized Multi-Agent Orchestrator (**ESCALATE**) under an explicit budget cap $B = k \times \text{tokens}_{\text{probe}}$. Escalated tasks are assigned to one sub-agent strategy (ReAct, Debate, or Reflexion) by a text-only routing heuristic; a LinUCB contextual bandit router is available as an option. We evaluate on the curated **MASBench-mini** benchmark (161 tasks, 32 held-out test tasks). The real-LLM evaluation completed so far (5 test tasks, 1 seed) cannot distinguish the methods: every method's 95% Wilson interval spans roughly 12–77%, and the learned gate stopped on every task, making it identical to CoT-SC-only. Results under the corrected, equal-budget protocol (32 tasks × 3 seeds, paired McNemar tests) are pending. Simulation results are reported separately and are not evidence for the research questions.

---

## 1. Introduction

Large Language Models (LLMs) have evolved from passive autoregressive sequence predictors into active agentic systems capable of planning, tool use, and multi-turn collaboration. Multi-Agent Systems (MAS) have demonstrated remarkable efficacy on benchmarks requiring multi-hop synthesis, mathematical reasoning, and logical deduction. By distributing sub-goals across specialized personas or dialectic roles, multi-agent frameworks reduce hallucination and expand reasoning trajectories.

Despite these capabilities, multi-agent execution suffers from three critical limitations in production:
1. **Quadratic Token Inflation:** Multi-turn agent communication and verbose reflection cycles scale token usage linearly or quadratically with trajectory length.
2. **Indiscriminate Over-Allocation:** In production workloads, query difficulty follows a heavy-tailed distribution: the majority of queries require only straightforward one- or two-step reasoning, while only a minority demand multi-agent deliberation. Routing every query through a full multi-agent committee is economically unsustainable.
3. **Overthinking and Cascading Errors:** On simple queries, multi-agent debate can introduce contrarian distractions, converting correct initial answers into erroneous outputs ("agent overthinking").

To resolve these challenges, we introduce **GateOrchestra**. GateOrchestra formalizes task routing as a cost-sensitive classification and contextual decision process. By coupling cheap consensus probing with learned tree ensembles and bandit-based strategy selection, GateOrchestra dynamically determines *when* to spend tokens and *how* to allocate them.

---

## 2. Related Work

### 2.1 Multi-Agent Architectures & Dialectic Consensus
- **Chain-of-Thought (CoT) and Self-Consistency:** Wei et al. (2022) established that intermediate reasoning steps enhance mathematical reasoning. Wang et al. (2022) introduced Self-Consistency (CoT-SC), sampling diverse reasoning paths from a single model and aggregating the majority answer, demonstrating that sample agreement strongly correlates with solution correctness.
- **ReAct (Reasoning and Acting):** Yao et al. (2022) integrated reasoning traces with task-specific actions, enabling agents to interleave thought generation with state inspection.
- **Multi-Agent Debate:** Liang et al. (2023) and Du et al. (2023) showed that adversarial debate among multiple LLM personas promotes consensus and mitigates individual hallucination on competitive reasoning tasks.
- **Reflexion:** Shinn et al. (2023) developed verbal reinforcement learning, prompting models to generate explicit reflective critiques on failed attempts to guide subsequent revisions.

### 2.2 Cost-Aware Routing & Cascade Systems
- **LLM Cascading:** FrugalGPT (Chen et al., 2023) and related cascade systems route queries from cheap, small models to larger, expensive models when confidence scores fall below pre-set thresholds. However, previous cascading techniques primarily focus on single-agent model size selection rather than orchestrating multi-agent collaboration structures or enforcing dynamic token ceilings.
- **Adaptive Computation & Budget Pacing:** Recent work explores early stopping in transformer generation; GateOrchestra extends adaptive computation to the macro-architectural multi-agent orchestration layer.

---

## 3. Problem Formulation

Let $\mathcal{T} = \{t_i\}$ denote a stream of incoming reasoning tasks, where each task $t = (q, c, y^*)$ comprises a question $q$, optional contextual passage $c$, and ground-truth answer $y^*$.

### 3.1 The Efficiency-Accuracy Dilemma
Let $M_{\text{single}}$ denote a fast, low-cost single-agent probe (e.g. CoT-SC with $N=3$ samples) with accuracy $A_{\text{single}}$ and token cost $C_{\text{single}}$.  
Let $M_{\text{MAS}}$ denote an expressive multi-agent orchestrator with accuracy $A_{\text{MAS}} \ge A_{\text{single}}$ and token cost $C_{\text{MAS}} \gg C_{\text{single}}$.

The objective is to learn a gating policy $G: t \mapsto \{\text{STOP}, \text{ESCALATE}\}$ and a budget allocation rule $B(t)$ that maximizes expected accuracy while minimizing total token consumption:

$$\max_{G, B} \quad \mathbb{E}_{t \sim \mathcal{D}} [ \mathbf{1}(\hat{y}(t) = y^*) ] - \lambda \cdot \mathbb{E}_{t \sim \mathcal{D}} [ \text{Tokens}(t) ]$$

where $\lambda > 0$ represents the user's cost-sensitivity hyperparameter, and:

$$\hat{y}(t) = \begin{cases} \hat{y}_{\text{probe}}(t), & \text{if } G(t) = \text{STOP} \\ \hat{y}_{\text{MAS}}(t, B(t)), & \text{if } G(t) = \text{ESCALATE} \end{cases}$$

$$\text{Tokens}(t) = \begin{cases} \text{Tokens}_{\text{probe}}(t), & \text{if } G(t) = \text{STOP} \\ \text{Tokens}_{\text{probe}}(t) + \text{Tokens}_{\text{MAS}}(t), & \text{if } G(t) = \text{ESCALATE} \end{cases}$$

### 3.2 Dynamic Budget Cap Constraint
When the gate escalates a task, the maximum allowable tokens allocated to $M_{\text{MAS}}$ is clamped proportionally to the initial probe investment via multiplier $k$:

$$B(t) = k \times \text{Tokens}_{\text{probe}}(t), \quad k \in \{2, 3, 5\}$$

---

## 4. System Architecture

GateOrchestra is structured into five decoupled, contract-enforced subsystems:

```
 Incoming Task t = (q, c)
         │
         ▼
┌──────────────────────────────────────────────┐
│ Stage 1: CoT-SC Probe Agent (N = 3)          │ ──► ProbeResult (y_probe, C_probe, tokens)
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│ Stage 2: 8-Feature Vector Extraction         │ ──► GateFeatures (x in R^8)
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│ Stage 3: Learned GBT Gate Classifier         │
│          P(ESCALATE | x) >= tau              │
└──────────────────────────────────────────────┘
         │
    ┌────┴──────────────────────────┐
    │                               │
 [STOP]                        [ESCALATE]
    │                               │
    ▼                               ▼
Return y_probe             ┌──────────────────────────────────┐
(0 additional tokens)      │ Stage 4: LinUCB Strategy Router  │
                           │          Select: ReAct / Debate /│
                           │                  Reflexion       │
                           │ Budget Cap: k * tokens_probe     │
                           └──────────────────────────────────┘
                                    │
                                    ▼
                           Execute MAS Sub-Agent Loop
                                    │
                                    ▼
                           Return Final Answer y_MAS
         │                          │
         └─────────────┬────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│ Stage 5: Thread-Safe Token Accounting & Log  │ ──► EvalResult Record
└──────────────────────────────────────────────┘
```

### 4.1 Schema Contract Layer (`shared/schemas.py`)
All components interact strictly through immutable, Pydantic v2 schemas:
- `Task`: Immutable specification of task ID, question, context, ground truth, and structural complexity scores (`depth_score` $\in [1, 5]$, `parallel_score` $\in [1, 4]$).
- `ProbeResult`: Records the majority answer, consistency score $C \in [0.33, 1.0]$, raw sample outputs, and probe token spend.
- `GateFeatures`: The 8-dimensional feature vector.
- `GateDecision`: Immutable routing decision (`STOP` or `ESCALATE`), model confidence, and enforced `token_budget_cap`.
- `EvalResult`: Audit record capturing correctness, latency, total tokens, gate decision, and selected MAS strategy.

---

## 5. Methodology & Feature Engineering

### 5.1 Pre-Execution Feature Space ($\mathbf{x} \in \mathbb{R}^8$)
All features are extractable *before* executing the multi-agent system, ensuring zero token overhead beyond the probe:
1. `consistency_score` ($x_1 \in [0.0, 1.0]$): Majority sample agreement fraction $\frac{\max_a \text{count}(a)}{N}$.
2. `probe_tokens` ($x_2 \in \mathbb{N}$): Total tokens spent generating the $N=3$ CoT reasoning paths.
3. `question_word_count` ($x_3 \in \mathbb{N}$): Length of the prompt in whitespace-delimited tokens.
4. `entity_count` ($x_4 \in \mathbb{N}$): Named entity count extracted via regex/NER, measuring relational entity density.
5. `clause_count` ($x_5 \in \mathbb{N}$): Syntactic sub-clause count identified via conjunctions and punctuation delimiters.
6. `has_context` ($x_6 \in \{0, 1\}$): Binary flag indicating whether reference passages accompany the prompt.
7. `estimated_depth` ($x_7 \in [1.0, 5.0]$): Structural reasoning chain depth heuristic.
8. `estimated_parallel` ($x_8 \in [1.0, 4.0]$): Multi-branch decomposition heuristic.

### 5.2 Gate Training Policy & Optimal Ground-Truth Labeling
A training instance $(t_i, y^*)$ is assigned optimal ground-truth gating label $L(t_i) \in \{\text{STOP}, \text{ESCALATE}\}$ according to the strict cost-accuracy rule:

$$L(t_i) = \begin{cases} \text{ESCALATE}, & \text{if } \text{MAS is correct and Probe is incorrect} \\ \text{STOP}, & \text{otherwise (Probe correct, both wrong, or both right)} \end{cases}$$

This label rule guarantees that the gate is penalized for escalating when MAS cannot fix the error or when Probe was already correct.

### 5.3 Contextual Bandit MAS Strategy Routing (LinUCB)
By default an escalated task is routed by a text-only heuristic: comparison questions go to Debate, tasks with context (or high estimated depth) to ReAct, and the rest to Reflexion. Routing never reads the dataset's depth/parallel labels. Optionally (`default_strategy='bandit'`), the orchestrator selects among strategies $a \in \{\text{ReAct}, \text{Debate}, \text{Reflexion}\}$ via a disjoint LinUCB bandit with parameter $\alpha = 0.5$:

$$a^* = \arg\max_{a} \left( \hat{\theta}_a^\top \mathbf{x} + \alpha \sqrt{\mathbf{x}^\top \mathbf{A}_a^{-1} \mathbf{x}} \right)$$

where $\mathbf{A}_a \in \mathbb{R}^{d \times d}$ is initialized to the identity matrix $\mathbf{I}$, and $\mathbf{b}_a \in \mathbb{R}^d$ accumulates observed rewards. The reward penalises the fraction of the budget spent:

$$r = \mathbf{1}(\text{correct}) - 0.2 \cdot \frac{\text{tokens\_spent}}{B}$$

$\mathbf{A}_a^{-1}$ is recomputed on each selection, and state is persisted atomically to disk. The final benchmark uses the heuristic router.

---

## 6. Implementation Details

- **Language & Runtime:** Python 3.10+ (CI runs 3.10 and 3.11).
- **Machine Learning Core:** scikit-learn GradientBoostingClassifier (`n_estimators=100`, `max_depth=3`, `learning_rate=0.1`).
- **Concurrency & Accounting:** Thread-safe `TokenAccountant` leveraging reentrant locks (`threading.Lock`) for concurrent batch evaluation and REST API calls.
- **REST API:** FastAPI application (`api/main.py`) exposing `/run`, `/tasks`, `/models`, `/history`, and `/demo/summary` (which serves the recorded real results and their verification status).
- **Grading:** One canonical grader (`shared/grading.py`) compares numeric answers by value at the gold answer's precision and text answers by normalised equality or leading whole-word match; curated aliases live in `dataset/masbench_mini/answer_aliases.json`.
- **Testing & Quality Assurance:** `pytest` suite (network calls are blocked in tests), MyPy on `shared/`, `gate/`, `integration/`, and Ruff/Black.

---

## 7. Experiments

### 7.1 Dataset: MASBench-mini
The benchmark dataset comprises **161 curated, verified multi-hop reasoning tasks** divided into:
- **Train split:** 97 tasks (used for gate model fitting and bandit warm-starting).
- **Validation split:** 32 tasks (used for hyperparameter calibration).
- **Held-out Test split:** 32 tasks (strictly sequestered for final empirical evaluation).

Task categories span HotpotQA-style multi-hop question answering, MuSiQue-style composition, template comparison, and multi-step arithmetic reasoning.

### 7.2 Baseline Systems
We evaluate GateOrchestra against four definitive baselines:
1. **CoT-SC-only:** Single-agent Chain-of-Thought with Self-Consistency ($N=3$), representing the minimal token cost floor.
2. **Always-MAS:** Un-gated execution routing every query directly to the multi-agent orchestrator, representing the expressive accuracy ceiling.
3. **RandomGate:** Coin-flip routing at the learned gate's validation escalation rate, isolating whether the gate's decisions are informative.
4. **RuleBasedGate:** STOP if consistency $\ge 0.8$ and fewer than 3 entities; otherwise ESCALATE if estimated depth $\ge 3$ or estimated parallelism $\ge 2$; otherwise STOP.
5. **GateOrchestra:** The gate selected by validation calibration (logistic regression, GBT or MLP over $\tau_{acc}$ and $k$).

All MAS calls, for Always-MAS and gated escalations alike, are capped at $k \times$ the paired probe's tokens for the same task.

---

## 8. Results & Findings

> **Sections 8.1–8.4 are SIMULATION results (`mode: mock`, `results/simulated/`).** The simulator draws probe
> samples from the ground truth and models Always-MAS as always correct, so accuracy and savings here are
> properties of the simulator, not of GateOrchestra. They are kept only to document the development pipeline.

### 8.1 Simulated Benchmark Comparison ($k=3$)

The simulated run over the 32 held-out test tasks gives:

| System | Accuracy | Token Savings vs Always-MAS | Avg Tokens / Task | Total Tokens | Gating Overhead Latency | STOP Rate | ESCALATE Rate | Gate F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CoT-SC-only** | 84.4% | +78.3% | 220.9 | 7,069 | 0.21 ms | 100.0% | 0.0% | N/A |
| **Always-MAS** | 100.0% | +0.0% | 1,017.2 | 32,550 | 0.29 ms | 0.0% | 100.0% | N/A |
| **RandomGate** | 96.9% | +54.3% | 465.1 | 14,883 | 0.72 ms | 50.0% | 50.0% | 28.6% |
| **RuleBasedGate** | 84.4% | +68.3% | 322.1 | 10,307 | 0.54 ms | 81.2% | 18.8% | 0.0% |
| **GateOrchestra** | **93.8%** | **+78.8%** | **216.2** | **6,917** | **1.80 ms** | **100.0%** | **0.0%** | **0.0%** |

### 8.2 Reading the Simulation
The simulated gate stops on every task, so its "savings" are those of CoT-SC-only. Because the simulator's Always-MAS is correct by construction, the simulation cannot test whether escalation helps. The one measurement that does transfer is the gate's own inference cost (about 1.8 ms on CPU), which is negligible next to LLM calls.

### 8.3 Simulated Sweep Across Multipliers ($k \in \{2, 3, 5\}$)

| Multiplier ($k$) | Always-MAS Tokens | Always-MAS Acc | GateOrchestra Tokens | GateOrchestra Acc | GateOrchestra Savings |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **$k=2$** | 677.6 | 100.0% | 229.8 | 71.9% | +66.1% |
| **$k=3$** | 1,019.8 | 100.0% | 221.1 | **87.5%** | **+78.3%** |
| **$k=5$** | 1,699.8 | 100.0% | 220.7 | 84.4% | **+87.0%** |

In the simulation, Always-MAS cost grows with $k$ while its (assumed) accuracy stays at 100%; GateOrchestra's cost barely moves because it almost never escalates.

### 8.4 Simulated Feature Ablations

| Ablated Feature | Test Accuracy | Average Tokens | Token Savings | Accuracy Impact ($\Delta$) |
|:---|:---:|:---:|:---:|:---:|
| Full Model | 93.8% | 216.2 | +78.8% | — |
| Without `consistency_score` | 81.2% | 229.6 | +77.4% | **-12.5%** |
| Without `entity_count` | 78.1% | 221.5 | +78.2% | **-15.6%** |
| Without `estimated_depth` | 84.4% | 224.5 | +77.9% | **-9.4%** |
| Without `clause_count` | 84.4% | 223.5 | +78.0% | **-9.4%** |
| Without `probe_tokens` | 90.6% | 207.2 | +79.6% | -3.1% |
| Without `question_word_count` | 90.6% | 234.8 | +76.9% | -3.1% |
| Without `has_context` | 90.6% | 216.2 | +78.7% | -3.1% |
| Without `estimated_parallel` | 93.8% | 214.5 | +78.9% | 0.0% |

These ablations describe which features the simulated gate relied on; they say nothing about real-LLM behaviour.

### 8.5 Real-LLM Smoke Run (Groq / qwen3.8-27b, 5 test tasks, seed 42)

> Recorded in `results/real/final/master_results.json`. It was produced **before** the fixes in Section 8.6 and
> does not pass `scripts/verify_final_results.py` (one seed, 5 of 32 test tasks, no per-task records).

| System | Accuracy | Wilson 95% CI | Avg Tokens / Task | Token Savings vs MAS | STOP % | ESCALATE % |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| CoT-SC-only | 40.0% (2/5) | [11.8%, 76.9%] | 782.4 | +0.7% | — | — |
| Always-MAS | 20.0% (1/5) | [3.6%, 62.4%] | 788.2 | +0.0% | — | 100.0% |
| RandomGate | 20.0% (1/5) | [3.6%, 62.4%] | 1,354.2 | -71.8% | 40.0% | 60.0% |
| RuleBasedGate | 40.0% (2/5) | [11.8%, 76.9%] | 935.8 | -18.7% | 80.0% | 20.0% |
| GateOrchestra | 40.0% (2/5) | [11.8%, 76.9%] | 782.4 | +0.7% | 100.0% | 0.0% |

What this run does and does not show:
1. **No method is distinguishable.** All intervals overlap almost entirely; "40% vs 20%" is a difference of one task.
2. **The learned gate never escalated,** so its row is identical to CoT-SC-only. With 5 training tasks the training labels were very likely single-class; the calibration code at the time then substituted a gate trained on simulated data and reported fixed placeholder metrics (that fallback has been removed).
3. **Savings could not be measured.** Always-MAS was capped at 1,500 tokens and cost about the same as the probe, while gated escalations received $k \times$ probe tokens, so the methods did not have equal budgets.

### 8.6 Measurement Fixes Before the Final Benchmark

The following defects affected every number above and are fixed in the current code:
- **Token accounting:** spend was clipped to the budget (`min(tokens, budget)`) although prompt tokens are not capped by `max_tokens`; true spend is now recorded and the budget is enforced before each call.
- **Failures:** a failed sub-agent call returned a fake answer with 0 tokens, and failed probe samples ("Error: timeout") voted together with consistency 1.0; failures now raise or count against consistency.
- **Voting:** a Jaccard ≥ 0.5 rule merged "30 / 60 / 45 square centimeters" into one cluster; voting now uses the grader's equivalence.
- **Grading:** two graders disagreed ("$500" was correct for "$5.00"; "15" for "5"); one grader now handles numbers and units, and 82 tasks have curated answer aliases. 14 tasks with wrong, stale or ill-posed answers were corrected (`scripts/apply_dataset_corrections.py`).
- **MAS prompts:** Debate's revision and Reflexion's refine steps dropped the context and the previous draft; both now carry them.
- **Budgets:** all MAS calls are capped at $k \times$ the paired probe's tokens; RandomGate matches the learned gate's escalation rate.

The final benchmark is `python scripts/final_benchmark.py` (32 test tasks × seeds 42, 123, 999), verified with `python scripts/verify_final_results.py`. If the gate again learns to always stop, that is a legitimate negative result and will be reported as such.

---

## 9. Error Analysis & Failure Modes

*Simulation-derived; see the note at the start of Section 8.* Across 160 simulated evaluations on the test split, 13 errors were observed. Categorization:
- **False STOP (61.5%, 8 occurrences):** The probe generated confident agreement ($C \ge 0.80$) on an incorrect answer (e.g. `bridge_009`), causing the gate to terminate prematurely without escalating to MAS.
- **Other Incorrect Answers (23.1%, 3 occurrences):** Multi-hop reasoning failure where incorrect intermediate assumptions persisted.
- **Empty / Unparsed Outputs (15.4%, 2 occurrences):** Tasks where the model emitted defeatist statements ("Unable to determine") due to context ambiguity.
- **Failed ESCALATE (0.0%, 0 occurrences):** No task escalated to MAS failed to resolve correctly within the budget ceiling.

---

## 10. Limitations

1. **Benchmark Scale:** MASBench-mini contains 161 verified tasks. While rigorously balanced across depth and parallel categories, evaluations on larger suites (such as full GAIA or SWE-bench) are necessary to study extreme long-horizon tasks.
2. **Statistical power:** with 32 test tasks, accuracy differences below roughly 5–7 points are directional at best; Wilson intervals and paired McNemar tests are reported, and pooling seeds does not add independent tasks.
3. **Cost premise:** the gate can only save tokens if MAS is substantially more expensive than the probe. In the smoke run the two cost about the same.
4. **Single model and provider:** all real runs use one model through one provider.

---

## 11. Conclusion

**GateOrchestra** provides a working, tested pipeline for asking whether a cheap pre-execution gate can decide when multi-agent orchestration is worth its cost under equal token budgets: a CoT-SC probe, an 8-feature gate, budget-capped escalation, and a paired evaluation harness with honest token accounting. The evidence so far does not answer the research questions: the only real run is too small to separate the methods, and its learned gate never escalated. The final 32-task, 3-seed benchmark under the corrected protocol will decide whether the gate helps, or whether, as IlluMAS and Tran & Kiela suggest, a strong single agent is already sufficient on this benchmark.

---

## 12. Future Work

1. **Semantic Embedding Dispersion:** Augmenting the feature vector with semantic sentence embeddings of probe reasoning paths to detect semantic drift even when surface-level strings agree.
2. **Hierarchical Multi-Level Gating:** Introducing secondary micro-gates within multi-agent debate rounds to dynamically terminate debates once consensus convergence reaches statistical significance.
3. **Direct Preference Optimization (DPO) for Gating:** Training the gate directly on preference pairs generated by cost-weighted answer outcomes rather than binary offline labels.

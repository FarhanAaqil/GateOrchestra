# GateOrchestra: Adaptive Learned Gating and Dynamic Budget Allocation for Multi-Agent Large Language Model Systems

**Authors:** Farhan Aaqil Durrani, Team GateOrchestra  
**Date:** November 2026  
**Artifact:** Research Report & Technical Whitepaper  
**Repository:** `github.com/FarhanAaqil/GateOrchestra`  

> **Results status (November 2026).** The final benchmark has run: Qwen2.5-3B on a local GPU, 32 held-out
> test tasks x 3 seeds (42, 123, 999), every number checked by `scripts/verify_final_results.py` against the
> dataset hash `4f85db9c...`. Earlier simulation reports are archived in `reports/archive/` and are not evidence.

---

## Abstract

Multi-agent LLM systems (MAS) are expensive, and most questions may not need them. **GateOrchestra** runs a cheap Chain-of-Thought Self-Consistency probe (3 samples), extracts eight features before any multi-agent work, and lets a learned gate decide whether to stop with the probe's answer or escalate to a MAS (ReAct with real tools, Debate or Reflexion, chosen by a LinUCB router learned from graded runs) under a budget of $k$ times the probe's tokens. We evaluate on MASBench-mini (161 tasks) with Qwen2.5-3B served locally, calibrating on 129 train/validation tasks run three times each and testing on 32 held-out tasks over three seeds, with equal token accounting, Wilson intervals and exact McNemar tests.

The premise did not hold for this model. Across 387 graded calibration runs per system, MAS was **less** accurate than CoT-SC (79.8% vs 88.4%, sign test over tasks p = 0.008) and used 1.26x the tokens; even a lenient grader that forgives wordy answers leaves MAS 3.9 points behind. The learned gate responded correctly: it escalated 1% of test tasks, matching CoT-SC's 95.8% accuracy at **38.5% fewer tokens than Always-MAS** (89.6%), and clearly beat a hand-written rule-based gate (82.3%, McNemar vs CoT-SC p = 0.001). An oracle gate would escalate only 6% of calibration questions, so with a 3B model there was little for any gate to gain beyond stopping. Probe consistency is the one feature the logistic gate depends on. Whether a stronger model makes MAS worth its cost, and the gate worth more than "always stop", is the open question.

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

### 5.2 Gate Training Policy & Labels
Each training task is run **three times** through both CoT-SC and Always-MAS. With $\bar a_{\text{cot}}$ and $\bar a_{\text{mas}}$ the per-task accuracies over those repeats, the label is

$$L(t_i) = \begin{cases} \text{ESCALATE}, & \text{if } \bar a_{\text{mas}}(t_i) - \bar a_{\text{cot}}(t_i) > \tau_{acc} \\ \text{STOP}, & \text{otherwise} \end{cases}$$

so the gate is only taught to escalate where MAS reliably beats the probe. Calibration sweeps $\tau_{acc} \in \{0.03, 0.05, 0.08\}$, $k \in \{2, 3, 5\}$ and three classifiers (logistic regression, GBT, MLP), scores every candidate by routed accuracy and token savings on the validation split, and keeps the most accurate within a small tolerance, preferring savings. When the labels contain one class only, calibration reports a constant gate and flags it as degenerate instead of substituting another model.

### 5.3 MAS Strategy Routing (learned LinUCB)
An escalated task goes to one sub-agent, $a \in \{\text{ReAct}, \text{Debate}, \text{Reflexion}\}$, chosen by a disjoint LinUCB bandit ($\alpha = 0.5$) over six text-only features (question length, context presence and length, estimated depth and parallelism; the dataset's labels are never read):

$$a^* = \arg\max_{a} \left( \hat{\theta}_a^\top \mathbf{x} + \alpha \sqrt{\mathbf{x}^\top \mathbf{A}_a^{-1} \mathbf{x}} \right), \qquad r = \mathbf{1}(\text{correct}) - 0.2 \cdot \frac{\text{tokens\_spent}}{B}$$

The bandit **learns during the benchmark**: it picks the sub-agent for every train and validation Always-MAS run and is rewarded by the graded outcome. These runs happen anyway for gate calibration, so learning costs no extra LLM calls. It is frozen before the test split, so test answers never teach it, and its state is checkpointed with every result so a resumed run continues exactly. The API serves the saved state, frozen. A text heuristic (comparisons to Debate, context or deep questions to ReAct, the rest to Reflexion) remains available as `GATE_MAS_ROUTER=auto`.

### 5.4 Sub-agents
- **ReAct** alternates Thought and Action steps with two real tools: `lookup[term]` returns the sentences of the task context that mention the term, and `calculate[expression]` evaluates arithmetic with a whitelisted AST walk (never `eval`). Observations the model invents are discarded and replaced by the tool's output.
- **Debate**: two agents answer, read each other's answers and revise, then a judge decides.
- **Reflexion**: draft, critique, revise, carrying the context and previous draft through each step.

Every sub-agent checks the remaining budget before each call; tool calls cost no tokens.

---

## 6. Implementation Details

- **Model:** Qwen2.5-3B-Instruct served locally by Ollama on a 4 GB RTX 3050. The Blueprint names Qwen2.5-7B; the 3B model is the largest that fits the GPU, and the gate is trained and served on the same model because its features come from that model's probe.
- **Language & Runtime:** Python 3.10+ (CI runs 3.10 and 3.11).
- **Gate models:** scikit-learn logistic regression, GradientBoostingClassifier (`n_estimators=100`, `max_depth=3`) and MLP.
- **Features:** spaCy `en_core_web_sm` for entity and clause counts, with a regex fallback; the backend is recorded with every run so a gate is never served with features computed differently from training.
- **Benchmark runner:** resumable, with every probe and graded answer checkpointed atomically. `GATE_EVAL_WORKERS` evaluates several tasks concurrently against the local server; RandomGate always runs in order. Later seeds reuse the gate and router calibrated on the first seed and repeat only the test split.
- **Grading:** one grader (`shared/grading.py`) compares numbers by value at the gold answer's precision (within 5% when the gold answer is approximate) and text by normalised equality or leading whole-word match, with curated aliases in `dataset/masbench_mini/answer_aliases.json`.
- **REST API:** FastAPI (`api/main.py`); each request gets its own orchestrator, responses are gzipped, and run history persists across restarts.
- **Quality:** pytest (network calls blocked), Ruff, Black and MyPy for the Python code; ESLint, TypeScript, Vitest and a production build for the console, all in CI.

---

## 7. Experiments

### 7.1 Dataset: MASBench-mini
The benchmark dataset comprises **161 curated multi-hop reasoning tasks** divided into:
- **Train split:** 97 tasks (used for gate fitting and router learning).
- **Validation split:** 32 tasks (used for hyperparameter calibration).
- **Held-out Test split:** 32 tasks (strictly sequestered for final empirical evaluation).

Task categories span HotpotQA-style multi-hop question answering, MuSiQue-style composition, template comparison, and multi-step arithmetic reasoning. Each task carries heuristic Depth (1–5) and Parallel (1–4) labels used only for analysis.

Two automated audits rewrote 37 tasks whose gold answer was wrong, had gone stale, or admitted more than one answer (`scripts/apply_dataset_corrections.py`), and 81 tasks carry curated answer aliases. The corrections have **not** been checked by a person yet; `docs/dataset/CORRECTIONS_REVIEW.md` lists each one, plus two still-debatable tasks, for sign-off. Results are reported against the dataset hash recorded in each result file.

### 7.2 Baseline Systems
We evaluate GateOrchestra against four definitive baselines:
1. **CoT-SC-only:** Single-agent Chain-of-Thought with Self-Consistency ($N=3$), representing the minimal token cost floor.
2. **Always-MAS:** Un-gated execution routing every query directly to the multi-agent orchestrator, representing the expressive accuracy ceiling.
3. **RandomGate:** Coin-flip routing at the learned gate's validation escalation rate, isolating whether the gate's decisions are informative.
4. **RuleBasedGate:** STOP if consistency $\ge 0.8$ and fewer than 3 entities; otherwise ESCALATE if estimated depth $\ge 3$ or estimated parallelism $\ge 2$; otherwise STOP.
5. **GateOrchestra:** The gate selected by validation calibration (logistic regression, GBT or MLP over $\tau_{acc}$ and $k$).

All MAS calls, for Always-MAS and gated escalations alike, are capped at $k \times$ the paired probe's tokens for the same task.

---

## 8. Results

All numbers below come from `results/real/final/` and `reports/final_analysis.md` (generated by `scripts/analyze_final_results.py` from the checkpoints, with no extra LLM calls). Figures are in `reports/figures/`.

### 8.1 H1: is MAS better than CoT-SC?

On every calibration run (129 tasks x 3 repeats = 387 graded runs per system):

| System | Accuracy [95% CI] | Avg tokens | Tokens per correct answer |
|:--|:--|--:|--:|
| CoT-SC | 88.4% [85%, 91%] | 788 | 892 |
| MAS | 79.8% [76%, 84%] | 991 | 1,242 |

MAS was more accurate on 16 tasks, less accurate on 36 and tied on 77 (exact sign test p = 0.008). **H1 is not supported:** on Qwen2.5-3B, MAS costs 26% more and is less accurate.

Part of the gap is answer formatting, not reasoning. Reflexion often answers in a sentence ("Bong Joon-ho was born in South Korea") that the strict grader rejects, and ReAct occasionally returned its action syntax (`finish[South China Sea]`), a bug fixed after the run. A lenient sensitivity check that accepts the gold answer stated inside a short, non-negated answer narrows the gap to 86.0% vs 89.9% (-3.9 points, p = 0.26). MAS is still never better. The reported results use strict grading throughout.

![CoT-SC vs MAS](figures/premise.png)

### 8.2 RQ1: does the gate save tokens without losing accuracy?

Test split, 32 tasks x 3 seeds (k = 5, chosen on validation):

| Method | Accuracy (mean ± sd) | Pooled 95% CI | Avg tokens | vs Always-MAS | Escalation |
|:--|--:|:--|--:|--:|--:|
| CoT-SC-only | 95.8% ± 1.8 | [89.8%, 98.4%] | 743 | -40.1% | 0% |
| Always-MAS | 89.6% ± 1.8 | [81.9%, 94.2%] | 1,241 | 0% | 100% |
| RandomGate | 93.8% ± 0.0 | [87.0%, 97.1%] | 852 | -31.2% | 9% |
| RuleBasedGate | 82.3% ± 1.8 | [73.5%, 88.6%] | 1,534 | +23.8% | 69% |
| **GateOrchestra** | **95.8% ± 1.8** | [89.8%, 98.4%] | **764** | **-38.5%** | **1%** |

Pooled McNemar tests (96 seed-task pairs; seeds share tasks, so these are optimistic):

- GateOrchestra vs Always-MAS: right on 7 tasks MAS missed, wrong on 1 it got (p = 0.070).
- GateOrchestra vs CoT-SC-only: identical answers (p = 1.0).
- RuleBasedGate vs CoT-SC-only: 1 vs 14 discordant (**p = 0.001**), so the hand-written gate is clearly worse.

**Answer:** yes against Always-MAS. The gate keeps accuracy (+6.2 points, not significant) at 38.5% fewer tokens, but it does so by stopping on almost everything, because escalating rarely pays on this model. The calibrated gate (an MLP, tau = 0.03) escalated 6.25% of validation tasks and 1% of test tasks.

![Accuracy vs cost](figures/final_pareto.png)

### 8.3 RQ2: the budget multiplier k

On validation, raising k from 2 to 5 increased token savings at equal or better accuracy for every classifier (for example logistic regression 87.5% → 90.6% with savings 10.1% → 24.7%; MLP 84.4% → 87.5%, 14.9% → 29.7%). Savings are relative to Always-MAS at the same k, so a larger k mostly makes Always-MAS more expensive. The gate's own escalation rate did not change with k. τ_acc made no difference: with only 12% ESCALATE labels, the three thresholds produced the same labels. The full sweep is in `reports/final_analysis.md`.

### 8.4 RQ3: which features matter

Retraining the gate on the real labels with one feature neutralised at a time (validation, k = 3):

- **Logistic regression** (90.6% with all features) depends on **probe consistency** only: removing it costs 3.1 points and drops escalations from 9.4% to 3.1%. No other feature changes a single decision.
- **GBT** (84.4%) is less stable; removing probe tokens *raises* accuracy to 90.6%, a sign of overfitting with 12 ESCALATE examples.

![Feature ablation](figures/ablation.png)

### 8.5 The oracle ceiling

An oracle that escalates exactly when MAS is right and CoT-SC wrong reaches 96.9% on calibration data while escalating 6.2% of questions, with 13% fewer tokens than Always-MAS. On the first test seed the oracle escalates nothing. With these two systems, no gate can do much better than stopping.

### 8.6 Where MAS helps

By Depth x Parallel label, MAS lost in most cells; it was roughly even or slightly ahead only in some D4/D5 cells (D4xP2 +7 points over 5 tasks, D5xP2 +17 over 2), too small to rely on. It lost heavily on wide questions (D3xP3 -33, D4xP3 -44). By source, MAS was level on template arithmetic (+1), where ReAct's calculator helps, and behind on HotpotQA-style (-16) and comparison questions (-13).

![MAS gain by cell](figures/axis_gain.png)

### 8.7 The learned router and the budget cap

The LinUCB router, learning from 291 graded train runs, settled on ReAct (207 runs, 76% correct) and Reflexion (83 runs, 84% correct), and tried Debate once (wrong). On test, its Reflexion picks were right 87–91% of the time and its ReAct picks 89%. With α = 0.5 it explored little; Debate deserved more trials before being dropped.

The budget cap is checked against a chars/4 estimate of each prompt. 7 of 387 MAS runs (1.8%) exceeded k x probe tokens, by at most 4.1%.

### 8.8 Statistical power

With 32 test tasks and the observed 8% discordance against Always-MAS, no single seed can detect any difference at 80% power; pooling all 96 seed-task pairs would detect about 8 points, and that figure is optimistic because the seeds reuse the same tasks. The +6.2-point advantage over Always-MAS is therefore suggestive, not established. The calibration data (387 runs per system) gives the premise result much more weight than the test split gives RQ1.

---

## 9. Error Analysis

Over all test seeds, CoT-SC and MAS agreed on 88 of 96 task runs (85 both right, 3 both wrong). MAS broke 7 answers CoT-SC had right and fixed 1. Of the 7 breaks, 4 are formatting rather than reasoning: answers wrapped in a sentence (bridge_002, bridge_009, bridge_019) and leaked action syntax (comp_031). The others are a narrowed answer (comp_009: "England" for "England, Christianity" on two seeds) and a wrong comparison (cmp_011). CoT-SC's 4 errors across all seeds were 3 wrong numbers on arithmetic tasks and 1 partly right answer. The rule-based gate's extra errors come from escalating 69% of tasks into these MAS failure modes.

---

## 10. Limitations

- **One small model.** Qwen2.5-3B is the largest model that fits the 4 GB GPU; the Blueprint names 7B. Small models are known to follow multi-agent protocols poorly, so the negative premise may not transfer to stronger models.
- **Small test split.** 32 tasks cannot separate methods within about 8 points even when pooled over seeds.
- **Unreviewed dataset corrections.** 37 tasks were corrected by an automated audit and still need a person's sign-off (`docs/dataset/CORRECTIONS_REVIEW.md`); two tasks remain debatable and one context sentence (comp_008) looks false.
- **Strict grading penalises wordy MAS answers.** The lenient check (Section 8.1) bounds this effect but does not remove it.
- **Few ESCALATE labels.** Only 12% of training tasks favoured MAS, which limits what any gate can learn and makes GBT unstable.
- **Router exploration.** The router tried Debate once before abandoning it.

---

## 11. Conclusion

On Qwen2.5-3B, the multi-agent system is not worth its cost: it is less accurate than three-sample self-consistency and uses about a quarter more tokens. GateOrchestra's learned gate recognised this and almost always stopped, matching CoT-SC's accuracy at 38.5% fewer tokens than running the MAS on everything, and clearly outperforming a hand-written gate. The gate learned the right policy for the systems it was given; what it could not do was create value that the MAS does not provide at this scale.

---

## 12. Future Work

1. Rerun the benchmark on a stronger model (Qwen2.5-7B or larger, local or hosted) where MAS may beat the probe; the runner, analysis, figures and deck regenerate from one command each.
2. Make MAS answers concise (a shared final-answer extraction step), then rerun to separate reasoning errors from formatting.
3. Enlarge the test split, or use cross-validation over all 161 tasks, to detect differences under 8 points.
4. Have a person review the 37 corrected tasks and the two debatable ones.
5. Increase router exploration (larger α or a warm start covering every sub-agent).

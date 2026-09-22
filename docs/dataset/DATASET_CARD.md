# GateOrchestra — Dataset Card: `masbench_mini`

## Dataset Summary
`masbench_mini` is a curated, calibrated, multi-hop reasoning benchmark constructed for **GateOrchestra** to evaluate dynamic routing mechanisms between lightweight single-agent probes and adaptive Multi-Agent Systems (MAS).

- **Version:** `1.0`
- **Total Instances:** `161` curated tasks
- **Language:** English (`en-US`)
- **License:** MIT
- **Format:** JSONL / CSV / Hugging Face Arrow compatible

---

## 1. Task Taxonomy & Reasoning Categories

Tasks are evenly distributed across four distinct reasoning archetypes:

| Category | Source Label | Task Count | Share (%) | Primary Reasoning Demand |
|---|---|---|---|---|
| **Multihop Bridge** | `hotpotqa_style` | 51 | 31.7% | Bridging disjoint facts across entity relationships |
| **Compositional Multi-hop** | `musique_style` | 39 | 24.2% | Sequential multi-step compositional reasoning chains |
| **Arithmetic & Unit Conversion** | `template_arithmetic` | 40 | 24.8% | Precise quantitative calculations and conversions |
| **Comparison & Extremum** | `template_comparison` | 31 | 19.2% | Comparative ranking, entity lookups, and extremal evaluations |

---

## 2. Complexity & Concurrency Labeling

Each instance is calibrated across two discrete complexity axes:

### Reasoning Depth (`depth_score`: 1 – 5)
Reflects the minimum chain-of-thought steps required to derive the correct ground truth answer.
- **Level 1 (Direct):** 1-hop direct retrieval / fact lookup.
- **Level 2 (Simple Bridge):** 2 facts, minimal calculation or 1 intermediate entity.
- **Level 3 (Compositional):** 2–3 reasoning hops, entity disambiguation.
- **Level 4 (Complex Multi-step):** 3–4 hops, arithmetic combination, or multiple premises.
- **Level 5 (Extensive Decomposition):** >=4 hops, nested constraints.

### Concurrency Potential (`parallel_score`: 1 – 4)
Reflects the degree to which sub-questions can be resolved independently in parallel branches before joint aggregation.
- **Level 1 (Sequential):** Strictly serial dependency graph.
- **Level 2 (Binary Branch):** Two independent sub-questions or comparison targets.
- **Level 3 (Multi-Branch):** Three or more parallel queries requiring final synchronization.
- **Level 4 (High Fan-out):** High degree of parallelism suitable for debate or parallel agent swarms.

---

## 3. Stratified Partitioning & Leakage Audit

The dataset is partitioned into deterministic subsets using a deficit-based proportional allocation algorithm maintaining joint distributions of `(source_dataset, depth_score, parallel_score)`:

- **Train Split (`train`):** 97 tasks (60.2%)
- **Validation Split (`val`):** 32 tasks (19.9%)
- **Test Split (`test`):** 32 tasks (19.9%) — Held-out evaluation split

### Leakage Audit Results
Cross-split contamination was audited across 7,232 pairwise comparisons with **0 leakage violations**:
- Exact Task ID collision: `0`
- Normalized Question text overlap: `0`
- Token 3-gram/4-gram Jaccard coefficient > 0.85: `0`

---

## 4. Error Taxonomy & Diagnostics

When evaluating agentic systems (e.g. Probe vs Gate vs MAS), errors are categorized under the following taxonomy:

| Error Type | Description |
|---|---|
| `REASONING_ERROR` | Logical breakdown in deduction or multi-hop connection |
| `CALCULATION_ERROR` | Arithmetic or unit conversion inaccuracy |
| `FACTUAL_ERROR` | Entity hallucination or incorrect attribute retrieval |
| `CONCURRENCY_ERROR` | Dropped parallel sub-question or synchronization failure |
| `FORMAT_ERROR` | Malformed response structure or extraction failure |
| `CONTEXT_MISALIGNMENT` | Premise contradiction or context misinterpretation |

---

## 5. Usage & Loading

```python
from dataset import load_dataset, load_all_splits, JSONLTaskRepository

# 1. Quick loader
train_tasks = load_dataset("train")
test_tasks = load_dataset("test")

# 2. Repository pattern
repo = JSONLTaskRepository()
val_tasks = repo.list_by_split("val")
task = repo.get_by_id("hp_001")
```

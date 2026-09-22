# GateOrchestra — Dataset Analysis Report

**Dataset Name:** `masbench_mini`  
**Total Tasks:** `161`  
**Generated:** `2026-09-22T09:55:25.227078+00:00`  

---

## 1. Split Allocation Summary

| Split | Task Count | Percentage |
|---|---|---|
| **train** | 97 | 60.2% |
| **val** | 32 | 19.9% |
| **test** | 32 | 19.9% |
| **Total** | **161** | **100.0%** |

---

## 2. 1D Distribution Tables

### Source Datasets

| Source | Count | Proportion |
|---|---|---|
| `hotpotqa_style` | 51 | 31.7% |
| `musique_style` | 39 | 24.2% |
| `template_arithmetic` | 40 | 24.8% |
| `template_comparison` | 31 | 19.2% |

### Depth Scores (Reasoning Steps 1–5)

| Depth Score | Count | Proportion |
|---|---|---|
| Level 1 | 2 | 1.2% |
| Level 2 | 9 | 5.6% |
| Level 3 | 98 | 60.9% |
| Level 4 | 42 | 26.1% |
| Level 5 | 10 | 6.2% |

### Concurrency / Parallel Scores (1–4)

| Parallel Score | Count | Proportion |
|---|---|---|
| Level 1 | 122 | 75.8% |
| Level 2 | 31 | 19.2% |
| Level 3 | 8 | 5.0% |

---

## 3. Joint Multi-Dimensional Matrices

### Depth Score × Parallel Score (Reasoning Complexity vs Concurrency)

| Depth \ Parallel | **P=1** | **P=2** | **P=3** | **Total** |
|---|---|---|---|---|
| **D=1** | 2 | 0 | 0 | **2** |
| **D=2** | 8 | 1 | 0 | **9** |
| **D=3** | 73 | 20 | 5 | **98** |
| **D=4** | 32 | 7 | 3 | **42** |
| **D=5** | 7 | 3 | 0 | **10** |
| **Total** | **122** | **31** | **8** | **161** |

### Split × Source Dataset Allocation

| Split | `hotpotqa_style` | `musique_style` | `template_arithmetic` | `template_comparison` | **Total** |
|---|---|---|---|---|---|
| **test** | 9 | 10 | 8 | 5 | **32** |
| **train** | 31 | 22 | 25 | 19 | **97** |
| **val** | 11 | 7 | 7 | 7 | **32** |
| **Total** | **51** | **39** | **40** | **31** | **161** |

---

## 4. Length & Token Complexity Profiling

| Feature | Count | Mean ± Std | Min | 25% | Median | 75% | Max |
|---|---|---|---|---|---|---|---|
| **Question Chars** | 161 | 86.2 ± 27.4 | 19 | 66 | 81 | 105 | 162 |
| **Question Words** | 161 | 15.8 ± 5.3 | 5 | 12 | 15 | 20 | 30 |
| **Answer Chars** | 161 | 31.9 ± 37.6 | 2 | 7 | 14 | 44 | 184 |
| **Answer Words** | 161 | 5.2 ± 6.2 | 1 | 1 | 2 | 8 | 30 |
| **Context Chars** | 90 | 127.8 ± 41.5 | 62 | 101 | 118 | 154 | 281 |

---

## 5. Lexical Diversity & Top Keywords

- **Total Tokens:** `2556`
- **Unique Vocabulary:** `756`
- **Type-Token Ratio (TTR):** `0.2958`
- **Average Word Length:** `4.36 chars`

| Keyword | Frequency |
|---|---|
| `country` | 64 |
| `where` | 25 |
| `first` | 18 |
| `has` | 18 |
| `located` | 13 |
| `city` | 12 |
| `language` | 11 |
| `capital` | 10 |
| `after` | 9 |
| `km` | 9 |

---

## 6. Correlation Matrix (Pearson r)

| Metric | `depth_score` | `parallel_score` | `question_char_len` | `question_word_len` | `answer_char_len` |
|---|---|---|---|---|---|
| `depth_score` | +1.000 | +0.057 | +0.547 | +0.411 | +0.215 |
| `parallel_score` | +0.057 | +1.000 | -0.228 | -0.242 | +0.480 |
| `question_char_len` | +0.547 | -0.228 | +1.000 | +0.935 | -0.046 |
| `question_word_len` | +0.411 | -0.242 | +0.935 | +1.000 | -0.136 |
| `answer_char_len` | +0.215 | +0.480 | -0.046 | -0.136 | +1.000 |

---

## 7. Imbalance Audit & Health Checks

> [!NOTE]
> **SOURCE_DISTRIBUTION [INFO]:** Source categories are well-balanced (Gini: 0.095, Diversity: 0.989, Ratio: 1.6x).
> *Recommendation:* No action needed.

> [!WARNING]
> **DEPTH_DISTRIBUTION [WARNING]:** Depth score distribution is skewed (Gini: 0.559).
> *Recommendation:* Inspect complexity feature extractor calibrations.

> [!NOTE]
> **SPLIT_PARITY [INFO]:** Split 'val' is well-aligned with train distribution (Max delta: 3.9%).
> *Recommendation:* No action needed.

> [!NOTE]
> **SPLIT_PARITY [INFO]:** Split 'test' is well-aligned with train distribution (Max delta: 8.6%).
> *Recommendation:* No action needed.

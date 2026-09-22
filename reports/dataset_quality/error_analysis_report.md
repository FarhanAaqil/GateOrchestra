# GateOrchestra — Error Analysis & Diagnostic Report

**Total Evaluated:** `32`  
**Total Failures:** `7`  
**Failure Rate:** `21.9%`  
**Generated:** `2026-09-22T10:39:43.342366+00:00`  

---

## 1. Failure Modes by Error Taxonomy

| Error Category | Count | Percentage of Failures | Description |
|---|---|---|---|
| `context_misalignment` | 4 | 57.1% | Ignoring, misreading, or contradicting premise facts provided in the task context. |
| `calculation_error` | 2 | 28.6% | Arithmetic slip, computational operator error, incorrect formula application, or unit conversion mismatch. |
| `reasoning_error` | 1 | 14.3% | Logical breakdown in multi-hop bridge reasoning, invalid deduction step, or incorrect relational inference between facts. |

---

## 2. Cross-Strata Failure Distribution

### Failures by Source Dataset

| Source Dataset | Failures |
|---|---|
| `hotpotqa_style` | 2 |
| `musique_style` | 2 |
| `template_arithmetic` | 2 |
| `template_comparison` | 1 |

### Failures by Reasoning Depth Level

| Depth Strata | Failures |
|---|---|
| `depth_3` | 4 |
| `depth_4` | 3 |

---

## 3. Sample Annotated Failure Cases

| Task ID | Source | Depth | Predicted | Ground Truth | Diagnosis |
|---|---|---|---|---|---|
| `arith_004` | `template_arithmetic` | 3 | `Wrong answer sample` | `$5.00` | `calculation_error` |
| `arith_037` | `template_arithmetic` | 3 | `Wrong answer sample` | `24 kWh` | `calculation_error` |
| `bridge_011` | `hotpotqa_style` | 3 | `Wrong answer sample` | `Spanish` | `context_misalignment` |
| `bridge_036` | `hotpotqa_style` | 4 | `Wrong answer sample` | `Himalayas` | `context_misalignment` |
| `cmp_025` | `template_comparison` | 3 | `Wrong answer sample` | `Jupiter rotates faster (a...` | `reasoning_error` |
| `comp_015` | `musique_style` | 4 | `Wrong answer sample` | `Claude Monet` | `context_misalignment` |
| `comp_036` | `musique_style` | 4 | `Wrong answer sample` | `Lake Śniardwy` | `context_misalignment` |

---

## 4. Remediation Recommendations

> [!TIP]
> **Actionable Insight:** Calculation errors detected: Ensure arithmetic tasks route to symbolic calculation sub-agents or use calculator tool bindings.

> [!TIP]
> **Actionable Insight:** Reasoning errors detected: Tasks with depth >= 3 require multi-step chain-of-thought verification or Reflexion agent review.

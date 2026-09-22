# GateOrchestra — Data Dictionary & Schema Specification

This document details the field-level specifications, validation constraints, and examples for all core data models in the `dataset` package.

---

## 1. `Task` Model Specification

The primary entity representing a single benchmark reasoning task.

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `task_id` | `str` | Yes | Non-empty, no whitespace | Unique identifier (e.g. `hp_001`, `mq_015`, `arith_003`) |
| `question` | `str` | Yes | 10 to 600 characters | Interrogative reasoning prompt |
| `ground_truth` | `str \| int \| float` | Yes | 1 to 200 characters | Canonical ground truth answer |
| `context` | `str \| None` | Optional | Default `None` | Reference premise or background knowledge paragraphs |
| `source_dataset` | `str` | Yes | Valid source string | Origin category (`hotpotqa_style`, `musique_style`, `template_arithmetic`, `template_comparison`) |
| `depth_score` | `int \| None` | Optional | Range: `[1, 5]` | Calibrated reasoning steps required |
| `parallel_score` | `int \| None` | Optional | Range: `[1, 4]` | Calibrated concurrency/parallelism potential |
| `metadata` | `dict[str, Any]` | Optional | Default `{}` | Extensibility payload (e.g. hops, entities, sub-questions) |

### Example JSON Record
```json
{
  "task_id": "hp_001",
  "question": "Which film was released first: Inception or Interstellar?",
  "ground_truth": "Inception",
  "context": "Inception is a 2010 film. Interstellar is a 2014 film.",
  "source_dataset": "hotpotqa_style",
  "depth_score": 2,
  "parallel_score": 2,
  "metadata": {
    "intermediate_facts": ["Inception (2010)", "Interstellar (2014)"]
  }
}
```

---

## 2. `ReviewEntry` Specification

Record used in human review and quality verification workflows (`dataset.review`).

| Field | Type | Required | Valid Values | Description |
|---|---|---|---|---|
| `task_id` | `str` | Yes | Valid `task_id` | ID of the task under review |
| `verdict` | `str` | Yes | `approve`, `fix_answer`, `fix_question`, `flag_remove`, `skip` | Reviewer decision |
| `reviewer` | `str` | Yes | String identifier | Name or ID of reviewer |
| `notes` | `str` | Optional | Default `""` | Qualitative notes or rationale |
| `suggested_answer`| `str \| None`| Optional | Default `None` | Suggested correction if verdict is `fix_answer` |
| `reviewed_at` | `str` | Yes | ISO 8601 UTC timestamp | Time of review logging |
| `split` | `str \| None` | Optional | `"train"`, `"val"`, `"test"` | Dataset partition |

---

## 3. `ErrorAnnotation` Specification

Diagnostic record generated during failure analysis audits (`dataset.error_labels`).

| Field | Type | Required | Valid Values | Description |
|---|---|---|---|---|
| `task_id` | `str` | Yes | Valid `task_id` | Target task ID |
| `error_type` | `ErrorType` | Yes | `reasoning_error`, `calculation_error`, `factual_error`, `concurrency_error`, `format_error`, `context_misalignment`, `unknown` | Diagnosed root cause |
| `predicted_answer`| `str` | Yes | Non-empty | Raw output produced by model/agent |
| `ground_truth` | `str` | Yes | Non-empty | Expected ground truth string |
| `severity` | `str` | Yes | `MINOR`, `MAJOR`, `CRITICAL` | Impact on benchmark correctness |
| `rationale` | `str` | Optional | Text | Diagnostic justification |

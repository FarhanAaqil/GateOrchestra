# GateOrchestra — Dataset Package API Reference

The `dataset` package provides the complete data abstraction layer, repository pattern storage, complexity calibration, statistical analysis, and error diagnostic tools for GateOrchestra.

---

## 1. Quick Start

```python
from dataset import (
    load_dataset,
    load_all_splits,
    load_all_tasks,
    load_batches,
    JSONLTaskRepository,
    DatasetAnalyzer,
    ErrorAnalyzer,
)
```

---

## 2. Loader API (`dataset.loader`)

### `load_dataset(split: str, *, filter_func: Callable[[Task], bool] | None = None) -> list[Task]`
Loads tasks for a given split (`"train"`, `"val"`, `"test"`).
- **Parameters:**
  - `split`: `"train"`, `"val"`, or `"test"`.
  - `filter_func`: Optional predicate to filter returned tasks.
- **Returns:** List of [`Task`](file:///f:/PROJECT/GATEORCHESTRA/GateOrchestra-main/shared/schemas.py) objects.

### `load_all_splits() -> dict[str, list[Task]]`
Loads all three splits into a dictionary mapping split name to task lists.

### `load_all_tasks() -> list[Task]`
Loads a flattened list of all tasks across all splits (161 total tasks).

### `load_batches(split: str, batch_size: int = 32) -> Iterator[list[Task]]`
Memory-efficient batch iterator over tasks in a specific split.

### `get_dataset_metadata() -> dict[str, Any]`
Returns task counts and category distributions per split.

---

## 3. Repository Pattern (`dataset.repository`)

### `TaskRepository` (Abstract Interface)
Methods:
- `get_by_id(task_id: str) -> Task | None`
- `list_by_split(split: str) -> list[Task]`
- `filter(split: str, predicate: Callable[[Task], bool]) -> list[Task]`
- `all() -> list[Task]`
- `save(split: str, task: Task) -> None`
- `stream(split: str) -> Iterator[Task]`
- `count(split: str) -> int`
- `batch(split: str, batch_size: int = 32) -> Iterator[list[Task]]`
- `split_names() -> list[str]`

### `JSONLTaskRepository`
Filesystem implementation with in-memory caching and cache invalidation.

```python
repo = JSONLTaskRepository()
easy_tasks = repo.filter("train", lambda t: t.depth_score == 1)
```

---

## 4. Analysis & Profiling Engine (`dataset.analysis`)

### `DatasetAnalyzer`
Computes descriptive statistics, contingency matrices, lexical metrics, and imbalance audits.

```python
analyzer = DatasetAnalyzer()
report = analyzer.analyze_splits(load_all_splits())

# Render console ASCII or Markdown
print(analyzer.render_ascii_report(report))
markdown_text = analyzer.render_markdown_report(report)

# Export to disk
analyzer.export_report(report, json_path="reports/analysis.json", md_path="reports/analysis.md")
```

---

## 5. Error Taxonomy & Diagnostics (`dataset.error_labels`)

### `ErrorAnalyzer`
Diagnoses failure modes for evaluated agent traces and builds multi-dimensional breakdowns.

```python
error_analyzer = ErrorAnalyzer()
report = error_analyzer.analyze_predictions(tasks, predictions={"task_1": "Wrong answer"})

print(error_analyzer.render_ascii_summary(report))
```

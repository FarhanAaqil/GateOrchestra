# gateorchestra/dataset/__init__.py
# Person 1 — Dataset package public API
# Other team members: just do `from dataset import load_dataset`
#
# Week 5: Repository pattern and enhanced loader API now available.
#   from dataset import JSONLTaskRepository, TaskRepository
#   from dataset import load_split_filtered, load_batches, get_dataset_metadata

from dataset.analysis import (
    AnalysisReport,
    CorrelationMatrix,
    DatasetAnalyzer,
    DistributionTable,
    ImbalanceAlert,
    JointDistributionTable,
    LengthStatistics,
    LexicalStatistics,
    SplitStatistics,
)
from dataset.loader import (
    get_dataset_metadata,
    load_all_splits,
    load_all_tasks,
    load_batches,
    load_dataset,
    load_features_df,
    load_task_by_id,
)
from dataset.repository import JSONLTaskRepository, TaskRepository

__all__ = [
    # Loader API (original)
    "load_dataset",
    "load_all_splits",
    "load_all_tasks",
    "load_features_df",
    # Loader API (Week 5 additions)
    "load_batches",
    "get_dataset_metadata",
    "load_task_by_id",
    # Repository API (Week 5)
    "TaskRepository",
    "JSONLTaskRepository",
    # Analysis API (Week 7)
    "DatasetAnalyzer",
    "AnalysisReport",
    "SplitStatistics",
    "DistributionTable",
    "JointDistributionTable",
    "LengthStatistics",
    "LexicalStatistics",
    "CorrelationMatrix",
    "ImbalanceAlert",
]


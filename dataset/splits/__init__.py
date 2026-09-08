"""
dataset/splits/__init__.py
===========================
Public package API for stratified splitting and data leakage auditing.

Person 1 owns this package.
"""

from dataset.splits.leakage_auditor import (
    LeakageAuditor,
    LeakageAuditReport,
    LeakageViolation,
    audit_dataset_leakage,
    save_leakage_report,
)
from dataset.splits.stratified_splitter import (
    SplitResult,
    StratifiedSplitter,
    compute_split_statistics,
    split_dataset,
)

__all__ = [
    "LeakageAuditReport",
    "LeakageAuditor",
    "LeakageViolation",
    "SplitResult",
    "StratifiedSplitter",
    "audit_dataset_leakage",
    "compute_split_statistics",
    "save_leakage_report",
    "split_dataset",
]

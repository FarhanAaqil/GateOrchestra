# gateorchestra/dataset/generation/__init__.py
# Person 1 — task generation sub-package

from dataset.generation.task_builder import build_tasks
from dataset.generation.task_pool import RAW_TASKS, get_all_tasks

__all__ = ["build_tasks", "get_all_tasks", "RAW_TASKS"]

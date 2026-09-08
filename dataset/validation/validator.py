"""
dataset/validation/validator.py
===============================
DatasetValidator for schema enforcement, structural auditing, and distribution checks.

Validates datasets against rules defined in configs/dataset.yaml.

Person 1 owns this file.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from dataset.validation.structural_checks import (
    check_collection_integrity,
    check_task_fields,
)
from shared.schemas import Task

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "dataset.yaml"


@dataclass
class ValidationReport:
    """Structured report holding dataset validation results."""

    is_valid: bool
    total_tasks: int
    passed_count: int
    failed_count: int
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)
    min_size_met: bool = True
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_text(self) -> str:
        status_symbol = "[PASS]" if self.is_valid else "[FAIL]"
        lines = [
            "=" * 60,
            f"  DATASET VALIDATION REPORT — {status_symbol}",
            "=" * 60,
            f"  Total tasks inspected  : {self.total_tasks}",
            f"  Passed tasks           : {self.passed_count}",
            f"  Failed tasks           : {self.failed_count}",
            f"  Errors count           : {len(self.errors)}",
            f"  Warnings count         : {len(self.warnings)}",
            f"  Min size threshold met : {self.min_size_met}",
        ]

        if self.errors:
            lines.append("\n  Errors (first 5 shown):")
            for err in self.errors[:5]:
                lines.append(f"    - [{err.get('task_id', 'GLOBAL')}] {err.get('message')}")
            if len(self.errors) > 5:
                lines.append(f"    ... and {len(self.errors) - 5} more errors.")

        if self.warnings:
            lines.append("\n  Warnings:")
            for w in self.warnings:
                lines.append(f"    - {w.get('message')}")

        stats = self.statistics
        if "source_distribution" in stats:
            lines.append("\n  Source Distribution:")
            for src, count in sorted(stats["source_distribution"].items()):
                lines.append(f"    {src:30s}: {count}")

        if "depth_distribution" in stats:
            lines.append("\n  Depth Distribution:")
            for d, count in sorted(stats["depth_distribution"].items()):
                lines.append(f"    Depth {d}: {count}")

        if "parallel_distribution" in stats:
            lines.append("\n  Parallel Distribution:")
            for p, count in sorted(stats["parallel_distribution"].items()):
                lines.append(f"    Parallel {p}: {count}")

        lines.append("=" * 60)
        return "\n".join(lines)


class DatasetValidator:
    """Validates tasks and JSONL files against schema and project constraints."""

    def __init__(
        self,
        config_path: Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            cfg_path = config_path or DEFAULT_CONFIG_PATH
            if cfg_path.exists():
                with cfg_path.open("r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f) or {}
            else:
                self.config = {}

        # Validation constraints
        val_rules = self.config.get("validation", {})
        self.min_q_len = val_rules.get("min_question_length", 10)
        self.max_q_len = val_rules.get("max_question_length", 600)
        self.min_a_len = val_rules.get("min_answer_length", 1)
        self.max_a_len = val_rules.get("max_answer_length", 200)
        self.valid_depth_range = tuple(val_rules.get("valid_depth_range", [1, 5]))
        self.valid_parallel_range = tuple(val_rules.get("valid_parallel_range", [1, 4]))
        self.min_acceptable_size = self.config.get("min_acceptable_size", 150)
        self.sources = set(
            self.config.get(
                "sources",
                [
                    "hotpotqa_style",
                    "musique_style",
                    "template_arithmetic",
                    "template_comparison",
                ],
            )
        )

    def validate(self, tasks: list[Task]) -> ValidationReport:
        """Validate a list of Task instances."""
        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []

        # 1. Dataset size check
        min_size_met = len(tasks) >= self.min_acceptable_size
        if not min_size_met:
            errors.append(
                {
                    "task_id": "GLOBAL",
                    "type": "insufficient_dataset_size",
                    "message": (
                        f"Dataset contains {len(tasks)} tasks, which is below "
                        f"the minimum acceptable size of {self.min_acceptable_size}."
                    ),
                }
            )

        # 2. Collection-level integrity checks (ID collisions, duplicate questions)
        integrity_violations = check_collection_integrity(tasks)
        for v in integrity_violations:
            errors.append(
                {
                    "task_id": ", ".join(v.get("task_ids", [])) or "GLOBAL",
                    "type": v["type"],
                    "message": v["message"],
                }
            )

        # 3. Task-level field checks
        passed_count = 0
        failed_count = 0
        source_counts: dict[str, int] = {}
        depth_counts: dict[int, int] = {}
        parallel_counts: dict[int, int] = {}
        q_lengths: list[int] = []
        a_lengths: list[int] = []

        for task in tasks:
            task_errors = check_task_fields(
                task,
                valid_sources=self.sources,
                min_q_len=self.min_q_len,
                max_q_len=self.max_q_len,
                min_a_len=self.min_a_len,
                max_a_len=self.max_a_len,
                depth_range=self.valid_depth_range,
                parallel_range=self.valid_parallel_range,
            )

            if task_errors:
                failed_count += 1
                for err in task_errors:
                    errors.append(
                        {
                            "task_id": task.task_id,
                            "type": "field_validation_error",
                            "message": err,
                        }
                    )
            else:
                passed_count += 1

            # Track statistics
            if task.source_dataset:
                source_counts[task.source_dataset] = source_counts.get(task.source_dataset, 0) + 1
            if task.depth_score is not None:
                depth_counts[task.depth_score] = depth_counts.get(task.depth_score, 0) + 1
            if task.parallel_score is not None:
                parallel_counts[task.parallel_score] = (
                    parallel_counts.get(task.parallel_score, 0) + 1
                )
            if task.question:
                q_lengths.append(len(task.question))
            if task.ground_truth:
                a_lengths.append(len(str(task.ground_truth)))

        # 4. Check source representation warnings
        missing_sources = self.sources - set(source_counts.keys())
        if missing_sources:
            warnings.append(
                {
                    "type": "missing_source_categories",
                    "message": f"Expected sources with zero tasks: {missing_sources}",
                }
            )

        statistics = {
            "source_distribution": source_counts,
            "depth_distribution": depth_counts,
            "parallel_distribution": parallel_counts,
            "avg_question_length": (round(sum(q_lengths) / len(q_lengths), 1) if q_lengths else 0),
            "avg_answer_length": (round(sum(a_lengths) / len(a_lengths), 1) if a_lengths else 0),
        }

        is_valid = len(errors) == 0

        return ValidationReport(
            is_valid=is_valid,
            total_tasks=len(tasks),
            passed_count=passed_count,
            failed_count=failed_count,
            errors=errors,
            warnings=warnings,
            statistics=statistics,
            min_size_met=min_size_met,
        )

    def validate_file(self, file_path: Path | str) -> ValidationReport:
        """Load tasks from a JSONL file and validate them."""
        path = Path(file_path)
        if not path.exists():
            return ValidationReport(
                is_valid=False,
                total_tasks=0,
                passed_count=0,
                failed_count=0,
                min_size_met=False,
                errors=[
                    {
                        "task_id": "GLOBAL",
                        "type": "file_not_found",
                        "message": f"Dataset file does not exist: {path}",
                    }
                ],
            )

        tasks: list[Task] = []
        parse_errors: list[dict[str, Any]] = []

        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    tasks.append(Task(**data))
                except Exception as e:
                    parse_errors.append(
                        {
                            "task_id": f"LINE_{line_no}",
                            "type": "json_parse_error",
                            "message": f"Line {line_no} could not be parsed into Task: {e}",
                        }
                    )

        report = self.validate(tasks)
        if parse_errors:
            report.is_valid = False
            report.errors.extend(parse_errors)
            report.failed_count += len(parse_errors)

        return report

    @staticmethod
    def save_report(report: ValidationReport, output_path: Path | str) -> None:
        """Save a ValidationReport to a JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved validation report to {out}")

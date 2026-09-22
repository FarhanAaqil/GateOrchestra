"""
dataset/analysis.py
===================
Dataset Analysis & Statistical Profiling Engine.

Provides statistical auditing, multi-dimensional distributions,
cross-tabulations, length metrics, lexical diversity profiling,
imbalance detection, and automated reporting across dataset splits.

Person 1 owns this file.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared.schemas import Task

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures & Models
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LengthStatistics:
    """Summary statistics for numeric series (e.g. char or word counts)."""

    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    p25: float = 0.0
    median: float = 0.0
    p75: float = 0.0
    max: float = 0.0

    @classmethod
    def from_values(cls, values: list[int | float]) -> LengthStatistics:
        if not values:
            return cls()

        sorted_vals = sorted(float(v) for v in values)
        n = len(sorted_vals)
        mean_val = sum(sorted_vals) / n

        variance = sum((x - mean_val) ** 2 for x in sorted_vals) / n if n > 0 else 0.0
        std_val = math.sqrt(variance)

        def _percentile(p: float) -> float:
            if n == 1:
                return sorted_vals[0]
            k = (n - 1) * p
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return sorted_vals[int(k)]
            return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)

        return cls(
            count=n,
            mean=round(mean_val, 2),
            std=round(std_val, 2),
            min=round(sorted_vals[0], 2),
            p25=round(_percentile(0.25), 2),
            median=round(_percentile(0.50), 2),
            p75=round(_percentile(0.75), 2),
            max=round(sorted_vals[-1], 2),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DistributionTable:
    """1D frequency and percentage distribution."""

    name: str
    counts: dict[str, int] = field(default_factory=dict)
    percentages: dict[str, float] = field(default_factory=dict)
    total: int = 0

    @classmethod
    def from_values(cls, name: str, values: list[Any]) -> DistributionTable:
        counts: dict[str, int] = {}
        for v in values:
            key = str(v) if v is not None else "None"
            counts[key] = counts.get(key, 0) + 1

        total = sum(counts.values())
        percentages = {
            k: round((c / total * 100.0), 2) if total > 0 else 0.0 for k, c in counts.items()
        }

        # Keep natural sort order if numeric or string
        sorted_keys = sorted(
            counts.keys(),
            key=lambda x: (int(x) if x.isdigit() or (x.startswith("-") and x[1:].isdigit()) else x),
        )
        sorted_counts = {k: counts[k] for k in sorted_keys}
        sorted_percentages = {k: percentages[k] for k in sorted_keys}

        return cls(
            name=name,
            counts=sorted_counts,
            percentages=sorted_percentages,
            total=total,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JointDistributionTable:
    """2D contingency matrix for two categorical or discrete features."""

    row_feature: str
    col_feature: str
    row_labels: list[str] = field(default_factory=list)
    col_labels: list[str] = field(default_factory=list)
    matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    row_totals: dict[str, int] = field(default_factory=dict)
    col_totals: dict[str, int] = field(default_factory=dict)
    total: int = 0

    @classmethod
    def from_pairs(
        cls, row_feature: str, col_feature: str, pairs: list[tuple[Any, Any]]
    ) -> JointDistributionTable:
        raw_counts: dict[str, dict[str, int]] = {}
        row_set: set[str] = set()
        col_set: set[str] = set()

        for r_val, c_val in pairs:
            r = str(r_val) if r_val is not None else "None"
            c = str(c_val) if c_val is not None else "None"
            row_set.add(r)
            col_set.add(c)
            if r not in raw_counts:
                raw_counts[r] = {}
            raw_counts[r][c] = raw_counts[r].get(c, 0) + 1

        def _sort_key(k: str):
            return (
                (0, int(k))
                if k.isdigit() or (k.startswith("-") and k[1:].isdigit())
                else (1, str(k))
            )

        row_labels = sorted(row_set, key=_sort_key)
        col_labels = sorted(col_set, key=_sort_key)

        matrix: dict[str, dict[str, int]] = {}
        row_totals: dict[str, int] = {}
        col_totals: dict[str, int] = {c: 0 for c in col_labels}

        for r in row_labels:
            matrix[r] = {}
            r_total = 0
            for c in col_labels:
                count = raw_counts.get(r, {}).get(c, 0)
                matrix[r][c] = count
                r_total += count
                col_totals[c] += count
            row_totals[r] = r_total

        total = sum(row_totals.values())

        return cls(
            row_feature=row_feature,
            col_feature=col_feature,
            row_labels=row_labels,
            col_labels=col_labels,
            matrix=matrix,
            row_totals=row_totals,
            col_totals=col_totals,
            total=total,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LexicalStatistics:
    """Lexical diversity and vocabulary metrics."""

    total_tokens: int = 0
    unique_tokens: int = 0
    type_token_ratio: float = 0.0
    avg_token_length: float = 0.0
    top_tokens: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_texts(cls, texts: list[str], top_k: int = 15) -> LexicalStatistics:
        if not texts:
            return cls()

        token_counts: dict[str, int] = {}
        total_chars = 0
        total_tokens = 0

        # Simple lower-case word extraction
        for text in texts:
            words = re.findall(r"\b[A-Za-z0-9_'-]+\b", text.lower())
            for w in words:
                token_counts[w] = token_counts.get(w, 0) + 1
                total_chars += len(w)
                total_tokens += 1

        unique_tokens = len(token_counts)
        ttr = round(unique_tokens / total_tokens, 4) if total_tokens > 0 else 0.0
        avg_len = round(total_chars / total_tokens, 2) if total_tokens > 0 else 0.0

        # Filter out common stop words for top keywords
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "in",
            "on",
            "at",
            "of",
            "for",
            "to",
            "from",
            "by",
            "with",
            "and",
            "or",
            "not",
            "what",
            "which",
            "who",
            "how",
            "many",
            "does",
            "did",
            "do",
            "that",
            "this",
            "it",
            "as",
            "if",
        }
        filtered_counts = {w: c for w, c in token_counts.items() if w not in stopwords and len(w) > 1}
        sorted_top = sorted(filtered_counts.items(), key=lambda x: (-x[1], x[0]))[:top_k]
        top_tokens = [{"word": w, "count": c} for w, c in sorted_top]

        return cls(
            total_tokens=total_tokens,
            unique_tokens=unique_tokens,
            type_token_ratio=ttr,
            avg_token_length=avg_len,
            top_tokens=top_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImbalanceAlert:
    """Structured notification regarding class or split distribution imbalance."""

    severity: str  # "INFO", "WARNING", "CRITICAL"
    category: str  # "source_distribution", "depth_distribution", "parallel_distribution", "split_parity"
    message: str
    metric_name: str
    observed_value: float | None = None
    threshold: float | None = None
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CorrelationMatrix:
    """Pairwise Pearson correlation matrix for numeric task properties."""

    features: list[str] = field(default_factory=list)
    correlations: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_tasks(cls, tasks: list[Task]) -> CorrelationMatrix:
        feature_names = [
            "depth_score",
            "parallel_score",
            "question_char_len",
            "question_word_len",
            "answer_char_len",
        ]
        if not tasks:
            return cls(features=feature_names, correlations={})

        vectors: dict[str, list[float]] = {
            "depth_score": [float(t.depth_score or 1) for t in tasks],
            "parallel_score": [float(t.parallel_score or 1) for t in tasks],
            "question_char_len": [float(len(t.question or "")) for t in tasks],
            "question_word_len": [float(len((t.question or "").split())) for t in tasks],
            "answer_char_len": [float(len(str(t.ground_truth or ""))) for t in tasks],
        }

        def _pearson(x: list[float], y: list[float]) -> float:
            n = len(x)
            if n <= 1:
                return 0.0
            mean_x = sum(x) / n
            mean_y = sum(y) / n
            var_x = sum((xi - mean_x) ** 2 for xi in x)
            var_y = sum((yi - mean_y) ** 2 for yi in y)
            if var_x == 0.0 or var_y == 0.0:
                return 0.0
            cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
            return round(cov / math.sqrt(var_x * var_y), 3)

        corrs: dict[str, dict[str, float]] = {}
        for f1 in feature_names:
            corrs[f1] = {}
            for f2 in feature_names:
                if f1 == f2:
                    corrs[f1][f2] = 1.0
                else:
                    corrs[f1][f2] = _pearson(vectors[f1], vectors[f2])

        return cls(features=feature_names, correlations=corrs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SplitStatistics:
    """Comprehensive statistical profile for a single task subset or overall dataset."""

    split_name: str
    total_tasks: int = 0
    source_distribution: DistributionTable = field(
        default_factory=lambda: DistributionTable(name="source_distribution")
    )
    depth_distribution: DistributionTable = field(
        default_factory=lambda: DistributionTable(name="depth_distribution")
    )
    parallel_distribution: DistributionTable = field(
        default_factory=lambda: DistributionTable(name="parallel_distribution")
    )
    question_char_stats: LengthStatistics = field(default_factory=LengthStatistics)
    question_word_stats: LengthStatistics = field(default_factory=LengthStatistics)
    answer_char_stats: LengthStatistics = field(default_factory=LengthStatistics)
    answer_word_stats: LengthStatistics = field(default_factory=LengthStatistics)
    context_char_stats: LengthStatistics = field(default_factory=LengthStatistics)
    lexical_stats: LexicalStatistics = field(default_factory=LexicalStatistics)

    @classmethod
    def from_tasks(cls, split_name: str, tasks: list[Task]) -> SplitStatistics:
        if not tasks:
            return cls(split_name=split_name)

        sources = [t.source_dataset for t in tasks]
        depths = [t.depth_score for t in tasks]
        parallels = [t.parallel_score for t in tasks]

        q_chars = [len(t.question or "") for t in tasks]
        q_words = [len((t.question or "").split()) for t in tasks]
        a_chars = [len(str(t.ground_truth or "")) for t in tasks]
        a_words = [len(str(t.ground_truth or "").split()) for t in tasks]
        ctx_chars = [len(t.context) for t in tasks if t.context]

        questions = [t.question for t in tasks if t.question]

        return cls(
            split_name=split_name,
            total_tasks=len(tasks),
            source_distribution=DistributionTable.from_values("source_distribution", sources),
            depth_distribution=DistributionTable.from_values("depth_distribution", depths),
            parallel_distribution=DistributionTable.from_values("parallel_distribution", parallels),
            question_char_stats=LengthStatistics.from_values(q_chars),
            question_word_stats=LengthStatistics.from_values(q_words),
            answer_char_stats=LengthStatistics.from_values(a_chars),
            answer_word_stats=LengthStatistics.from_values(a_words),
            context_char_stats=LengthStatistics.from_values(ctx_chars),
            lexical_stats=LexicalStatistics.from_texts(questions),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisReport:
    """Full dataset analysis report including overall metrics, splits, cross-tabs, and alerts."""

    dataset_name: str
    total_tasks: int
    overall: SplitStatistics
    splits: dict[str, SplitStatistics] = field(default_factory=dict)
    joint_depth_parallel: JointDistributionTable = field(
        default_factory=lambda: JointDistributionTable(
            row_feature="depth_score", col_feature="parallel_score"
        )
    )
    joint_source_depth: JointDistributionTable = field(
        default_factory=lambda: JointDistributionTable(
            row_feature="source_dataset", col_feature="depth_score"
        )
    )
    joint_source_parallel: JointDistributionTable = field(
        default_factory=lambda: JointDistributionTable(
            row_feature="source_dataset", col_feature="parallel_score"
        )
    )
    joint_split_source: JointDistributionTable = field(
        default_factory=lambda: JointDistributionTable(
            row_feature="split", col_feature="source_dataset"
        )
    )
    correlations: CorrelationMatrix = field(default_factory=CorrelationMatrix)
    imbalance_alerts: list[ImbalanceAlert] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Analyzer Engine
# ─────────────────────────────────────────────────────────────────────────────


class DatasetAnalyzer:
    """Analyzes GateOrchestra dataset tasks and splits to produce structured reports."""

    def __init__(
        self,
        min_acceptable_size: int = 150,
        gini_threshold: float = 0.40,
        max_class_ratio_threshold: float = 3.5,
    ) -> None:
        self.min_acceptable_size = min_acceptable_size
        self.gini_threshold = gini_threshold
        self.max_class_ratio_threshold = max_class_ratio_threshold

    def analyze_tasks(self, tasks: list[Task], split_name: str = "overall") -> SplitStatistics:
        """Compute statistical profile for a list of tasks."""
        return SplitStatistics.from_tasks(split_name, tasks)

    def compute_joint_distribution(
        self, tasks: list[Task], row_feature: str, col_feature: str
    ) -> JointDistributionTable:
        """Compute 2D contingency table for any pair of Task attributes."""
        pairs: list[tuple[Any, Any]] = []
        for t in tasks:
            r_val = getattr(t, row_feature, None)
            c_val = getattr(t, col_feature, None)
            pairs.append((r_val, c_val))
        return JointDistributionTable.from_pairs(row_feature, col_feature, pairs)

    @staticmethod
    def calculate_gini_coefficient(counts: list[int | float]) -> float:
        """Calculate Gini coefficient (0.0 = perfect equality, 1.0 = maximum inequality)."""
        valid = [float(x) for x in counts if x > 0]
        if not valid:
            return 0.0
        n = len(valid)
        if n == 1:
            return 0.0
        sorted_vals = sorted(valid)
        total = sum(sorted_vals)
        if total == 0:
            return 0.0
        cum_sum = 0.0
        for i, val in enumerate(sorted_vals, 1):
            cum_sum += i * val
        gini = (2.0 * cum_sum) / (n * total) - (n + 1.0) / n
        return round(max(0.0, min(1.0, gini)), 3)

    @staticmethod
    def calculate_shannon_entropy(counts: list[int | float]) -> float:
        """Calculate normalized Shannon entropy diversity index (0.0 to 1.0)."""
        valid = [float(x) for x in counts if x > 0]
        n_classes = len(counts)
        if n_classes <= 1 or not valid:
            return 1.0
        total = sum(valid)
        if total == 0:
            return 0.0
        h = -sum((c / total) * math.log2(c / total) for c in valid)
        max_h = math.log2(n_classes)
        return round(h / max_h, 3) if max_h > 0 else 1.0

    def check_imbalances(
        self,
        splits: dict[str, list[Task]] | None = None,
        all_tasks: list[Task] | None = None,
    ) -> list[ImbalanceAlert]:
        """Perform comprehensive imbalance audits across categories, score distributions, and splits."""
        alerts: list[ImbalanceAlert] = []

        if all_tasks is None and splits:
            all_tasks = [t for task_list in splits.values() for t in task_list]
        elif all_tasks is None:
            all_tasks = []

        total_count = len(all_tasks)

        # 1. Total Sample Size Check
        if total_count < self.min_acceptable_size:
            alerts.append(
                ImbalanceAlert(
                    severity="CRITICAL",
                    category="sample_size",
                    message=(
                        f"Dataset size ({total_count}) is below minimum target "
                        f"threshold of {self.min_acceptable_size}."
                    ),
                    metric_name="total_tasks",
                    observed_value=float(total_count),
                    threshold=float(self.min_acceptable_size),
                    recommendation="Generate additional seed tasks to ensure statistical validity.",
                )
            )

        # 2. Source Category Imbalance Check
        source_counts: dict[str, int] = {}
        for t in all_tasks:
            s = str(t.source_dataset)
            source_counts[s] = source_counts.get(s, 0) + 1

        if len(source_counts) > 1:
            vals = list(source_counts.values())
            max_val = max(vals)
            min_val = min(vals)
            ratio = round(max_val / min_val, 2) if min_val > 0 else 999.0
            gini = self.calculate_gini_coefficient(vals)
            entropy = self.calculate_shannon_entropy(vals)

            if ratio > self.max_class_ratio_threshold or gini > self.gini_threshold:
                alerts.append(
                    ImbalanceAlert(
                        severity="WARNING",
                        category="source_distribution",
                        message=(
                            f"Source category imbalance detected (Max/Min Ratio: {ratio:.1f}x, "
                            f"Gini: {gini}, Shannon Diversity: {entropy})."
                        ),
                        metric_name="source_max_min_ratio",
                        observed_value=ratio,
                        threshold=self.max_class_ratio_threshold,
                        recommendation="Re-balance generation seed pool or apply stratified sampling.",
                    )
                )
            else:
                alerts.append(
                    ImbalanceAlert(
                        severity="INFO",
                        category="source_distribution",
                        message=(
                            f"Source categories are well-balanced (Gini: {gini}, "
                            f"Diversity: {entropy}, Ratio: {ratio:.1f}x)."
                        ),
                        metric_name="source_gini",
                        observed_value=gini,
                        threshold=self.gini_threshold,
                        recommendation="No action needed.",
                    )
                )

        # 3. Depth Score Distribution Check
        depth_counts: dict[int, int] = {}
        for t in all_tasks:
            if t.depth_score is not None:
                depth_counts[t.depth_score] = depth_counts.get(t.depth_score, 0) + 1

        if depth_counts:
            d_vals = list(depth_counts.values())
            d_gini = self.calculate_gini_coefficient(d_vals)
            if d_gini > 0.45:
                alerts.append(
                    ImbalanceAlert(
                        severity="WARNING",
                        category="depth_distribution",
                        message=f"Depth score distribution is skewed (Gini: {d_gini}).",
                        metric_name="depth_gini",
                        observed_value=d_gini,
                        threshold=0.45,
                        recommendation="Inspect complexity feature extractor calibrations.",
                    )
                )

        # 4. Cross-Split Distribution Parity Check
        if splits and len(splits) >= 2:
            # Check source proportion parity between train and val/test
            train_tasks = splits.get("train", [])
            train_n = len(train_tasks)
            if train_n > 0:
                train_sources: dict[str, float] = {}
                for t in train_tasks:
                    s = str(t.source_dataset)
                    train_sources[s] = train_sources.get(s, 0) + 1
                train_props = {s: c / train_n for s, c in train_sources.items()}

                for eval_split in ["val", "test"]:
                    eval_tasks = splits.get(eval_split, [])
                    eval_n = len(eval_tasks)
                    if eval_n == 0:
                        continue
                    eval_sources: dict[str, float] = {}
                    for t in eval_tasks:
                        s = str(t.source_dataset)
                        eval_sources[s] = eval_sources.get(s, 0) + 1
                    eval_props = {s: c / eval_n for s, c in eval_sources.items()}

                    max_dev = 0.0
                    for s in set(train_props.keys()) | set(eval_props.keys()):
                        tp = train_props.get(s, 0.0)
                        ep = eval_props.get(s, 0.0)
                        max_dev = max(max_dev, abs(tp - ep))

                    if max_dev > 0.15:
                        alerts.append(
                            ImbalanceAlert(
                                severity="WARNING",
                                category="split_parity",
                                message=(
                                    f"Cross-split source drift detected between train and {eval_split} "
                                    f"(Max proportion delta: {max_dev * 100:.1f}%)."
                                ),
                                metric_name=f"train_{eval_split}_source_drift",
                                observed_value=round(max_dev, 3),
                                threshold=0.15,
                                recommendation="Run stratified split creation with balanced joint strata.",
                            )
                        )
                    else:
                        alerts.append(
                            ImbalanceAlert(
                                severity="INFO",
                                category="split_parity",
                                message=(
                                    f"Split '{eval_split}' is well-aligned with train distribution "
                                    f"(Max delta: {max_dev * 100:.1f}%)."
                                ),
                                metric_name=f"train_{eval_split}_source_drift",
                                observed_value=round(max_dev, 3),
                                threshold=0.15,
                                recommendation="No action needed.",
                            )
                        )

        return alerts

    def analyze_splits(
        self,
        splits: dict[str, list[Task]],
        dataset_name: str = "masbench_mini",
    ) -> AnalysisReport:
        """Run full analysis across splits and return an aggregated AnalysisReport."""
        all_tasks = [t for split_tasks in splits.values() for t in split_tasks]
        overall_stats = self.analyze_tasks(all_tasks, split_name="overall")

        split_stats_dict = {
            split_name: self.analyze_tasks(tasks, split_name=split_name)
            for split_name, tasks in splits.items()
        }

        # Joint 2D distributions
        joint_depth_parallel = self.compute_joint_distribution(
            all_tasks, "depth_score", "parallel_score"
        )
        joint_source_depth = self.compute_joint_distribution(
            all_tasks, "source_dataset", "depth_score"
        )
        joint_source_parallel = self.compute_joint_distribution(
            all_tasks, "source_dataset", "parallel_score"
        )

        # Split x Source joint distribution
        split_source_pairs: list[tuple[str, str]] = []
        for s_name, s_tasks in splits.items():
            for t in s_tasks:
                split_source_pairs.append((s_name, str(t.source_dataset)))
        joint_split_source = JointDistributionTable.from_pairs(
            "split", "source_dataset", split_source_pairs
        )

        correlations = CorrelationMatrix.from_tasks(all_tasks)
        imbalance_alerts = self.check_imbalances(splits=splits, all_tasks=all_tasks)

        return AnalysisReport(
            dataset_name=dataset_name,
            total_tasks=len(all_tasks),
            overall=overall_stats,
            splits=split_stats_dict,
            joint_depth_parallel=joint_depth_parallel,
            joint_source_depth=joint_source_depth,
            joint_source_parallel=joint_source_parallel,
            joint_split_source=joint_split_source,
            correlations=correlations,
            imbalance_alerts=imbalance_alerts,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Renderers and Exporters
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def render_ascii_table(dist: DistributionTable) -> str:
        """Render 1D DistributionTable as a formatted ASCII table."""
        lines = [
            f"+----------------------------------+-------+---------+",
            f"| {dist.name:<32} | Count | Percent |",
            f"+----------------------------------+-------+---------+",
        ]
        for k, count in dist.counts.items():
            pct = dist.percentages.get(k, 0.0)
            lines.append(f"| {k:<32} | {count:>5} | {pct:>6.1f}% |")
        lines.append(f"+----------------------------------+-------+---------+")
        lines.append(f"| {'Total':<32} | {dist.total:>5} |  100.0% |")
        lines.append(f"+----------------------------------+-------+---------+")
        return "\n".join(lines)

    @staticmethod
    def render_joint_ascii_table(joint: JointDistributionTable) -> str:
        """Render 2D JointDistributionTable as a formatted ASCII grid."""
        # Row label column width
        max_r_len = max([len(r) for r in joint.row_labels] + [len(joint.row_feature), 5])
        col_w = 8

        header = f"| {joint.row_feature:<{max_r_len}} |"
        for c in joint.col_labels:
            header += f" {c:>{col_w}} |"
        header += f" {'Total':>{col_w}} |"

        border = "+" + "-" * (max_r_len + 2) + "+"
        for _ in joint.col_labels:
            border += "-" * (col_w + 2) + "+"
        border += "-" * (col_w + 2) + "+"

        lines = [border, header, border]
        for r in joint.row_labels:
            row_line = f"| {r:<{max_r_len}} |"
            for c in joint.col_labels:
                val = joint.matrix.get(r, {}).get(c, 0)
                row_line += f" {val:>{col_w}} |"
            r_total = joint.row_totals.get(r, 0)
            row_line += f" {r_total:>{col_w}} |"
            lines.append(row_line)

        lines.append(border)
        footer = f"| {'Total':<{max_r_len}} |"
        for c in joint.col_labels:
            c_total = joint.col_totals.get(c, 0)
            footer += f" {c_total:>{col_w}} |"
        footer += f" {joint.total:>{col_w}} |"
        lines.append(footer)
        lines.append(border)
        return "\n".join(lines)

    def render_ascii_report(self, report: AnalysisReport) -> str:
        """Generate a complete plain-text / ASCII analysis summary for CLI printing."""
        lines = [
            "=" * 70,
            f"  GATEORCHESTRA DATASET ANALYSIS REPORT — {report.dataset_name.upper()}",
            "=" * 70,
            f"  Total Tasks Inspected : {report.total_tasks}",
            f"  Generated At (UTC)    : {report.generated_at}",
            "",
            "1. SPLIT SUMMARY",
            "-" * 40,
        ]
        for s_name, s_stat in report.splits.items():
            pct = (
                (s_stat.total_tasks / report.total_tasks * 100.0) if report.total_tasks > 0 else 0.0
            )
            lines.append(f"  * {s_name:<8} : {s_stat.total_tasks:>3} tasks ({pct:>5.1f}%)")

        lines.extend(["", "2. SOURCE CATEGORY DISTRIBUTION", "-" * 40])
        lines.append(self.render_ascii_table(report.overall.source_distribution))

        lines.extend(["", "3. DEPTH SCORE (1-5) DISTRIBUTION", "-" * 40])
        lines.append(self.render_ascii_table(report.overall.depth_distribution))

        lines.extend(["", "4. PARALLEL SCORE (1-4) DISTRIBUTION", "-" * 40])
        lines.append(self.render_ascii_table(report.overall.parallel_distribution))

        lines.extend(
            [
                "",
                "5. JOINT MATRIX: DEPTH SCORE x PARALLEL SCORE",
                "-" * 40,
                self.render_joint_ascii_table(report.joint_depth_parallel),
            ]
        )

        lines.extend(
            [
                "",
                "6. JOINT MATRIX: SOURCE DATASET x DEPTH SCORE",
                "-" * 40,
                self.render_joint_ascii_table(report.joint_source_depth),
            ]
        )

        lines.extend(
            [
                "",
                "7. JOINT MATRIX: SPLIT x SOURCE DATASET",
                "-" * 40,
                self.render_joint_ascii_table(report.joint_split_source),
            ]
        )

        lines.extend(["", "8. LENGTH AND COMPLEXITY METRICS", "-" * 40])
        q_len = report.overall.question_char_stats
        q_word = report.overall.question_word_stats
        a_len = report.overall.answer_char_stats
        lines.append(
            f"  * Question Characters : Mean={q_len.mean:.1f} +/- {q_len.std:.1f} "
            f"[Min={q_len.min:.0f}, Med={q_len.median:.0f}, Max={q_len.max:.0f}]"
        )
        lines.append(
            f"  * Question Words      : Mean={q_word.mean:.1f} +/- {q_word.std:.1f} "
            f"[Min={q_word.min:.0f}, Med={q_word.median:.0f}, Max={q_word.max:.0f}]"
        )
        lines.append(
            f"  * Answer Characters   : Mean={a_len.mean:.1f} +/- {a_len.std:.1f} "
            f"[Min={a_len.min:.0f}, Med={a_len.median:.0f}, Max={a_len.max:.0f}]"
        )

        lex = report.overall.lexical_stats
        lines.extend(
            [
                "",
                "9. LEXICAL DIVERSITY & VOCABULARY",
                "-" * 40,
                f"  * Total Tokens Inspected : {lex.total_tokens}",
                f"  * Unique Vocabulary Size : {lex.unique_tokens}",
                f"  * Type-Token Ratio (TTR) : {lex.type_token_ratio:.4f}",
                f"  * Top Question Keywords  : "
                + ", ".join(f"{t['word']} ({t['count']})" for t in lex.top_tokens[:8]),
            ]
        )

        lines.extend(["", "10. IMBALANCE AUDIT & ALERTS", "-" * 40])
        for alert in report.imbalance_alerts:
            prefix = f"[{alert.severity}]"
            lines.append(f"  {prefix:<10} {alert.message}")
            if alert.recommendation:
                lines.append(f"             -> Recommendation: {alert.recommendation}")

        lines.append("=" * 70)
        return "\n".join(lines)

    def render_markdown_report(self, report: AnalysisReport) -> str:
        """Generate a GitHub Flavored Markdown report with structured tables."""
        lines = [
            f"# GateOrchestra — Dataset Analysis Report",
            "",
            f"**Dataset Name:** `{report.dataset_name}`  ",
            f"**Total Tasks:** `{report.total_tasks}`  ",
            f"**Generated:** `{report.generated_at}`  ",
            "",
            "---",
            "",
            "## 1. Split Allocation Summary",
            "",
            "| Split | Task Count | Percentage |",
            "|---|---|---|",
        ]
        for s_name, s_stat in report.splits.items():
            pct = (
                (s_stat.total_tasks / report.total_tasks * 100.0) if report.total_tasks > 0 else 0.0
            )
            lines.append(f"| **{s_name}** | {s_stat.total_tasks} | {pct:.1f}% |")

        lines.extend(
            [
                f"| **Total** | **{report.total_tasks}** | **100.0%** |",
                "",
                "---",
                "",
                "## 2. 1D Distribution Tables",
                "",
                "### Source Datasets",
                "",
                "| Source | Count | Proportion |",
                "|---|---|---|",
            ]
        )
        for k, c in report.overall.source_distribution.counts.items():
            p = report.overall.source_distribution.percentages.get(k, 0.0)
            lines.append(f"| `{k}` | {c} | {p:.1f}% |")

        lines.extend(
            [
                "",
                "### Depth Scores (Reasoning Steps 1–5)",
                "",
                "| Depth Score | Count | Proportion |",
                "|---|---|---|",
            ]
        )
        for k, c in report.overall.depth_distribution.counts.items():
            p = report.overall.depth_distribution.percentages.get(k, 0.0)
            lines.append(f"| Level {k} | {c} | {p:.1f}% |")

        lines.extend(
            [
                "",
                "### Concurrency / Parallel Scores (1–4)",
                "",
                "| Parallel Score | Count | Proportion |",
                "|---|---|---|",
            ]
        )
        for k, c in report.overall.parallel_distribution.counts.items():
            p = report.overall.parallel_distribution.percentages.get(k, 0.0)
            lines.append(f"| Level {k} | {c} | {p:.1f}% |")

        # 2D Joint Contingency Tables
        lines.extend(
            [
                "",
                "---",
                "",
                "## 3. Joint Multi-Dimensional Matrices",
                "",
                "### Depth Score × Parallel Score (Reasoning Complexity vs Concurrency)",
                "",
            ]
        )

        joint_dp = report.joint_depth_parallel
        header = "| Depth \\ Parallel | " + " | ".join(f"**P={c}**" for c in joint_dp.col_labels) + " | **Total** |"
        sep = "|---" + "|---" * len(joint_dp.col_labels) + "|---|"
        lines.extend([header, sep])
        for r in joint_dp.row_labels:
            row_cells = [f"**D={r}**"]
            for c in joint_dp.col_labels:
                row_cells.append(str(joint_dp.matrix.get(r, {}).get(c, 0)))
            row_cells.append(f"**{joint_dp.row_totals.get(r, 0)}**")
            lines.append("| " + " | ".join(row_cells) + " |")

        col_tot_cells = ["**Total**"] + [
            f"**{joint_dp.col_totals.get(c, 0)}**" for c in joint_dp.col_labels
        ] + [f"**{joint_dp.total}**"]
        lines.append("| " + " | ".join(col_tot_cells) + " |")

        # Split x Source Joint Table
        lines.extend(
            [
                "",
                "### Split × Source Dataset Allocation",
                "",
            ]
        )
        joint_ss = report.joint_split_source
        header = "| Split | " + " | ".join(f"`{c}`" for c in joint_ss.col_labels) + " | **Total** |"
        sep = "|---" + "|---" * len(joint_ss.col_labels) + "|---|"
        lines.extend([header, sep])
        for r in joint_ss.row_labels:
            row_cells = [f"**{r}**"]
            for c in joint_ss.col_labels:
                row_cells.append(str(joint_ss.matrix.get(r, {}).get(c, 0)))
            row_cells.append(f"**{joint_ss.row_totals.get(r, 0)}**")
            lines.append("| " + " | ".join(row_cells) + " |")

        col_tot_cells = ["**Total**"] + [
            f"**{joint_ss.col_totals.get(c, 0)}**" for c in joint_ss.col_labels
        ] + [f"**{joint_ss.total}**"]
        lines.append("| " + " | ".join(col_tot_cells) + " |")

        # Length Statistics
        lines.extend(
            [
                "",
                "---",
                "",
                "## 4. Length & Token Complexity Profiling",
                "",
                "| Feature | Count | Mean ± Std | Min | 25% | Median | 75% | Max |",
                "|---|---|---|---|---|---|---|---|",
            ]
        )

        def _fmt_len_row(name: str, s: LengthStatistics) -> str:
            return (
                f"| **{name}** | {s.count} | {s.mean:.1f} ± {s.std:.1f} | "
                f"{s.min:.0f} | {s.p25:.0f} | {s.median:.0f} | {s.p75:.0f} | {s.max:.0f} |"
            )

        lines.append(_fmt_len_row("Question Chars", report.overall.question_char_stats))
        lines.append(_fmt_len_row("Question Words", report.overall.question_word_stats))
        lines.append(_fmt_len_row("Answer Chars", report.overall.answer_char_stats))
        lines.append(_fmt_len_row("Answer Words", report.overall.answer_word_stats))
        if report.overall.context_char_stats.count > 0:
            lines.append(_fmt_len_row("Context Chars", report.overall.context_char_stats))

        # Lexical Statistics
        lex = report.overall.lexical_stats
        lines.extend(
            [
                "",
                "---",
                "",
                "## 5. Lexical Diversity & Top Keywords",
                "",
                f"- **Total Tokens:** `{lex.total_tokens}`",
                f"- **Unique Vocabulary:** `{lex.unique_tokens}`",
                f"- **Type-Token Ratio (TTR):** `{lex.type_token_ratio:.4f}`",
                f"- **Average Word Length:** `{lex.avg_token_length:.2f} chars`",
                "",
                "| Keyword | Frequency |",
                "|---|---|",
            ]
        )
        for t in lex.top_tokens[:10]:
            lines.append(f"| `{t['word']}` | {t['count']} |")

        # Correlation Matrix
        lines.extend(
            [
                "",
                "---",
                "",
                "## 6. Correlation Matrix (Pearson r)",
                "",
            ]
        )
        corrs = report.correlations
        header = "| Metric | " + " | ".join(f"`{f}`" for f in corrs.features) + " |"
        sep = "|---" + "|---" * len(corrs.features) + "|"
        lines.extend([header, sep])
        for f1 in corrs.features:
            row_cells = [f"`{f1}`"]
            for f2 in corrs.features:
                val = corrs.correlations.get(f1, {}).get(f2, 0.0)
                row_cells.append(f"{val:+.3f}")
            lines.append("| " + " | ".join(row_cells) + " |")

        # Imbalance Audit
        lines.extend(
            [
                "",
                "---",
                "",
                "## 7. Imbalance Audit & Health Checks",
                "",
            ]
        )
        for alert in report.imbalance_alerts:
            alert_type = (
                "CAUTION"
                if alert.severity == "CRITICAL"
                else ("WARNING" if alert.severity == "WARNING" else "NOTE")
            )
            lines.extend(
                [
                    f"> [!{alert_type}]",
                    f"> **{alert.category.upper()} [{alert.severity}]:** {alert.message}",
                ]
            )
            if alert.recommendation:
                lines.append(f"> *Recommendation:* {alert.recommendation}")
            lines.append("")

        return "\n".join(lines)

    def export_report(
        self,
        report: AnalysisReport,
        json_path: Path | str,
        md_path: Path | str | None = None,
    ) -> None:
        """Export analysis report to JSON and optionally Markdown."""
        json_file = Path(json_path)
        json_file.parent.mkdir(parents=True, exist_ok=True)
        with json_file.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved JSON analysis report to {json_file}")

        if md_path is not None:
            md_file = Path(md_path)
            md_file.parent.mkdir(parents=True, exist_ok=True)
            with md_file.open("w", encoding="utf-8") as f:
                f.write(self.render_markdown_report(report))
            logger.info(f"Saved Markdown analysis report to {md_file}")

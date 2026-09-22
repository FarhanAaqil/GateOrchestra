"""
tests/test_dataset_analysis.py
==============================
Unit and integration tests for dataset analysis and statistical profiling engine (Week 7).

Person 1 owns this test suite.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from dataset.analysis import (
    CorrelationMatrix,
    DatasetAnalyzer,
    DistributionTable,
    JointDistributionTable,
    LengthStatistics,
    LexicalStatistics,
)
from shared.schemas import Task


@pytest.fixture
def sample_tasks() -> list[Task]:
    """Sample diverse tasks for testing statistical engine."""
    return [
        Task(
            task_id="test_001",
            question="What is the capital of France and what is its population?",
            ground_truth="Paris, 2.1 million",
            source_dataset="hotpotqa_style",
            depth_score=3,
            parallel_score=2,
            context="Paris is the capital of France.",
        ),
        Task(
            task_id="test_002",
            question="Who directed Inception and when was it released?",
            ground_truth="Christopher Nolan, 2010",
            source_dataset="hotpotqa_style",
            depth_score=3,
            parallel_score=2,
            context="Inception was directed by Christopher Nolan in 2010.",
        ),
        Task(
            task_id="test_003",
            question="Calculate 15 multiplied by 4 plus 10.",
            ground_truth="70",
            source_dataset="template_arithmetic",
            depth_score=2,
            parallel_score=1,
            context="15 * 4 = 60, 60 + 10 = 70.",
        ),
        Task(
            task_id="test_004",
            question="Which river is longer: the Nile or the Amazon?",
            ground_truth="The Nile",
            source_dataset="template_comparison",
            depth_score=1,
            parallel_score=1,
            context="The Nile is 6650 km and the Amazon is 6400 km.",
        ),
        Task(
            task_id="test_005",
            question="Find the mother of the father of Queen Victoria.",
            ground_truth="Princess Augusta of Saxe-Gotha",
            source_dataset="musique_style",
            depth_score=4,
            parallel_score=3,
            context="Prince Edward Duke of Kent was father of Queen Victoria.",
        ),
    ]


@pytest.fixture
def sample_splits(sample_tasks: list[Task]) -> dict[str, list[Task]]:
    """Sample train/val/test splits."""
    return {
        "train": sample_tasks[:3],
        "val": [sample_tasks[3]],
        "test": [sample_tasks[4]],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. LengthStatistics Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLengthStatistics:
    def test_empty_values(self):
        stats = LengthStatistics.from_values([])
        assert stats.count == 0
        assert stats.mean == 0.0
        assert stats.min == 0.0
        assert stats.max == 0.0

    def test_single_value(self):
        stats = LengthStatistics.from_values([42])
        assert stats.count == 1
        assert stats.mean == 42.0
        assert stats.std == 0.0
        assert stats.min == 42.0
        assert stats.median == 42.0
        assert stats.max == 42.0

    def test_known_values(self):
        values = [10, 20, 30, 40, 50]
        stats = LengthStatistics.from_values(values)
        assert stats.count == 5
        assert stats.mean == 30.0
        assert stats.min == 10.0
        assert stats.median == 30.0
        assert stats.max == 50.0
        assert stats.p25 == 20.0
        assert stats.p75 == 40.0
        assert stats.std > 0.0

    def test_to_dict(self):
        stats = LengthStatistics.from_values([1, 2, 3])
        d = stats.to_dict()
        assert isinstance(d, dict)
        assert "mean" in d
        assert "median" in d
        assert d["count"] == 3


# ─────────────────────────────────────────────────────────────────────────────
# 2. DistributionTable Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDistributionTable:
    def test_empty_distribution(self):
        dist = DistributionTable.from_values("empty", [])
        assert dist.total == 0
        assert dist.counts == {}
        assert dist.percentages == {}

    def test_counts_and_percentages(self):
        values = ["A", "B", "A", "A", "B"]
        dist = DistributionTable.from_values("test_cat", values)
        assert dist.total == 5
        assert dist.counts["A"] == 3
        assert dist.counts["B"] == 2
        assert dist.percentages["A"] == 60.0
        assert dist.percentages["B"] == 40.0

    def test_numeric_keys_sorting(self):
        values = [3, 1, 2, 1, 3]
        dist = DistributionTable.from_values("scores", values)
        keys = list(dist.counts.keys())
        assert keys == ["1", "2", "3"]

    def test_to_dict(self):
        dist = DistributionTable.from_values("cat", ["x", "y"])
        d = dist.to_dict()
        assert d["name"] == "cat"
        assert d["total"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# 3. JointDistributionTable Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestJointDistributionTable:
    def test_empty_pairs(self):
        joint = JointDistributionTable.from_pairs("row", "col", [])
        assert joint.total == 0
        assert joint.row_labels == []
        assert joint.col_labels == []

    def test_matrix_and_marginals(self):
        pairs = [
            ("cat1", 1),
            ("cat1", 2),
            ("cat2", 1),
            ("cat2", 1),
        ]
        joint = JointDistributionTable.from_pairs("source", "depth", pairs)
        assert joint.total == 4
        assert "cat1" in joint.row_labels
        assert "cat2" in joint.row_labels
        assert "1" in joint.col_labels
        assert "2" in joint.col_labels
        assert joint.matrix["cat1"]["1"] == 1
        assert joint.matrix["cat1"]["2"] == 1
        assert joint.matrix["cat2"]["1"] == 2
        assert joint.matrix["cat2"]["2"] == 0
        assert joint.row_totals["cat1"] == 2
        assert joint.row_totals["cat2"] == 2
        assert joint.col_totals["1"] == 3
        assert joint.col_totals["2"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 4. LexicalStatistics Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLexicalStatistics:
    def test_empty_texts(self):
        lex = LexicalStatistics.from_texts([])
        assert lex.total_tokens == 0
        assert lex.unique_tokens == 0
        assert lex.type_token_ratio == 0.0

    def test_lexical_computation(self):
        texts = [
            "Quantum computing uses quantum bits.",
            "Classical computing uses classical bits.",
        ]
        lex = LexicalStatistics.from_texts(texts)
        assert lex.total_tokens == 10
        assert lex.unique_tokens > 0
        assert 0.0 < lex.type_token_ratio <= 1.0
        assert lex.avg_token_length > 0.0
        # Check stopwords exclusion in top tokens
        top_words = [t["word"] for t in lex.top_tokens]
        assert "computing" in top_words or "quantum" in top_words


# ─────────────────────────────────────────────────────────────────────────────
# 5. Imbalance Metrics Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestImbalanceMetrics:
    def test_gini_coefficient_equal(self):
        counts = [25, 25, 25, 25]
        gini = DatasetAnalyzer.calculate_gini_coefficient(counts)
        assert gini == 0.0

    def test_gini_coefficient_unequal(self):
        counts = [100, 1, 1, 1]
        gini = DatasetAnalyzer.calculate_gini_coefficient(counts)
        assert gini > 0.50

    def test_shannon_entropy_equal(self):
        counts = [10, 10, 10, 10]
        entropy = DatasetAnalyzer.calculate_shannon_entropy(counts)
        assert pytest.approx(entropy, 0.01) == 1.0

    def test_shannon_entropy_skewed(self):
        counts = [95, 2, 2, 1]
        entropy = DatasetAnalyzer.calculate_shannon_entropy(counts)
        assert entropy < 0.50

    def test_imbalance_alerts_small_dataset(self, sample_tasks: list[Task]):
        analyzer = DatasetAnalyzer(min_acceptable_size=100)
        alerts = analyzer.check_imbalances(all_tasks=sample_tasks)
        crit_alerts = [a for a in alerts if a.severity == "CRITICAL"]
        assert len(crit_alerts) >= 1
        assert "below minimum target" in crit_alerts[0].message


# ─────────────────────────────────────────────────────────────────────────────
# 6. CorrelationMatrix Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCorrelationMatrix:
    def test_empty_tasks(self):
        matrix = CorrelationMatrix.from_tasks([])
        assert "depth_score" in matrix.features
        assert matrix.correlations == {}

    def test_diagonal_identity(self, sample_tasks: list[Task]):
        matrix = CorrelationMatrix.from_tasks(sample_tasks)
        for f in matrix.features:
            assert matrix.correlations[f][f] == 1.0

    def test_correlation_range(self, sample_tasks: list[Task]):
        matrix = CorrelationMatrix.from_tasks(sample_tasks)
        for f1 in matrix.features:
            for f2 in matrix.features:
                val = matrix.correlations[f1][f2]
                assert -1.0 <= val <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. DatasetAnalyzer & Report Generation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDatasetAnalyzer:
    def test_analyze_tasks(self, sample_tasks: list[Task]):
        analyzer = DatasetAnalyzer()
        stats = analyzer.analyze_tasks(sample_tasks, split_name="sample")
        assert stats.split_name == "sample"
        assert stats.total_tasks == 5
        assert stats.source_distribution.total == 5
        assert stats.depth_distribution.total == 5
        assert stats.question_char_stats.count == 5

    def test_analyze_splits(self, sample_splits: dict[str, list[Task]]):
        analyzer = DatasetAnalyzer(min_acceptable_size=5)
        report = analyzer.analyze_splits(sample_splits, dataset_name="test_bench")
        assert report.dataset_name == "test_bench"
        assert report.total_tasks == 5
        assert len(report.splits) == 3
        assert report.joint_depth_parallel.total == 5
        assert report.joint_source_depth.total == 5
        assert report.joint_split_source.total == 5

    def test_render_ascii_report(self, sample_splits: dict[str, list[Task]]):
        analyzer = DatasetAnalyzer(min_acceptable_size=5)
        report = analyzer.analyze_splits(sample_splits)
        ascii_text = analyzer.render_ascii_report(report)
        assert "GATEORCHESTRA DATASET ANALYSIS REPORT" in ascii_text
        assert "SOURCE CATEGORY DISTRIBUTION" in ascii_text
        assert "JOINT MATRIX" in ascii_text

    def test_render_markdown_report(self, sample_splits: dict[str, list[Task]]):
        analyzer = DatasetAnalyzer(min_acceptable_size=5)
        report = analyzer.analyze_splits(sample_splits)
        md_text = analyzer.render_markdown_report(report)
        assert "# GateOrchestra — Dataset Analysis Report" in md_text
        assert "## 1. Split Allocation Summary" in md_text
        assert "## 3. Joint Multi-Dimensional Matrices" in md_text

    def test_export_report_files(self, sample_splits: dict[str, list[Task]], tmp_path: Path):
        analyzer = DatasetAnalyzer(min_acceptable_size=5)
        report = analyzer.analyze_splits(sample_splits)
        json_out = tmp_path / "analysis.json"
        md_out = tmp_path / "analysis.md"

        analyzer.export_report(report, json_path=json_out, md_path=md_out)
        assert json_out.exists()
        assert md_out.exists()

        with json_out.open("r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["total_tasks"] == 5
            assert "overall" in data


# ─────────────────────────────────────────────────────────────────────────────
# 8. CLI Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCLIIntegration:
    def test_cli_help(self):
        res = subprocess.run(
            [sys.executable, "scripts/dataset_stats.py", "--help"],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        assert "--split" in res.stdout
        assert "--output-json" in res.stdout

    def test_cli_run_on_existing_splits(self):
        res = subprocess.run(
            [sys.executable, "scripts/dataset_stats.py", "--no-save"],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        assert "GATEORCHESTRA DATASET ANALYSIS REPORT" in res.stdout

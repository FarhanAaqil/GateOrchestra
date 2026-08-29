"""
tests/test_dataset_labeling.py
================================
Unit and integration tests for the dataset labeling pipeline (Week 2):
  - feature_extractor.py
  - depth_labeler.py
  - parallel_labeler.py
  - build_dataset.py integration & schema validation
"""

import json

from dataset.generation.task_builder import build_tasks
from dataset.generation.task_pool import RAW_TASKS
from dataset.labeling.depth_labeler import assign_depth
from dataset.labeling.feature_extractor import (
    _count_choice_entities,
    _count_clauses,
    _count_conjunctions,
    _count_list_items,
    _count_sub_questions,
    _detect_arithmetic,
    _detect_comparison,
    extract_labeling_features,
)
from dataset.labeling.parallel_labeler import assign_parallel
from shared.schemas import Task

# ─────────────────────────────────────────────────────────────────────────────
# 1. Feature Extractor Unit Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFeatureExtractor:
    def test_extract_labeling_features_keys(self):
        features = extract_labeling_features(
            question="Which is larger: the Pacific Ocean or the Atlantic Ocean?",
            context="The Pacific Ocean is the largest ocean.",
            hop_hint=2,
        )
        expected_keys = {
            "question_word_count",
            "entity_count",
            "clause_count",
            "hop_count",
            "sub_question_count",
            "conjunction_count",
            "list_count",
            "has_context",
            "context_word_count",
            "is_comparison",
            "is_arithmetic",
            "choice_count",
            "parallel_branches",
        }
        assert expected_keys.issubset(set(features.keys()))
        assert features["has_context"] is True
        assert features["context_word_count"] > 0
        assert features["hop_count"] == 2
        assert features["is_comparison"] is True

    def test_detect_comparison_queries(self):
        comp_q1 = "Which is deeper on average: the Pacific or the Atlantic?"
        comp_q2 = "Who ruled longer: Queen Victoria of Britain or King Louis XIV of France?"
        comp_q3 = "Which telescope was launched first: Hubble or James Webb?"
        non_comp = "What is the capital of France?"

        assert _detect_comparison(comp_q1) is True
        assert _detect_comparison(comp_q2) is True
        assert _detect_comparison(comp_q3) is True
        assert _detect_comparison(non_comp) is False

    def test_detect_arithmetic_queries(self):
        arith_q1 = "If a car travels at 60 km/h for 2.5 hours, what is the total distance?"
        arith_q2 = "Convert 100 meters to feet and calculate the sum."
        non_arith = "Who was the first president of the United States?"

        assert _detect_arithmetic(arith_q1) is True
        assert _detect_arithmetic(arith_q2) is True
        assert _detect_arithmetic(non_arith) is False

    def test_count_choice_entities(self):
        assert _count_choice_entities("Which is older: Python or Java?") == 2
        assert (
            _count_choice_entities(
                "Between Apollo 11, Apollo 12, and Apollo 13, which landed first?"
            )
            == 3
        )
        assert _count_choice_entities("What is the capital of Canada?") == 0

    def test_count_clauses(self):
        multi_clause = (
            "Who was the director of the film that won Best Picture when Parasite was released?"
        )
        assert _count_clauses(multi_clause) >= 2

        simple = "What is 2 + 2?"
        assert _count_clauses(simple) == 0

    def test_count_sub_questions(self):
        multi_q = "What is the capital of France? And what is its population?"
        assert _count_sub_questions(multi_q) >= 1

        single_q = "What is the birth city of Marie Curie?"
        assert _count_sub_questions(single_q) == 0

    def test_count_conjunctions_and_lists(self):
        text = (
            "Find both the length and width, and compute first the area and second the perimeter."
        )
        assert _count_conjunctions(text) >= 2
        assert _count_list_items(text) >= 2

    def test_empty_and_whitespace_question(self):
        features = extract_labeling_features(question="   ", context=None, hop_hint=1)
        assert features["question_word_count"] == 0
        assert features["has_context"] is False
        assert features["entity_count"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Depth Labeler Unit Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDepthLabeler:
    def test_assign_depth_bounds(self):
        for hop in [1, 2, 3, 4]:
            for clauses in [0, 1, 3]:
                for entities in [0, 2, 5]:
                    features = {
                        "hop_count": hop,
                        "clause_count": clauses,
                        "entity_count": entities,
                        "question_word_count": 15,
                    }
                    result = assign_depth(features)
                    score = result["depth_score"]
                    assert isinstance(score, int)
                    assert 1 <= score <= 5

    def test_depth_monotonic_with_hops(self):
        base_features_1 = {
            "hop_count": 1,
            "clause_count": 0,
            "entity_count": 0,
            "question_word_count": 8,
        }
        base_features_2 = {
            "hop_count": 2,
            "clause_count": 1,
            "entity_count": 2,
            "question_word_count": 14,
        }
        base_features_3 = {
            "hop_count": 3,
            "clause_count": 2,
            "entity_count": 3,
            "question_word_count": 20,
        }

        r1 = assign_depth(base_features_1)
        r2 = assign_depth(base_features_2)
        r3 = assign_depth(base_features_3)

        assert r1["depth_raw_score"] < r2["depth_raw_score"] < r3["depth_raw_score"]
        assert r1["depth_score"] <= r2["depth_score"] <= r3["depth_score"]

    def test_custom_weights_and_thresholds(self):
        features = {"hop_count": 1, "clause_count": 0, "entity_count": 0, "question_word_count": 5}
        custom_weights = {
            "hop_count": 1.0,
            "clause_count": 0.0,
            "entity_count": 0.0,
            "word_count_norm": 0.0,
        }
        custom_thresholds = [(0.5, 1), (1.5, 3), (float("inf"), 5)]

        res = assign_depth(features, weights=custom_weights, thresholds=custom_thresholds)
        # raw_score = 1.0 -> 0.5 <= 1.0 < 1.5 -> label 3
        assert res["depth_score"] == 3
        assert res["depth_raw_score"] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Parallel Labeler Unit Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestParallelLabeler:
    def test_assign_parallel_bounds(self):
        for sub_q in [0, 1, 2, 4]:
            for conj in [0, 1, 3]:
                for lists in [0, 2]:
                    for branches in [1, 2, 3]:
                        features = {
                            "sub_question_count": sub_q,
                            "conjunction_count": conj,
                            "list_count": lists,
                            "choice_count": 2 if branches > 1 else 0,
                            "parallel_branches": branches,
                        }
                        result = assign_parallel(features)
                        score = result["parallel_score"]
                        assert isinstance(score, int)
                        assert 1 <= score <= 4

    def test_sequential_vs_comparison_parallelism(self):
        seq_features = extract_labeling_features(
            question="What is the capital of Australia?",
            hop_hint=1,
        )
        comp_features = extract_labeling_features(
            question="Which has more Nobel Prizes: the United States or the United Kingdom?",
            hop_hint=2,
        )

        seq_res = assign_parallel(seq_features)
        comp_res = assign_parallel(comp_features)

        assert seq_res["parallel_score"] == 1
        assert comp_res["parallel_score"] >= 2
        assert comp_res["parallel_raw_score"] > seq_res["parallel_raw_score"]

    def test_multi_sub_question_parallelism(self):
        multi_q_features = extract_labeling_features(
            question="What is the population of Tokyo? And what is the population of Delhi?",
            hop_hint=2,
        )
        res = assign_parallel(multi_q_features)
        assert res["parallel_score"] >= 2


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pipeline Integration & Schema Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLabelingPipelineIntegration:
    def test_raw_tasks_labeling_validity(self):
        tasks, rejected = build_tasks(raw_tasks=RAW_TASKS, verbose=False)
        assert len(tasks) >= 150
        assert len(rejected) == 0

        for task in tasks:
            features = extract_labeling_features(
                question=task.question,
                context=task.context,
                hop_hint=1,
            )
            d_res = assign_depth(features)
            p_res = assign_parallel(features)

            # Validate that Task model accepts the produced scores
            labeled_task = Task(
                task_id=task.task_id,
                question=task.question,
                context=task.context,
                depth_score=d_res["depth_score"],
                parallel_score=p_res["parallel_score"],
                ground_truth=task.ground_truth,
                source_dataset=task.source_dataset,
            )
            assert 1 <= labeled_task.depth_score <= 5
            assert 1 <= labeled_task.parallel_score <= 4

    def test_feature_matrix_serializable(self):
        features = extract_labeling_features("Who directed Inception?", hop_hint=2)
        d_res = assign_depth(features)
        p_res = assign_parallel(features)

        row = {
            "task_id": "test_001",
            **features,
            "depth_score": d_res["depth_score"],
            "depth_raw_score": d_res["depth_raw_score"],
            "parallel_score": p_res["parallel_score"],
            "parallel_raw_score": p_res["parallel_raw_score"],
        }
        # Must serialize cleanly to JSON
        json_str = json.dumps(row)
        loaded = json.loads(json_str)
        assert loaded["task_id"] == "test_001"

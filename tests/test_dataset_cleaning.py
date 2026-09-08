"""
tests/test_dataset_cleaning.py
==============================
Unit and integration tests for dataset cleaning and deduplication (Week 3):
  - text normalization & unicode sanitization
  - answer canonicalization & boolean standardizing
  - question formatting
  - exact & fuzzy deduplication
  - batch cleaning audit logging
"""

from dataset.cleaning.cleaner import (
    canonicalize_answer,
    clean_dataset_records,
    format_question,
    normalize_text,
)
from dataset.cleaning.deduplicator import (
    FuzzyDeduplicator,
    compute_question_similarity,
    exact_fingerprint,
)
from shared.schemas import Task


class TestTextNormalization:
    def test_unicode_and_smart_quotes(self):
        raw = "“What is the ‘capital’ of France?” — asked the student."
        cleaned = normalize_text(raw)
        assert '"' in cleaned
        assert "'" in cleaned
        assert "“" not in cleaned
        assert "”" not in cleaned
        assert "‘" not in cleaned
        assert "’" not in cleaned
        assert "—" not in cleaned
        assert "-" in cleaned

    def test_zero_width_and_invisible_chars(self):
        raw = "What\u200b is \ufeffthe\u00a0capital?"
        cleaned = normalize_text(raw)
        assert cleaned == "What is the capital?"
        assert "\u200b" not in cleaned
        assert "\ufeff" not in cleaned

    def test_whitespace_collapsing(self):
        raw = "   Which   country    has  the   largest    area? \n\t  "
        cleaned = normalize_text(raw)
        assert cleaned == "Which country has the largest area?"

    def test_empty_string(self):
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""


class TestAnswerCanonicalization:
    def test_boolean_standardization(self):
        assert canonicalize_answer("true") == "True"
        assert canonicalize_answer("FALSE") == "False"
        assert canonicalize_answer("yes") == "Yes"
        assert canonicalize_answer("no") == "No"

    def test_surrounding_quotes_removal(self):
        assert canonicalize_answer('"Paris"') == "Paris"
        assert canonicalize_answer("'Tokyo'") == "Tokyo"
        assert canonicalize_answer('  "Berlin"  ') == "Berlin"

    def test_trailing_period_stripping(self):
        assert canonicalize_answer("London.") == "London"
        assert canonicalize_answer("42.") == "42"
        # Preserve acronyms
        assert canonicalize_answer("Washington, D.C.") == "Washington, D.C."

    def test_empty_answer(self):
        assert canonicalize_answer("") == ""


class TestQuestionFormatting:
    def test_enforce_question_mark_for_interrogatives(self):
        assert format_question("Who directed Inception") == "Who directed Inception?"
        assert (
            format_question("Which is larger: Mars or Venus") == "Which is larger: Mars or Venus?"
        )
        assert (
            format_question("Calculate the sum of 10 and 20") == "Calculate the sum of 10 and 20?"
        )

    def test_already_punctuated(self):
        assert format_question("Who directed Inception?") == "Who directed Inception?"
        assert format_question("Find the capital.") == "Find the capital."


class TestFuzzyDeduplication:
    def test_exact_fingerprint(self):
        q1 = "What is the capital of France?"
        q2 = "  what IS the capital of france?  "
        assert exact_fingerprint(q1) == exact_fingerprint(q2)

    def test_fuzzy_similarity_scoring(self):
        q1 = "Which is larger: the Pacific Ocean or the Atlantic Ocean?"
        q2 = "Which ocean is larger: the Pacific Ocean or the Atlantic Ocean?"
        q3 = "What is the atomic number of Gold?"

        sim_high = compute_question_similarity(q1, q2)
        sim_low = compute_question_similarity(q1, q3)

        assert sim_high > 0.80
        assert sim_low < 0.30

    def test_deduplicator_removes_exact_and_fuzzy(self):
        t1 = Task(
            task_id="t1",
            question="What is the speed of light in a vacuum?",
            ground_truth="299,792,458 m/s",
            source_dataset="hotpotqa_style",
            depth_score=1,
            parallel_score=1,
        )
        t2 = Task(
            task_id="t2",
            question="what is the speed of light in a vacuum?",  # Exact dup
            ground_truth="299,792,458 m/s",
            source_dataset="hotpotqa_style",
            depth_score=1,
            parallel_score=1,
        )
        t3 = Task(
            task_id="t3",
            question="What is the speed of light in vacuum?",  # Fuzzy dup
            ground_truth="299,792,458 m/s",
            source_dataset="hotpotqa_style",
            depth_score=1,
            parallel_score=1,
        )
        t4 = Task(
            task_id="t4",
            question="Who painted the Mona Lisa?",  # Unique
            ground_truth="Leonardo da Vinci",
            source_dataset="hotpotqa_style",
            depth_score=1,
            parallel_score=1,
        )

        dedup = FuzzyDeduplicator(similarity_threshold=0.85)
        unique, exact_dups, fuzzy_dups = dedup.deduplicate([t1, t2, t3, t4])

        assert len(unique) == 2
        assert unique[0].task_id == "t1"
        assert unique[1].task_id == "t4"
        assert len(exact_dups) == 1
        assert len(fuzzy_dups) == 1


class TestCleanDatasetPipeline:
    def test_clean_dataset_records(self):
        tasks = [
            Task(
                task_id="t1",
                question="“Who was the first president of the US” \u200b",
                ground_truth='"George Washington."',
                source_dataset="hotpotqa_style",
                depth_score=1,
                parallel_score=1,
            ),
            Task(
                task_id="t2",
                question="What is 5 + 5?",
                ground_truth="10",
                source_dataset="template_arithmetic",
                depth_score=1,
                parallel_score=1,
            ),
        ]

        cleaned_tasks, summary = clean_dataset_records(tasks)
        assert len(cleaned_tasks) == 2
        assert summary["tasks_modified_count"] >= 1
        assert cleaned_tasks[0].question == '"Who was the first president of the US"?'
        assert cleaned_tasks[0].ground_truth == "George Washington"
        assert cleaned_tasks[1].ground_truth == "10"

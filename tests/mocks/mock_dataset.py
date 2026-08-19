"""Mock dataset — 10 toy Task objects covering different complexity levels."""

from __future__ import annotations

from shared.schemas import Task

MOCK_TASKS: list[Task] = [
    Task(
        task_id="simple_001",
        question="What is the capital of France?",
        ground_truth="Paris",
        depth_score=1,
        parallel_score=1,
        source_dataset="template_arithmetic",
    ),
    Task(
        task_id="simple_002",
        question="Who wrote Romeo and Juliet?",
        ground_truth="William Shakespeare",
        depth_score=1,
        parallel_score=1,
        source_dataset="template_arithmetic",
    ),
    Task(
        task_id="medium_001",
        question="What was the primary cause of World War I and when did it start?",
        ground_truth="Assassination of Archduke Franz Ferdinand, 1914",
        depth_score=2,
        parallel_score=2,
        source_dataset="hotpotqa",
    ),
    Task(
        task_id="medium_002",
        question="Which country invented paper and what dynasty ruled during that period?",
        ground_truth="China, Han dynasty",
        depth_score=2,
        parallel_score=2,
        source_dataset="musique",
    ),
    Task(
        task_id="deep_001",
        question=(
            "The scientist who discovered radioactivity was born in the same country as "
            "the mathematician who proved Fermat's Last Theorem. What is that country?"
        ),
        context="Marie Curie discovered radioactivity. Andrew Wiles proved Fermat's Last Theorem.",
        ground_truth="France",
        depth_score=4,
        parallel_score=3,
        source_dataset="hotpotqa",
    ),
    Task(
        task_id="deep_002",
        question=(
            "What is the GDP of the country whose president won the Nobel Peace Prize "
            "in 2009 and also served as the first African American president?"
        ),
        ground_truth="United States",
        depth_score=4,
        parallel_score=2,
        source_dataset="musique",
    ),
    Task(
        task_id="parallel_001",
        question="List the capitals of France, Germany, and Japan.",
        ground_truth="Paris, Berlin, Tokyo",
        depth_score=1,
        parallel_score=4,
        source_dataset="template_arithmetic",
    ),
    Task(
        task_id="parallel_002",
        question="Who founded Apple, Microsoft, and Amazon respectively?",
        ground_truth="Steve Jobs, Bill Gates, Jeff Bezos",
        depth_score=2,
        parallel_score=4,
        source_dataset="template_arithmetic",
    ),
    Task(
        task_id="multihop_001",
        question="What is the birth city of the person who invented the telephone?",
        context="Alexander Graham Bell invented the telephone.",
        ground_truth="Edinburgh",
        depth_score=3,
        parallel_score=1,
        source_dataset="hotpotqa",
    ),
    Task(
        task_id="multihop_002",
        question=(
            "The film that won Best Picture at the 2020 Oscars was directed by "
            "a director from which country?"
        ),
        ground_truth="South Korea",
        depth_score=3,
        parallel_score=1,
        source_dataset="musique",
    ),
]


def get_mock_tasks(n: int | None = None) -> list[Task]:
    """Return mock tasks (all 10, or first n if specified)."""
    tasks = MOCK_TASKS
    if n is not None:
        tasks = tasks[:n]
    return tasks

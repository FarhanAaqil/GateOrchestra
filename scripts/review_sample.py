"""
scripts/review_sample.py
=========================
Interactive Manual Review CLI — Week 6 Deliverable (Person 1).

Presents each sampled task one at a time for human review, records the
verdict in ``dataset/reviewed/review_log.jsonl``, and produces a final
progress report.

Review verdicts:
    a  approve        — task is correct as-is
    f  fix_answer     — ground_truth needs a correction (prompts for fix)
    q  fix_question   — question wording is ambiguous or malformed
    r  flag_remove    — task should be dropped from the dataset
    s  skip           — cannot judge right now (reviewable again later)

Keyboard shortcuts during review:
    Enter  — confirm current input
    Ctrl+C — interrupt and save progress report

Usage::

    # Review the default 20% sample (all splits, seed=42)
    python scripts/review_sample.py

    # Set a custom sample rate
    python scripts/review_sample.py --sample-rate 0.10

    # Resume a previous session (skips already-reviewed tasks)
    python scripts/review_sample.py --resume

    # Set reviewer name (recorded in log)
    python scripts/review_sample.py --reviewer "umar"

    # Dry-run: sample only, print summary, do not write log
    python scripts/review_sample.py --dry-run

    # Export all flagged tasks to CSV at end
    python scripts/review_sample.py --export-flagged

    # Review only a specific split
    python scripts/review_sample.py --split train

Person 1 owns this file.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── Bootstrap sys.path so we can import shared.* and dataset.* ───────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.repository import JSONLTaskRepository  # noqa: E402
from dataset.review.reviewer import VERDICT_LABELS, ReviewEntry, ReviewLog  # noqa: E402
from dataset.review.sampler import ReviewSampler  # noqa: E402
from shared.schemas import Task  # noqa: E402

logging.basicConfig(
    level=logging.WARNING,  # Keep CLI output clean; only warnings+
    format="%(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ─── ANSI colours (gracefully degraded on Windows if not supported) ───────────
try:
    import os

    _USE_COLOR = os.name != "nt" or "ANSICON" in os.environ or "WT_SESSION" in os.environ
except Exception:
    _USE_COLOR = False


def _c(text: str, code: str) -> str:
    """Wrap text in an ANSI colour code if colours are enabled."""
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


BOLD = "1"
GREEN = "32"
YELLOW = "33"
RED = "31"
CYAN = "36"
DIM = "2"


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────


def _display_task(task: Task, index: int, total: int, split: str | None) -> None:
    """Render a task card to stdout."""
    split_label = f"[{split}]" if split else ""
    print(f"\n{'─'*60}")
    print(_c(f"  Task {index}/{total}  {split_label}", BOLD) + _c(f"  ID: {task.task_id}", DIM))
    print(f"{'─'*60}")
    print(_c("  Question:", BOLD))
    print(f"    {task.question}")

    if task.context:
        print(_c("  Context:", BOLD))
        # Truncate long contexts
        ctx = task.context if len(task.context) <= 300 else task.context[:300] + "..."
        print(f"    {ctx}")

    print(_c("  Ground truth:", BOLD))
    print(f"    {task.ground_truth or '(none)'}")

    print(_c("  Labels:", BOLD))
    print(
        f"    source={task.source_dataset or '?'}  "
        f"depth={task.depth_score or '?'}  "
        f"parallel={task.parallel_score or '?'}"
    )
    print(f"{'─'*60}")


def _display_verdict_menu() -> None:
    """Print the verdict selection menu."""
    print(_c("\n  Verdict:", BOLD))
    shortcuts = {
        "a": "approve",
        "f": "fix_answer",
        "q": "fix_question",
        "r": "flag_remove",
        "s": "skip",
    }
    colours = {"a": GREEN, "f": YELLOW, "q": YELLOW, "r": RED, "s": DIM}
    for shortcut, verdict in shortcuts.items():
        label = VERDICT_LABELS[verdict]
        print(f"    {_c(shortcut, colours[shortcut])}  {label}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Interactive review loop
# ─────────────────────────────────────────────────────────────────────────────


_SHORTCUT_TO_VERDICT = {
    "a": "approve",
    "f": "fix_answer",
    "q": "fix_question",
    "r": "flag_remove",
    "s": "skip",
}


def _get_verdict(task: Task) -> tuple[str, str, str | None]:
    """Prompt the reviewer for a verdict.

    Returns:
        (verdict, notes, suggested_answer)
    """
    while True:
        try:
            raw = input("  > ").strip().lower()
        except EOFError:
            return "skip", "", None

        if raw in _SHORTCUT_TO_VERDICT:
            verdict = _SHORTCUT_TO_VERDICT[raw]
            notes = ""
            suggested_answer = None

            # Prompt for additional info when needed
            if verdict in ("fix_answer", "fix_question", "flag_remove"):
                try:
                    notes = input(_c("  Notes (optional): ", DIM)).strip()
                except EOFError:
                    notes = ""

            if verdict == "fix_answer":
                try:
                    suggested_answer = input(_c("  Suggested answer: ", YELLOW)).strip() or None
                except EOFError:
                    suggested_answer = None

            return verdict, notes, suggested_answer

        print(_c("  Invalid input. Enter a/f/q/r/s.", RED))


def run_review_session(
    tasks: list[Task],
    task_splits: dict[str, str],
    log: ReviewLog,
    reviewer: str,
    already_reviewed: set[str],
    dry_run: bool = False,
) -> list[ReviewEntry]:
    """Main interactive review loop.

    Args:
        tasks:            Tasks to review.
        task_splits:      Map of task_id -> split name for traceability.
        log:              ReviewLog instance to append entries to.
        reviewer:         Reviewer name/handle.
        already_reviewed: task_ids to skip (resume mode).
        dry_run:          If True, display tasks but don't write to log.

    Returns:
        List of ReviewEntry objects collected this session.
    """
    pending = [t for t in tasks if t.task_id not in already_reviewed]
    total = len(pending)

    if total == 0:
        print(_c("\n  All tasks in this sample have already been reviewed!", GREEN))
        return []

    print(f"\n{_c('  Starting review session', BOLD)}")
    print(f"  {total} tasks to review  |  reviewer: {_c(reviewer, CYAN)}")
    if dry_run:
        print(_c("  [DRY-RUN MODE — nothing will be written to disk]", YELLOW))
    print("  Press Ctrl+C at any time to stop and save progress.\n")

    session_entries: list[ReviewEntry] = []

    try:
        for idx, task in enumerate(pending, start=1):
            split = task_splits.get(task.task_id)
            _display_task(task, idx, total, split)
            _display_verdict_menu()

            verdict, notes, suggested_answer = _get_verdict(task)

            entry = ReviewEntry(
                task_id=task.task_id,
                verdict=verdict,  # type: ignore[arg-type]
                reviewer=reviewer,
                notes=notes,
                suggested_answer=suggested_answer,
                split=split,
            )

            session_entries.append(entry)
            if not dry_run:
                log.append(entry)

            # Inline confirmation
            verdict_colour = {
                "approve": GREEN,
                "fix_answer": YELLOW,
                "fix_question": YELLOW,
                "flag_remove": RED,
                "skip": DIM,
            }.get(verdict, BOLD)
            print(_c(f"  Logged: {verdict}", verdict_colour))

    except KeyboardInterrupt:
        print(_c("\n\n  Session interrupted. Progress saved.", YELLOW))

    return session_entries


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="review_sample",
        description="Interactive manual review CLI for the GateOrchestra dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/review_sample.py
  python scripts/review_sample.py --reviewer umar --resume
  python scripts/review_sample.py --sample-rate 0.10 --split train
  python scripts/review_sample.py --dry-run
  python scripts/review_sample.py --export-flagged
""",
    )
    parser.add_argument(
        "--reviewer",
        default="anonymous",
        help="Your name or handle (recorded in review log). Default: anonymous",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=0.20,
        metavar="RATE",
        help="Fraction of tasks to include in the review sample (default 0.20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default 42).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default=None,
        help="Restrict review to a single split. Default: all splits.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tasks already present in the review log (resume a previous session).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Display tasks and collect verdicts but do NOT write to the log file.",
    )
    parser.add_argument(
        "--export-flagged",
        action="store_true",
        help="After review, export flagged tasks (fix/remove) to a CSV file.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Custom path for the review log JSONL file.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ── Setup ────────────────────────────────────────────────────────────────
    log = ReviewLog(log_path=args.log_path)
    repo = JSONLTaskRepository()

    available_splits = repo.split_names()
    if not available_splits:
        print(_c("ERROR: No dataset splits found. Run create_splits.py first.", RED))
        sys.exit(1)

    # ── Load tasks to review ─────────────────────────────────────────────────
    target_splits = [args.split] if args.split else available_splits
    all_tasks: list[Task] = []
    task_splits: dict[str, str] = {}

    for split in target_splits:
        tasks = repo.list_by_split(split)
        all_tasks.extend(tasks)
        for t in tasks:
            task_splits[t.task_id] = split

    # ── Sample ───────────────────────────────────────────────────────────────
    sampler = ReviewSampler(sample_rate=args.sample_rate, seed=args.seed)
    sample = sampler.sample(all_tasks, split_label=args.split or "all")

    print(f"\n{'='*60}")
    print(_c("  GateOrchestra — Dataset Manual Review", BOLD))
    print(f"{'='*60}")
    sampler.print_summary(sample)

    # ── Resume: skip already-reviewed tasks ──────────────────────────────────
    already_reviewed: set[str] = set()
    if args.resume:
        already_reviewed = log.reviewed_ids()
        pending_count = len([t for t in sample if t.task_id not in already_reviewed])
        print(
            f"  Resume mode: {len(already_reviewed)} already reviewed, "
            f"{pending_count} remaining.\n"
        )

    # ── Run the interactive loop ──────────────────────────────────────────────
    session_entries = run_review_session(
        tasks=sample,
        task_splits=task_splits,
        log=log,
        reviewer=args.reviewer,
        already_reviewed=already_reviewed,
        dry_run=args.dry_run,
    )

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(_c("  Session Complete", BOLD))
    print(f"  Reviewed this session: {len(session_entries)} tasks")
    print(f"{'='*60}")

    if not args.dry_run:
        log.print_summary()

        # ── Export flagged tasks ──────────────────────────────────────────────
        if args.export_flagged:
            flagged_path = ROOT / "reports" / "dataset_quality" / "flagged_tasks.csv"
            log.export_flagged_csv(flagged_path)
            print(f"  Flagged tasks exported -> {flagged_path}\n")
    else:
        print(_c("  [DRY-RUN] No data was written to disk.", YELLOW))


if __name__ == "__main__":
    main()

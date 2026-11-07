"""
scripts/apply_dataset_corrections.py
====================================
Applies the audited MASBench-mini corrections to every dataset artifact.

Some generated tasks had a factually wrong ground truth (e.g. the U.S.
Capitol dome is not gold-covered), an answer that went stale (Greece's head
of state), or a question with no single correct answer ("which element is
named after Switzerland?").  Each such task is rewritten here so it has one
verifiable answer.  Task ids, splits and axis labels are unchanged.

The script is idempotent and patches:
  - dataset/generation/task_pool.py         (source of truth for regeneration)
  - dataset/raw/tasks_raw.jsonl
  - dataset/processed/tasks_cleaned.jsonl
  - dataset/masbench_mini/{split}/{split}.jsonl
  - exports/dataset/{split}.csv

Acceptable answer variants live separately in
dataset/masbench_mini/answer_aliases.json.

Usage:
    python scripts/apply_dataset_corrections.py
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CORRECTIONS: dict[str, dict[str, str]] = {
    # Gold does not cover the U.S. Capitol dome (it is painted cast iron).
    "comp_012": {
        "question": "The element with atomic number 79 was mined in a famous 1858 gold rush "
        "and later used to gild the dome of that state's capitol. In which city is that "
        "capitol building?",
        "context": "The element with atomic number 79 is gold. The Pike's Peak Gold Rush began "
        "in Colorado in 1858. The dome of the Colorado State Capitol in Denver is covered in "
        "gold leaf to commemorate it.",
        "ground_truth": "Denver",
    },
    # Trump (78 at his 2025 inauguration) is now the oldest; ask about first inaugurations.
    "cmp_027": {
        "question": "Who was older at their first inauguration as U.S. president: "
        "Joe Biden or Donald Trump?",
        "ground_truth": "Joe Biden",
    },
    # Greece's head of state changed in 2025; pin the question to a year.
    "bridge_030": {
        "question": "In 2022, who was the head of state of the country where the Acropolis "
        "is located?",
        "context": "The Acropolis is located in Athens, Greece. From 2020 to 2025 the "
        "President of Greece was Katerina Sakellaropoulou.",
    },
    # Röntgen discovered no element; the old ground truth was also wrong for helium.
    "bridge_044": {
        "question": "What is the boiling point, in degrees Celsius, of the element that "
        "William Ramsay first isolated on Earth in 1895 after it had been seen in the "
        "Sun's spectrum?",
        "context": "Helium was first detected in the solar spectrum in 1868 and was isolated "
        "on Earth by William Ramsay in 1895. Helium boils at about -268.9 degrees Celsius "
        "(4.2 K).",
        "ground_truth": "-268.9 degrees Celsius",
    },
    # Bacteria are not an element.
    "bridge_031": {
        "question": "What kind of organisms did the scientist who discovered penicillin "
        "primarily study?",
    },
    # No element is named after Switzerland; CERN also sits in France.
    "comp_027": {
        "question": "What is the atomic number of the element named after the country that "
        "shares the CERN site with Switzerland?",
        "context": "CERN straddles the border between Switzerland and France. Francium is "
        "named after France and has atomic number 87.",
        "ground_truth": "87",
    },
    # The United States has no prime minister.
    "comp_032": {
        "question": "Who was the head of government of the country that developed the first "
        "nuclear bomb at the time it was first used in war?",
        "context": "The United States developed the first nuclear bomb and used it in August "
        "1945. Harry S. Truman was President, and head of government, at the time.",
        "ground_truth": "Harry S. Truman",
    },
    "comp_013": {
        "question": "The founder of the country that launched the first human to space served "
        "as what kind of political leader?",
        "ground_truth": "Premier",
    },
    # Sun Microsystems created no other well-known language; it did create Solaris.
    "comp_023": {
        "question": "What operating system was developed by the company that created the "
        "Java programming language?",
        "ground_truth": "Solaris",
    },
    # Market capitalisation changes daily and has no stable answer.
    "cmp_011": {
        "question": "Which company was founded first: Apple or Microsoft?",
        "ground_truth": "Microsoft (founded 1975) was founded before Apple (founded 1976)",
    },
    # Becquerel discovered radioactivity; Curie coined the term. Trick premise removed.
    "comp_001": {
        "question": "In which country was the scientist who coined the term 'radioactivity' "
        "born?",
        "context": "Marie Curie coined the term 'radioactivity'. She was born in Warsaw, "
        "Poland.",
        "ground_truth": "Poland",
    },
    "comp_002": {
        "question": "In which city was the co-author of 'The Communist Manifesto' who later "
        "wrote 'Das Kapital' born?",
        "context": "Karl Marx co-wrote 'The Communist Manifesto' with Friedrich Engels and "
        "later wrote 'Das Kapital'. Marx was born in Trier, Germany.",
        "ground_truth": "Trier",
    },
    # --- Second audit (November 2026) ---
    # The answer key contradicted the task's own context.
    "bridge_021": {"ground_truth": "Hinduism"},
    "comp_008": {"ground_truth": "George W. Bush"},
    # Skiing was not invented in Switzerland.
    "bridge_015": {
        "question": "What is the currency of the country where the Red Cross was founded?",
        "context": "The Red Cross was founded in Geneva, Switzerland, in 1863. The currency of "
        "Switzerland is the Swiss franc.",
        "ground_truth": "Swiss franc",
    },
    # A GDP ranking changes every year.
    "bridge_017": {
        "question": "In what year did the country that launched the first artificial satellite "
        "send the first human into space?",
        "context": "The first artificial satellite, Sputnik 1, was launched by the Soviet Union "
        "in 1957. The Soviet Union sent Yuri Gagarin, the first human in space, into orbit in "
        "1961.",
        "ground_truth": "1961",
    },
    # Fourteen countries border China; ask for the longest border.
    "bridge_028": {
        "question": "Which country shares the longest land border with the country where the "
        "Great Wall is located?",
        "context": "The Great Wall is located in China. China's longest land border, about "
        "4,630 km, is with Mongolia.",
        "ground_truth": "Mongolia",
    },
    # Africa borders both the Atlantic and the Indian Ocean.
    "bridge_046": {
        "question": "Which ocean borders the western coast of the continent where the Sahara "
        "Desert is primarily located?",
    },
    # Pin office-holders to a year so the answer cannot go stale.
    "bridge_006": {
        "question": "In 2024, who was the president of the country that hosted the 2016 Summer "
        "Olympics?",
        "context": "The 2016 Summer Olympics were hosted in Rio de Janeiro, Brazil. In 2024 the "
        "president of Brazil was Luiz Inácio Lula da Silva.",
    },
    "bridge_043": {
        "question": "In 2024, who was the CEO of the company that owns Instagram?",
        "context": "Instagram is owned by Meta Platforms. In 2024 the CEO of Meta Platforms was "
        "Mark Zuckerberg.",
    },
    # Watson (American) and Crick (British) described DNA together.
    "bridge_003": {
        "question": "What was the nationality of the scientist who described the double-helix "
        "structure of DNA together with the American James Watson?",
    },
    # C was created in the US but is standardised by ISO; ask about its creation.
    "comp_014": {
        "question": "In which country was the programming language used to write most of the "
        "Linux kernel created?",
    },
    # Asia borders several oceans; the Indian Ocean is the one named after India.
    "comp_016": {
        "question": "Which ocean is named after the country where chess was invented?",
        "context": "Chess developed from chaturanga, which originated in India. The Indian Ocean "
        "is named after India.",
    },
    # Spain is itself in continental Europe.
    "comp_019": {
        "question": "Which strait separates the country that produces the most olive oil from "
        "Africa?",
        "context": "Spain is the world's largest producer of olive oil. The Strait of Gibraltar "
        "separates Spain from Morocco in Africa.",
    },
    # The river's length depends on which part is measured; ask which river.
    "comp_020": {
        "question": "What is the longest river flowing through the country that is the world's "
        "largest producer of coffee?",
        "context": "Brazil is the world's largest producer of coffee. The longest river flowing "
        "through Brazil is the Amazon River.",
        "ground_truth": "Amazon River",
    },
    # China, not India, consumes the most rice; a distance is not reliably gradable.
    "comp_029": {
        "question": "What is the capital of the country that produces the most rice?",
        "context": "China is the world's largest producer of rice. The capital of China is "
        "Beijing.",
        "ground_truth": "Beijing",
    },
    # Ottawa's population differs by city versus metro area.
    "comp_033": {
        "question": "What is the capital of the country that has the world's longest coastline?",
        "context": "Canada has the world's longest coastline. The capital of Canada is Ottawa.",
        "ground_truth": "Ottawa",
    },
    # Many composers were born in the 1770s; ask for the decade itself.
    "comp_034": {
        "question": "In which decade did the country where jazz music originated declare its "
        "independence?",
        "context": "Jazz music originated in the United States. The United States declared "
        "independence in 1776, in the 1770s.",
        "ground_truth": "1770s",
    },
    # Silver, not copper, is the best conductor; name copper directly.
    "comp_024": {
        "question": "What is the capital of the country that imports the most copper, the "
        "metal most widely used for electrical wiring?",
        "context": "Copper is the metal most widely used for electrical wiring. China is the "
        "world's largest importer of copper. The capital of China is Beijing.",
    },
    # Several French Impressionists fit; ask about one painting.
    "comp_015": {
        "question": "The philosopher who said 'I think, therefore I am' was born in the same "
        "country as the painter of 'Impression, Sunrise'. Who painted it?",
        "context": "René Descartes, who said 'I think, therefore I am', was born in France. "
        "'Impression, Sunrise' was painted by the French artist Claude Monet.",
    },
    # There were no Olympic Games in 2007, when the iPhone launched.
    "comp_026": {
        "question": "Which country hosted the Summer Olympics in the year after the first "
        "iPhone was released?",
        "context": "The first iPhone was released in 2007. The 2008 Summer Olympics were hosted "
        "by China in Beijing.",
    },
    # Cambodia's coast is on the Gulf of Thailand; Vietnam borders the South China Sea.
    "comp_031": {
        "question": "What is the name of the sea that borders both the country that invented "
        "paper money and the country that is home to Ha Long Bay?",
        "context": "Paper money was invented in China. Ha Long Bay is in Vietnam. Both China "
        "and Vietnam border the South China Sea.",
    },
    # The zloty's regular notes do not show Marie Curie.
    "comp_036": {
        "question": "What is the largest lake in the country where the discoverer of polonium "
        "was born?",
        "context": "Polonium was discovered by Marie Curie, who was born in Poland and named the "
        "element after it. Poland's largest lake is Lake Śniardwy.",
    },
    # Counting territories, the US also spans 11 time zones.
    "cmp_028": {
        "question": "Which country spans more time zones across its contiguous mainland: Russia "
        "or the United States?",
        "ground_truth": "Russia: 11 mainland time zones versus 4 in the contiguous United States",
    },
    # The Byzantine Empire was the Eastern Roman Empire.
    "cmp_030": {
        "question": "Which lasted longer: the Western Roman Empire or the Byzantine Empire?",
    },
    # China's land area is larger; the US total area (with water) is larger.
    "cmp_001": {
        "question": "Which has the larger total area, counting land and water: China or the "
        "United States?",
    },
    # Everest is not in India, and Everest did not survey it himself.
    "comp_038": {
        "question": "The world's highest mountain is named after a former Surveyor General of "
        "India. On the border of which two countries does it lie?",
        "context": "Mount Everest is named after Sir George Everest, a former Surveyor General "
        "of India. The mountain lies on the border between Nepal and China (Tibet "
        "Autonomous Region).",
        "ground_truth": "Nepal and China",
    },
}

SPLITS = ("train", "val", "test")
CSV_FIELDS = (
    "task_id",
    "question",
    "context",
    "ground_truth",
    "depth_score",
    "parallel_score",
    "source_dataset",
)


def _patch_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    changed = 0
    lines_out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        fix = CORRECTIONS.get(row.get("task_id", ""))
        if fix:
            for field, value in fix.items():
                if field in row and row[field] != value:
                    row[field] = value
                    changed += 1
        lines_out.append(json.dumps(row, ensure_ascii=False))
    path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return changed


def _patch_csv(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or list(CSV_FIELDS)
        rows = list(reader)
    changed = 0
    for row in rows:
        fix = CORRECTIONS.get(row.get("task_id", ""))
        if fix:
            for field, value in fix.items():
                if field in row and row[field] != value:
                    row[field] = value
                    changed += 1
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8")
    return changed


def _task_pool_patches(original_rows: dict[str, dict]) -> int:
    """Rewrite the matching entries of task_pool.py (located by original question)."""
    pool_path = ROOT / "dataset" / "generation" / "task_pool.py"
    lines = pool_path.read_text(encoding="utf-8").splitlines()
    pool_keys = {"question": "question", "context": "context", "ground_truth": "answer"}
    changed = 0
    for task_id, fix in CORRECTIONS.items():
        original = original_rows.get(task_id)
        if original is None:
            continue
        needle = json.dumps(original["question"], ensure_ascii=False)
        start = next(
            (i for i, line in enumerate(lines) if line.strip() == f'"question": {needle},'),
            None,
        )
        if start is None:
            continue
        entry_end = next(i for i in range(start, len(lines)) if lines[i].strip() == "},")
        for field, value in fix.items():
            key = pool_keys[field]
            for i in range(start, entry_end):
                if lines[i].strip().startswith(f'"{key}":'):
                    indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
                    new_line = f'{indent}"{key}": {json.dumps(value, ensure_ascii=False)},'
                    if lines[i] != new_line:
                        lines[i] = new_line
                        changed += 1
    pool_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return changed


def main() -> int:
    split_dir = ROOT / "dataset" / "masbench_mini"
    original_rows: dict[str, dict] = {}
    for split in SPLITS:
        path = split_dir / split / f"{split}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                original_rows[row["task_id"]] = row

    missing = sorted(set(CORRECTIONS) - set(original_rows))
    if missing:
        print(f"[FAIL] Unknown task ids in CORRECTIONS: {missing}")
        return 1

    report = {"dataset/generation/task_pool.py": _task_pool_patches(original_rows)}
    targets = [
        ROOT / "dataset" / "raw" / "tasks_raw.jsonl",
        ROOT / "dataset" / "processed" / "tasks_cleaned.jsonl",
        *(split_dir / split / f"{split}.jsonl" for split in SPLITS),
    ]
    for path in targets:
        report[str(path.relative_to(ROOT))] = _patch_jsonl(path)
    for split in SPLITS:
        path = ROOT / "exports" / "dataset" / f"{split}.csv"
        report[str(path.relative_to(ROOT))] = _patch_csv(path)

    for name, count in report.items():
        print(f"  {name:<45} {count} field(s) updated")
    print(f"[OK] {len(CORRECTIONS)} task corrections applied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

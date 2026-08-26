"""
dataset/generation/build_dataset.py
=====================================
Day 1 — Builds the MASBench-mini dataset: 150 synthetic tasks across 3 types.

Task types and their characteristics:
  factoid    — single-hop, low depth (1-2), low parallel (1)
  multi_hop  — chained reasoning, high depth (3-4), low parallel (1-2)
  parallel   — multiple concurrent sub-questions, medium depth (2-3), high parallel (3-4)

Run:
    python dataset/generation/build_dataset.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.config import DATASET_DIR, RANDOM_SEED, TRAIN_SPLIT, VAL_SPLIT
from shared.schemas import Task

random.seed(RANDOM_SEED)

# -----------------------------------------------------------------------------
# Question banks
# -----------------------------------------------------------------------------

FACTOID_QUESTIONS = [
    ("What is the capital of France?", "Paris"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("Who wrote Romeo and Juliet?", "William Shakespeare"),
    ("What is the chemical symbol for gold?", "Au"),
    ("How many sides does a hexagon have?", "6"),
    ("What planet is closest to the Sun?", "Mercury"),
    ("What is the largest ocean on Earth?", "Pacific Ocean"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("What year did World War II end?", "1945"),
    ("What is the speed of light in km/s?", "299792"),
    ("What is the smallest prime number?", "2"),
    ("What element has atomic number 1?", "Hydrogen"),
    ("What is the currency of Japan?", "Yen"),
    ("Who invented the telephone?", "Alexander Graham Bell"),
    ("What is the square root of 144?", "12"),
    ("What is the longest river in the world?", "Nile"),
    ("What gas do plants absorb during photosynthesis?", "Carbon dioxide"),
    ("How many bones are in the adult human body?", "206"),
    ("What is the hardest natural substance?", "Diamond"),
    ("What is the capital of Australia?", "Canberra"),
    ("Who discovered penicillin?", "Alexander Fleming"),
    ("What is the freezing point of water in Fahrenheit?", "32"),
    ("What is the national language of Brazil?", "Portuguese"),
    ("What is the largest continent by area?", "Asia"),
    ("What is 15 percent of 200?", "30"),
    ("Who wrote 1984?", "George Orwell"),
    ("What is the chemical formula for water?", "H2O"),
    ("How many days are in a leap year?", "366"),
    ("What is the capital of Canada?", "Ottawa"),
    ("What organ pumps blood through the human body?", "Heart"),
    ("What is the most spoken language in the world?", "Mandarin Chinese"),
    ("What force keeps planets in orbit around the Sun?", "Gravity"),
    ("Who was the first person to walk on the Moon?", "Neil Armstrong"),
    ("What is the powerhouse of the cell?", "Mitochondria"),
    ("How many continents are there?", "7"),
    ("What is 7 multiplied by 8?", "56"),
    ("What is the capital of Germany?", "Berlin"),
    ("What does DNA stand for?", "Deoxyribonucleic acid"),
    ("How many letters are in the English alphabet?", "26"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("What year did the Berlin Wall fall?", "1989"),
    ("What is the chemical symbol for iron?", "Fe"),
    ("What country has the longest coastline?", "Canada"),
    ("Who wrote Pride and Prejudice?", "Jane Austen"),
    ("What is the sum of angles in a triangle?", "180 degrees"),
    ("What is the capital of China?", "Beijing"),
    ("What is the national bird of the United States?", "Bald eagle"),
    ("What is the unit of electric resistance?", "Ohm"),
    ("Who developed the theory of relativity?", "Albert Einstein"),
    ("What is the largest desert in the world?", "Sahara"),
]

MULTIHOP_QUESTIONS = [
    (
        "The author of the Harry Potter series studied at which university?",
        "University of Exeter",
        "J.K. Rowling wrote Harry Potter. J.K. Rowling studied at the University of Exeter.",
    ),
    (
        "What is the capital of the country where the Eiffel Tower is located?",
        "Paris",
        "The Eiffel Tower is in France. The capital of France is Paris.",
    ),
    (
        "Which element is named after the country that discovered radium?",
        "Polonium",
        "Marie Curie discovered radium. She was from Poland. The element Polonium is named after Poland.",
    ),
    (
        "What language is spoken in the birthplace of the inventor of the World Wide Web?",
        "English",
        "Tim Berners-Lee invented the World Wide Web. He was born in London, England. English is spoken there.",
    ),
    (
        "What is the currency of the country where the Amazon River originates?",
        "Peruvian Sol",
        "The Amazon River originates in Peru. The currency of Peru is the Peruvian Sol.",
    ),
    (
        "Who was the leader of the country that first landed on the moon?",
        "Richard Nixon",
        "The US first landed on the moon in 1969. Richard Nixon was US President in 1969.",
    ),
    (
        "What is the official language of the largest country by area?",
        "Russian",
        "Russia is the largest country by area. The official language is Russian.",
    ),
    (
        "In what ocean is the island where Napoleon was exiled?",
        "Atlantic Ocean",
        "Napoleon was exiled to Saint Helena. Saint Helena is in the South Atlantic Ocean.",
    ),
    (
        "What is the tallest mountain in the country that won the most gold medals in the 2020 Olympics?",
        "Denali",
        "The USA won the most gold medals in 2020. The tallest mountain in the USA is Denali.",
    ),
    (
        "What is the capital of the country that borders both China and Russia?",
        "Ulaanbaatar",
        "Mongolia borders both China and Russia. The capital of Mongolia is Ulaanbaatar.",
    ),
    (
        "What language did the composer of Beethoven's 9th Symphony speak natively?",
        "German",
        "Beethoven composed the 9th Symphony. Beethoven was German. He spoke German natively.",
    ),
    (
        "What ocean surrounds the country with the most Nobel Prize winners per capita?",
        "Atlantic and Arctic Oceans",
        "Norway or Iceland typically tops Nobel per capita. Both border the Atlantic and Arctic.",
    ),
    (
        "What is the population of the city where the company that makes the iPhone was founded?",
        "Approximately 1 million",
        "Apple makes the iPhone. Apple was founded in Cupertino, California. Cupertino has ~60,000 people.",
    ),
    (
        "What element was discovered by the scientist who also discovered chlorine?",
        "Chlorine",
        "Actually, Carl Wilhelm Scheele discovered both oxygen and chlorine.",
    ),
    (
        "What is the main export of the country that invented chess?",
        "Software and textiles",
        "Chess originated in India. India's main exports include software services and textiles.",
    ),
    (
        "What is the height of the mountain named after the surveyor who calculated Everest's height?",
        "8849 meters",
        "George Everest surveyed Everest. The mountain named after him is Mount Everest at 8849m.",
    ),
    (
        "In what year was the author born who wrote the novel that inspired the movie Blade Runner?",
        "1928",
        "Blade Runner is based on Do Androids Dream of Electric Sheep by Philip K. Dick. He was born in 1928.",
    ),
    (
        "What sport is played at the stadium where the 1936 Summer Olympics were held?",
        "Football (Soccer)",
        "The 1936 Olympics were in Berlin at the Olympiastadion. It now hosts football matches.",
    ),
    (
        "What is the boiling point of the element with the lowest melting point?",
        "-246.1 °C",
        "Mercury has the lowest melting point of metals. But the lowest overall is Helium. Helium boils at -269°C. Actually Hydrogen melts at -259°C and boils at -253°C.",
    ),
    (
        "What is the GDP of the country that has the most time zones?",
        "Approximately $2.8 trillion",
        "France has the most time zones (12) due to overseas territories. France's GDP is about $2.8 trillion.",
    ),
    (
        "Which award did the author of the book that inspired Schindler's List receive?",
        "Booker Prize",
        "Schindler's Ark by Thomas Keneally inspired the film. Keneally won the Booker Prize for it.",
    ),
    (
        "What river flows through the capital of the country that hosted the first FIFA World Cup?",
        "Uruguay River",
        "Uruguay hosted the first FIFA World Cup in 1930. The capital is Montevideo. The Uruguay River flows near it.",
    ),
    (
        "What is the name of the mountain range in the country where the tango dance originated?",
        "Andes",
        "Tango originated in Argentina. The Andes mountain range runs through Argentina.",
    ),
    (
        "What is the national anthem of the country where Nikola Tesla was born?",
        "Our Beautiful Homeland",
        "Nikola Tesla was born in Serbia, which was then part of the Austrian Empire, in modern-day Croatia. Croatia's anthem is Our Beautiful Homeland.",
    ),
    (
        "What is the currency of the country where the company that makes Android was founded?",
        "US Dollar",
        "Google developed Android. Google was founded in California, USA. The currency is the US Dollar.",
    ),
]

PARALLEL_QUESTIONS = [
    (
        "What are the capitals of France, Germany, and Italy?",
        "Paris, Berlin, Rome",
        None,
    ),
    (
        "Compare the populations of China, India, and the United States.",
        "China ~1.4B, India ~1.4B, USA ~330M",
        None,
    ),
    (
        "What are the boiling points of water, ethanol, and nitrogen?",
        "100°C, 78.4°C, -196°C",
        None,
    ),
    (
        "Who won the Nobel Prize in Physics in 2020, 2021, and 2022?",
        "Penrose/Ghez/Genzel; Syukuro Manabe et al.; Aspect/Clauser/Zeilinger",
        None,
    ),
    (
        "What are the official languages of Canada, Belgium, and Switzerland?",
        "English and French; Dutch, French, German; German, French, Italian, Romansh",
        None,
    ),
    (
        "What are the highest mountains in Asia, Europe, and Africa?",
        "Everest, Elbrus, Kilimanjaro",
        None,
    ),
    (
        "Compare the GDP of the USA, China, and Japan.",
        "USA ~$25T, China ~$18T, Japan ~$4T",
        None,
    ),
    (
        "What years did World War I, World War II, and the Korean War end?",
        "1918, 1945, 1953",
        None,
    ),
    (
        "What are the atomic numbers of hydrogen, carbon, and oxygen?",
        "1, 6, 8",
        None,
    ),
    (
        "Name the founders of Microsoft, Apple, and Amazon.",
        "Bill Gates/Paul Allen; Steve Jobs/Steve Wozniak/Ronald Wayne; Jeff Bezos",
        None,
    ),
    (
        "What are the speeds of light in vacuum, water, and glass?",
        "300,000 km/s; 225,000 km/s; 200,000 km/s",
        None,
    ),
    (
        "Which countries border Germany, France, and Spain?",
        "Germany: 9 countries; France: 8 countries; Spain: 5 countries",
        None,
    ),
    (
        "What are the national sports of Canada, Australia, and India?",
        "Ice hockey/Lacrosse; Cricket; Field hockey",
        None,
    ),
    (
        "How far is Earth from the Moon, Mars, and the Sun (average)?",
        "384,400 km; 225 million km; 149.6 million km",
        None,
    ),
    (
        "What are the main ingredients of sushi, pizza, and tacos?",
        "Rice/fish/nori; dough/sauce/cheese; tortilla/meat/salsa",
        None,
    ),
    (
        "Who are the current leaders of the USA, UK, and France?",
        "Varies by year — requires up-to-date knowledge",
        None,
    ),
    (
        "Compare the area of Russia, Canada, and China.",
        "Russia 17.1M km², Canada 10M km², China 9.6M km²",
        None,
    ),
    (
        "What are the currencies of the UK, Switzerland, and Japan?",
        "Pound Sterling, Swiss Franc, Yen",
        None,
    ),
    (
        "What are the melting points of iron, aluminum, and copper?",
        "1538°C, 660°C, 1085°C",
        None,
    ),
    (
        "Name the largest cities in Brazil, Argentina, and Chile.",
        "São Paulo, Buenos Aires, Santiago",
        None,
    ),
    (
        "What are the dominant religions in India, Saudi Arabia, and Thailand?",
        "Hinduism, Islam, Buddhism",
        None,
    ),
    (
        "What are the Olympic records for 100m sprint, long jump, and marathon?",
        "Varies by year",
        None,
    ),
    (
        "Compare the literacy rates of Finland, South Korea, and Nigeria.",
        "Finland ~100%, South Korea ~98%, Nigeria ~62%",
        None,
    ),
    (
        "What are the national animals of China, England, and Canada?",
        "Giant Panda/Dragon, Lion, Beaver",
        None,
    ),
    (
        "Compare the ages of the Eiffel Tower, the Colosseum, and the Great Wall of China.",
        "~135 years, ~1950 years, ~2300 years",
        None,
    ),
]


# -----------------------------------------------------------------------------
# Task builder
# -----------------------------------------------------------------------------


def build_tasks() -> list[Task]:
    tasks: list[Task] = []
    idx = 0

    # 50 factoid tasks
    sampled_factoid = random.choices(FACTOID_QUESTIONS, k=50)
    for q, ans in sampled_factoid:
        tasks.append(
            Task(
                task_id=f"factoid_{idx:03d}",
                question=q,
                ground_truth=ans,
                depth_score=random.choice([1, 2]),
                parallel_score=1,
                source_dataset="synthetic_factoid",
            )
        )
        idx += 1

    # 50 multi-hop tasks
    sampled_mh = random.choices(MULTIHOP_QUESTIONS, k=50)
    for q, ans, ctx in sampled_mh:
        tasks.append(
            Task(
                task_id=f"multihop_{idx:03d}",
                question=q,
                context=ctx,
                ground_truth=ans,
                depth_score=random.choice([3, 4]),
                parallel_score=random.choice([1, 2]),
                source_dataset="synthetic_multihop",
            )
        )
        idx += 1

    # 50 parallel tasks
    sampled_par = random.choices(PARALLEL_QUESTIONS, k=50)
    for q, ans, ctx in sampled_par:
        tasks.append(
            Task(
                task_id=f"parallel_{idx:03d}",
                question=q,
                context=ctx,
                ground_truth=ans,
                depth_score=random.choice([2, 3]),
                parallel_score=random.choice([3, 4]),
                source_dataset="synthetic_parallel",
            )
        )
        idx += 1

    random.shuffle(tasks)
    return tasks


def split_and_save(tasks: list[Task]) -> dict[str, int]:
    n = len(tasks)
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)

    splits = {
        "train": tasks[:n_train],
        "val": tasks[n_train : n_train + n_val],
        "test": tasks[n_train + n_val :],
    }

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split_name, split_tasks in splits.items():
        out_path = DATASET_DIR / split_name / f"{split_name}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for t in split_tasks:
                f.write(t.model_dump_json() + "\n")
        counts[split_name] = len(split_tasks)
        print(f"  [+] Wrote {len(split_tasks):3d} tasks -> {out_path}")

    return counts


def print_sample(tasks: list[Task], n: int = 3) -> None:
    print(f"\n-- Sample Tasks ({'-'*40})")
    for t in random.sample(tasks, min(n, len(tasks))):
        depth = t.depth_score or "?"
        par = t.parallel_score or "?"
        q_short = t.question[:65] + ("..." if len(t.question) > 65 else "")
        print(f"  [{t.source_dataset:>22}] depth={depth} par={par}  {q_short!r}")


def print_stats(tasks: list[Task]) -> None:
    by_type = {}
    for t in tasks:
        src = t.source_dataset or "unknown"
        by_type[src] = by_type.get(src, 0) + 1

    print(f"\n-- Task Type Distribution ({'-'*30})")
    for src, cnt in sorted(by_type.items()):
        bar = "#" * (cnt // 2)
        print(f"  {src:>25}: {cnt:3d}  {bar}")

    depths = [t.depth_score for t in tasks if t.depth_score]
    pars = [t.parallel_score for t in tasks if t.parallel_score]
    print(f"\n  Depth  range: {min(depths)}–{max(depths)}, avg {sum(depths)/len(depths):.1f}")
    print(f"  Parallel range: {min(pars)}–{max(pars)}, avg {sum(pars)/len(pars):.1f}")


if __name__ == "__main__":
    print("=" * 60)
    print("  GateOrchestra — Dataset Builder (Day 1)")
    print("=" * 60)

    print("\n[*] Building 150 tasks across 3 types...")
    tasks = build_tasks()
    print(f"  Generated: {len(tasks)} total tasks")

    print_stats(tasks)
    print_sample(tasks)

    print(f"\n[>] Saving to {DATASET_DIR} ...")
    counts = split_and_save(tasks)

    print(f"\n{'-'*60}")
    print(f"  Train: {counts['train']} | Val: {counts['val']} | Test: {counts['test']}")
    print(f"  Total: {sum(counts.values())} tasks saved to disk")
    print(f"{'-'*60}")
    print("\n[OK] Day 1 complete. Dataset is on disk and ready to load.")

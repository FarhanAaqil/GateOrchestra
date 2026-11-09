"""
dataset/generation/test_expansion.py
====================================
64 extra held-out test tasks (ids continue each family's numbering), written
after the original 161 to give the test split 96 tasks instead of 32.

Design rules, so these tasks do not repeat the original dataset's mistakes:
  * Every factual task carries a context that states the facts it needs, so
    the gold answer can be checked against the text alone.
  * Nothing depends on the current date or an office-holder; dates are fixed
    ("as of 2023"), and contested figures are avoided.
  * Arithmetic gold answers are computed here from the question's numbers.
  * No question reuses an entity chain already in the train/validation splits
    (scripts/expand_test_split.py refuses near-duplicates).

These tasks have NOT been reviewed by a person. dataset/masbench_mini/REVIEW.json
records that the 161 original tasks were reviewed and these were not.
"""

from __future__ import annotations

from typing import Any


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _num(value: float) -> str:
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


def _arithmetic() -> list[dict[str, Any]]:
    """(question, gold, depth) with every gold computed from the numbers in the question."""
    rows: list[tuple[str, str, int]] = []

    price, qty, paid = 3.25, 7, 50
    rows.append(
        (
            f"A bakery sells muffins at ${price:.2f} each. If you buy {qty} muffins and pay with a "
            f"${paid} bill, how much change do you receive?",
            _money(paid - price * qty),
            3,
        )
    )
    length, width = 18, 11
    rows.append(
        (
            f"A rectangle has a length of {length} cm and a width of {width} cm. What is its area?",
            f"{length * width} square centimeters",
            2,
        )
    )
    base, height = 14, 9
    rows.append(
        (
            f"A triangle has a base of {base} cm and a height of {height} cm. What is its area?",
            f"{base * height // 2} square centimeters",
            3,
        )
    )
    cost, off = 120, 35
    rows.append(
        (
            f"A jacket costs ${cost} and is on sale for {off}% off. What is the sale price?",
            _money(cost * (100 - off) / 100),
            3,
        )
    )
    start, hours, period = 500, 12, 3
    rows.append(
        (
            f"A bacteria colony doubles every {period} hours. If it starts with {start} bacteria, "
            f"how many are there after {hours} hours?",
            _num(start * 2 ** (hours // period)),
            3,
        )
    )
    km, hrs = 72, 4
    rows.append(
        (
            f"A cyclist rides {km} km in {hrs} hours. What is the average speed in km/h?",
            f"{km // hrs} km/h",
            2,
        )
    )
    rows.append(("How many seconds are there in 3.5 hours?", f"{_num(3.5 * 3600)} seconds", 2))
    principal, rate, years = 2500, 6, 4
    rows.append(
        (
            f"What is the simple interest on ${principal:,} at {rate}% per year for {years} years?",
            _money(principal * rate / 100 * years),
            3,
        )
    )
    scores = [72, 85, 90, 64, 79]
    rows.append(
        (
            f"The scores of five students are {', '.join(map(str, scores[:-1]))} and {scores[-1]}. "
            "What is the average score?",
            _num(sum(scores) / len(scores)),
            3,
        )
    )
    cookies, flour, want = 12, 180, 30
    rows.append(
        (
            f"A recipe for {cookies} cookies uses {flour} g of flour. How much flour is needed for "
            f"{want} cookies?",
            f"{flour * want // cookies} grams",
            3,
        )
    )
    celsius = 35
    rows.append(
        (
            f"If the temperature is {celsius}°C, what is it in Fahrenheit?",
            f"{celsius * 9 // 5 + 32}°F",
            3,
        )
    )
    pages, h = 24, 2.5
    rows.append(
        (
            f"A printer prints {pages} pages per minute. How many pages does it print in {h} hours?",
            f"{_num(pages * h * 60)} pages",
            3,
        )
    )
    nb, pen, n_nb, n_pen = 4.50, 1.25, 6, 8
    rows.append(
        (
            f"Notebooks cost ${nb:.2f} each and pens cost ${pen:.2f} each. What is the total cost of "
            f"{n_nb} notebooks and {n_pen} pens?",
            _money(nb * n_nb + pen * n_pen),
            4,
        )
    )
    pop, grow = 40000, 15
    rows.append(
        (
            f"A town of {pop:,} people grows by {grow}% in a year. What is the new population?",
            _num(pop * (100 + grow) / 100),
            3,
        )
    )
    v1, t1, v2, t2 = 60, 2.5, 80, 1.5
    rows.append(
        (
            f"A car travels at {v1} km/h for {t1} hours and then at {v2} km/h for {t2} hours. "
            "How far does it travel in total?",
            f"{_num(v1 * t1 + v2 * t2)} km",
            4,
        )
    )
    a, b, c = 12, 8, 5
    rows.append(
        (
            f"A box measures {a} cm by {b} cm by {c} cm. What is its volume?",
            f"{a * b * c} cubic centimeters",
            2,
        )
    )
    value, drop, yrs = 8000, 10, 3
    rows.append(
        (
            f"A machine bought for ${value:,} loses {drop}% of its value each year. What is it worth "
            f"after {yrs} years?",
            _money(value * (1 - drop / 100) ** yrs),
            4,
        )
    )
    t_a, t_b = 6, 3
    rows.append(
        (
            f"Pipe A fills a tank in {t_a} hours and pipe B fills it in {t_b} hours. Working together, "
            "how many hours do they take to fill the tank?",
            f"{_num(1 / (1 / t_a + 1 / t_b))} hours",
            4,
        )
    )
    boys, girls, total = 3, 5, 40
    rows.append(
        (
            f"The ratio of boys to girls in a class is {boys}:{girls}. If there are {total} students, "
            "how many are girls?",
            f"{total * girls // (boys + girls)}",
            3,
        )
    )
    grams, cost_jar, want_g = 750, 6.00, 2000
    rows.append(
        (
            f"A {grams} g jar of jam costs ${cost_jar:.2f}. How much would {want_g / 1000:g} kg cost at "
            "the same price per gram?",
            _money(cost_jar / grams * want_g),
            4,
        )
    )
    return [
        {
            "question": q,
            "context": None,
            "ground_truth": gold,
            "depth_score": d,
            "parallel_score": 1,
            "source_dataset": "template_arithmetic",
        }
        for q, gold, d in rows
    ]


def _bridge() -> list[dict[str, Any]]:
    """Two-hop questions: (question, context, gold)."""
    rows = [
        (
            "In which city was the architect of the Sydney Opera House born?",
            "The Sydney Opera House was designed by Jørn Utzon. Jørn Utzon was born in Copenhagen.",
            "Copenhagen",
        ),
        (
            "What was the native language of the author of 'One Hundred Years of Solitude'?",
            "'One Hundred Years of Solitude' was written by Gabriel García Márquez. García Márquez's native "
            "language was Spanish.",
            "Spanish",
        ),
        (
            "In which year was the composer of 'The Four Seasons' born?",
            "'The Four Seasons' was composed by Antonio Vivaldi. Vivaldi was born in 1678.",
            "1678",
        ),
        (
            "Which sport did the athlete with the most Olympic gold medals in history compete in?",
            "Michael Phelps holds the record for the most Olympic gold medals, with 23. Michael Phelps "
            "competed in swimming.",
            "Swimming",
        ),
        (
            "On which continent is the site of Machu Picchu located?",
            "Machu Picchu is located in Peru. Peru is a country in South America.",
            "South America",
        ),
        (
            "Who painted the portrait of Lisa Gherardini that hangs in the Louvre?",
            "The Mona Lisa, a portrait of Lisa Gherardini, hangs in the Louvre. The Mona Lisa was painted by "
            "Leonardo da Vinci.",
            "Leonardo da Vinci",
        ),
        (
            "What is the official language of the country whose capital is Lisbon?",
            "Lisbon is the capital of Portugal. The official language of Portugal is Portuguese.",
            "Portuguese",
        ),
        (
            "Which river flows through the capital of Egypt?",
            "Cairo is the capital of Egypt. The Nile River flows through Cairo.",
            "Nile",
        ),
        (
            "What is the chemical symbol of the only metal that is liquid at room temperature?",
            "Mercury is the only metal that is liquid at room temperature. The chemical symbol of mercury is "
            "Hg.",
            "Hg",
        ),
        (
            "What currency is used in the country that contains Mount Fuji?",
            "Mount Fuji is located in Japan. The currency of Japan is the yen.",
            "Yen",
        ),
        (
            "Which ocean lies along the eastern coast of the continent where the Amazon River is found?",
            "The Amazon River is in South America. The Atlantic Ocean lies along the eastern coast of South "
            "America.",
            "Atlantic Ocean",
        ),
        (
            "In which year did the author of 'Pride and Prejudice' die?",
            "'Pride and Prejudice' was written by Jane Austen. Jane Austen died in 1817.",
            "1817",
        ),
        (
            "What is the official language of the country where the Acropolis stands?",
            "The Acropolis stands in Athens, Greece. The official language of Greece is Greek.",
            "Greek",
        ),
        (
            "Who was President of the United States when Apollo 11 landed on the Moon?",
            "Apollo 11 landed on the Moon in July 1969. Richard Nixon was President of the United States in "
            "July 1969.",
            "Richard Nixon",
        ),
        (
            "In what year was the scientist who discovered Jupiter's four largest moons born?",
            "Galileo Galilei discovered Jupiter's four largest moons in 1610. Galileo was born in 1564.",
            "1564",
        ),
        (
            "Which sea lies along the southern coast of the country where the city of Odessa is located?",
            "Odessa is a city in Ukraine. Ukraine's southern coast lies on the Black Sea.",
            "Black Sea",
        ),
    ]
    return [
        {
            "question": q,
            "context": c,
            "ground_truth": a,
            "depth_score": 3,
            "parallel_score": 1,
            "source_dataset": "hotpotqa_style",
        }
        for q, c, a in rows
    ]


def _compositional() -> list[dict[str, Any]]:
    """Three-fact chains and small computations over stated facts: (question, context, gold, depth)."""
    rows = [
        (
            "What is the capital of the country where the author of 'Don Quixote' was born?",
            "'Don Quixote' was written by Miguel de Cervantes. Cervantes was born in Alcalá de Henares, Spain. "
            "The capital of Spain is Madrid.",
            "Madrid",
            5,
        ),
        (
            "What is the capital of the country where the inventor of the World Wide Web was born?",
            "The World Wide Web was invented by Tim Berners-Lee. He was born in London, in the United "
            "Kingdom. The capital of the United Kingdom is London.",
            "London",
            5,
        ),
        (
            "What currency is used in the country where the composer of the Ninth Symphony with the 'Ode to "
            "Joy' was born?",
            "Ludwig van Beethoven composed the Ninth Symphony, which ends with the 'Ode to Joy'. Beethoven was "
            "born in Bonn, Germany. Germany's currency is the euro.",
            "Euro",
            5,
        ),
        (
            "What is the official language of the country where the painter of 'The Starry Night' was born?",
            "'The Starry Night' was painted by Vincent van Gogh. Van Gogh was born in the Netherlands. The "
            "official language of the Netherlands is Dutch.",
            "Dutch",
            5,
        ),
        (
            "What currency is used in the country where the inventor of dynamite was born?",
            "Alfred Nobel invented dynamite. Nobel was born in Stockholm, Sweden. The currency of Sweden is "
            "the Swedish krona.",
            "Swedish krona",
            5,
        ),
        (
            "How many years after the first Moon landing did the Berlin Wall fall?",
            "The first Moon landing took place in 1969. The Berlin Wall fell in 1989.",
            "20 years",
            4,
        ),
        (
            "According to the years given, how many years did Wolfgang Amadeus Mozart live?",
            "Mozart was born in 1756 and died in 1791.",
            "35 years",
            4,
        ),
        (
            "How many years after William Shakespeare's death did Isaac Newton publish the Principia?",
            "William Shakespeare died in 1616. Isaac Newton published the Principia in 1687.",
            "71 years",
            4,
        ),
        (
            "In which country is the tallest mountain in Africa located?",
            "The tallest mountain in Africa is Mount Kilimanjaro. Mount Kilimanjaro is located in Tanzania.",
            "Tanzania",
            4,
        ),
        (
            "In which city does the ceremony for the prize awarded by the Norwegian Nobel Committee take place?",
            "The Nobel Peace Prize is awarded by the Norwegian Nobel Committee. The Nobel Peace Prize "
            "ceremony takes place in Oslo.",
            "Oslo",
            4,
        ),
        (
            "Which element has an atomic number equal to the number of planets in the solar system?",
            "The solar system has eight planets. The element with atomic number 8 is oxygen.",
            "Oxygen",
            4,
        ),
        (
            "How many moons does the planet named after the Roman god of war have?",
            "Mars is named after the Roman god of war. Mars has two moons, Phobos and Deimos.",
            "2",
            4,
        ),
        (
            "Which gas that plants absorb during photosynthesis is also released by burning fossil fuels?",
            "Plants absorb carbon dioxide during photosynthesis. Burning fossil fuels releases carbon "
            "dioxide.",
            "Carbon dioxide",
            4,
        ),
        (
            "What is the capital of the largest country in South America by area?",
            "Brazil is the largest country in South America by area. The capital of Brazil is Brasília.",
            "Brasília",
            4,
        ),
        (
            "Which river flows through the city that houses the Louvre?",
            "The Louvre is a museum in Paris. The Seine flows through Paris.",
            "Seine",
            4,
        ),
        (
            "In which year did the president who signed the Emancipation Proclamation die?",
            "Abraham Lincoln signed the Emancipation Proclamation in 1863. Lincoln died in 1865.",
            "1865",
            4,
        ),
    ]
    return [
        {
            "question": q,
            "context": c,
            "ground_truth": a,
            "depth_score": d,
            "parallel_score": 1,
            "source_dataset": "musique_style",
        }
        for q, c, a, d in rows
    ]


def _comparison() -> list[dict[str, Any]]:
    """Two-way comparisons grounded by a context: (question, context, gold)."""
    rows = [
        (
            "Which mountain is taller: Mount Everest or K2?",
            "Mount Everest is 8,849 m tall. K2 is 8,611 m tall.",
            "Mount Everest is taller than K2",
        ),
        (
            "Which planet has the larger diameter: Uranus or Neptune?",
            "Uranus has an equatorial diameter of about 51,100 km. Neptune has an equatorial diameter of "
            "about 49,500 km.",
            "Uranus has a larger diameter than Neptune",
        ),
        (
            "Which river is longer: the Danube or the Rhine?",
            "The Danube is about 2,850 km long. The Rhine is about 1,230 km long.",
            "The Danube is longer than the Rhine",
        ),
        (
            "Which ocean is larger: the Atlantic or the Indian?",
            "The Atlantic Ocean covers about 85 million square kilometers. The Indian Ocean covers about 70 "
            "million square kilometers.",
            "The Atlantic is larger than the Indian Ocean",
        ),
        (
            "Which came first: Gutenberg's printing press or Columbus's first voyage to the Americas?",
            "Johannes Gutenberg introduced his printing press around 1440. Christopher Columbus made his first "
            "voyage to the Americas in 1492.",
            "Gutenberg's printing press came first",
        ),
        (
            "Who was born earlier: Marie Curie or Albert Einstein?",
            "Marie Curie was born in 1867. Albert Einstein was born in 1879.",
            "Marie Curie was born earlier",
        ),
        (
            "As of 2023, which country had the larger population: Canada or Australia?",
            "As of 2023, Canada had about 40 million people. As of 2023, Australia had about 26 million people.",
            "Canada had a larger population than Australia",
        ),
        (
            "Which metal is denser: gold or silver?",
            "Gold has a density of about 19.3 grams per cubic centimeter. Silver has a density of about 10.5 "
            "grams per cubic centimeter.",
            "Gold is denser than silver",
        ),
        (
            "Which planet is hotter on average: Mercury or Venus?",
            "Venus has an average surface temperature of about 464 degrees Celsius. Mercury has an average "
            "surface temperature of about 167 degrees Celsius.",
            "Venus is hotter than Mercury",
        ),
        (
            "Which is older: the Great Pyramid of Giza or the Parthenon?",
            "The Great Pyramid of Giza was built around 2560 BCE. The Parthenon was completed in 432 BCE.",
            "The Great Pyramid of Giza is older than the Parthenon",
        ),
        (
            "Which has the higher boiling point at sea level: ethanol or water?",
            "Ethanol boils at 78 degrees Celsius at sea level. Water boils at 100 degrees Celsius at sea "
            "level.",
            "Water has a higher boiling point than ethanol",
        ),
        (
            "Which flight route is longer: London to New York or London to Tokyo?",
            "The distance from London to New York is about 5,570 km. The distance from London to Tokyo is "
            "about 9,560 km.",
            "London to Tokyo is longer than London to New York",
        ),
    ]
    return [
        {
            "question": q,
            "context": c,
            "ground_truth": a,
            "depth_score": 3,
            "parallel_score": 2,
            "source_dataset": "template_comparison",
        }
        for q, c, a in rows
    ]


# Short accepted forms of each comparison answer (the shortest unambiguous winner).
COMPARISON_ALIASES = [
    "Mount Everest",
    "Everest",
    "Uranus",
    "Danube",
    "The Danube",
    "Atlantic",
    "The Atlantic",
    "Gutenberg",
    "Gutenberg's printing press",
    "printing press",
    "Marie Curie",
    "Curie",
    "Canada",
    "Gold",
    "Venus",
    "Great Pyramid",
    "Great Pyramid of Giza",
    "Water",
    "London to Tokyo",
    "Tokyo",
]


def build_new_tasks() -> list[dict[str, Any]]:
    """All 64 tasks with ids, in a fixed order."""
    families = [
        ("arith", 41, _arithmetic()),
        ("bridge", 52, _bridge()),
        ("comp", 40, _compositional()),
        ("cmp", 32, _comparison()),
    ]
    tasks: list[dict[str, Any]] = []
    for prefix, start, rows in families:
        for offset, row in enumerate(rows):
            tasks.append({"task_id": f"{prefix}_{start + offset:03d}", **row})
    return tasks


# Extra accepted forms, keyed by the gold answer so they cannot drift from the task ids.
EXTRA_ALIASES_BY_GOLD = {
    "Euro": ["euros", "EUR"],
    "Swedish krona": ["Krona", "SEK"],
    "Brasília": ["Brasilia"],
    "2": ["Two"],
    "Carbon dioxide": ["CO2"],
    "Yen": ["Japanese yen"],
    "Nile": ["Nile River"],
    "Atlantic Ocean": ["Atlantic"],
    "Black Sea": ["the Black Sea"],
}


def build_aliases() -> dict[str, list[str]]:
    """Accepted short answers keyed by task id (only where the gold is long or has variants)."""
    aliases: dict[str, list[str]] = {}
    for task in build_new_tasks():
        tid, gold = task["task_id"], task["ground_truth"]
        found: list[str] = []
        if tid.startswith("cmp_"):
            found = [a for a in COMPARISON_ALIASES if a.lower() in gold.lower()]
        elif gold.endswith(" years"):
            found = [gold.removesuffix(" years")]
        found += EXTRA_ALIASES_BY_GOLD.get(gold, [])
        if found:
            aliases[tid] = sorted(set(found), key=lambda a: (len(a), a))
    return aliases

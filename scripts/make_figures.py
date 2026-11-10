"""
scripts/make_figures.py
=======================
Draw the report figures from the real benchmark results:

  reports/figures/final_pareto.png   accuracy vs tokens per method (95% CIs)
  reports/figures/premise.png        CoT-SC vs MAS on the calibration runs
  reports/figures/axis_gain.png      MAS accuracy gain per Depth x Parallel cell
  reports/figures/ablation.png       gate accuracy with each feature removed

Run after scripts/final_benchmark.py and scripts/analyze_final_results.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.config import FINAL_REPORTS_DIR, FINAL_RESULTS_DIR  # noqa: E402

MASTER = FINAL_RESULTS_DIR / "master_results.json"
ANALYSIS = FINAL_REPORTS_DIR / "final_analysis.json"
OUT = FINAL_REPORTS_DIR / "figures"

# The console's palette.
INK, NAVY, SKY, EMBER, WINE, TANGERINE, CYAN, ICE = (
    "#1B1B1F",
    "#1B263B",
    "#A8DBFF",
    "#FF6B35",
    "#8B1538",
    "#FF8C42",
    "#00E5FF",
    "#F1FAFF",
)
METHOD_COLORS = {
    "CoT-SC-only": SKY,
    "Always-MAS": WINE,
    "RandomGate": NAVY,
    "RuleBasedGate": TANGERINE,
    "GateOrchestra": EMBER,
}


def _style(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=NAVY)
    ax.grid(alpha=0.25)


def pareto(master: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for row in master["aggregated"]:
        low, high = row["accuracy_pooled_ci95"]
        acc = row["accuracy_mean"]
        ax.errorbar(
            row["avg_tokens_mean"],
            acc,
            yerr=[[acc - low], [high - acc]],
            fmt="o",
            ms=9,
            color=METHOD_COLORS.get(row["method"], INK),
            capsize=4,
            label=row["method"],
        )
    ax.set_xlabel("Average tokens per task")
    ax.set_ylabel("Accuracy (%) with pooled 95% CI")
    seeds = ", ".join(map(str, master["seeds"]))
    _style(ax, f"Accuracy vs cost on the test split (seeds {seeds})")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def premise(analysis: dict, out: Path) -> None:
    p = analysis["premise"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.6))
    names = ["CoT-SC", "MAS"]
    accs = [100 * p["cot_sc"]["accuracy"], 100 * p["mas"]["accuracy"]]
    errs = [
        [100 * (p[k]["accuracy"] - p[k]["ci95"][0]) for k in ("cot_sc", "mas")],
        [100 * (p[k]["ci95"][1] - p[k]["accuracy"]) for k in ("cot_sc", "mas")],
    ]
    a1.bar(names, accs, yerr=errs, color=[SKY, WINE], capsize=5)
    a1.set_ylabel("Accuracy (%)")
    _style(a1, "Accuracy")
    a2.bar(names, [p["cot_sc"]["avg_tokens"], p["mas"]["avg_tokens"]], color=[SKY, WINE])
    a2.set_ylabel("Average tokens per run")
    _style(a2, "Cost")
    fig.suptitle(f"CoT-SC vs MAS on {p['n_tasks']} train/validation tasks x 3 repeats", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def axis_gain(analysis: dict, out: Path) -> None:
    cells = analysis["axis"]["depth_parallel"]
    depths = sorted({int(c["key"][1]) for c in cells})
    pars = sorted({int(c["key"].split("xP")[1]) for c in cells})
    grid = [[float("nan")] * len(pars) for _ in depths]
    labels = [[""] * len(pars) for _ in depths]
    for c in cells:
        d, q = int(c["key"][1]), int(c["key"].split("xP")[1])
        grid[depths.index(d)][pars.index(q)] = c["mas_gain_pts"]
        labels[depths.index(d)][pars.index(q)] = f"{c['mas_gain_pts']:+.0f}\nn={c['n_tasks']}"
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    limit = max(10.0, max(abs(c["mas_gain_pts"]) for c in cells))
    image = ax.imshow(grid, cmap="RdBu", vmin=-limit, vmax=limit, origin="lower")
    for i in range(len(depths)):
        for j in range(len(pars)):
            dark = abs(grid[i][j]) > 0.55 * limit  # NaN compares False
            ax.text(
                j,
                i,
                labels[i][j],
                ha="center",
                va="center",
                fontsize=8,
                color="white" if dark else INK,
            )
    ax.set_xticks(range(len(pars)), [f"P{q}" for q in pars])
    ax.set_yticks(range(len(depths)), [f"D{d}" for d in depths])
    ax.set_xlabel("Parallel label")
    ax.set_ylabel("Depth label")
    fig.colorbar(image, ax=ax, label="MAS - CoT-SC accuracy (pts)")
    ax.set_title("Where MAS helps", loc="left", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def ablation(analysis: dict, out: Path) -> bool:
    rows = analysis.get("feature_ablation", {}).get("rows", [])
    if not rows:
        return False
    classifiers = sorted({r["classifier"] for r in rows})
    fig, axes = plt.subplots(1, len(classifiers), figsize=(5 * len(classifiers), 4), squeeze=False)
    for ax, name in zip(axes[0], classifiers, strict=True):
        mine = [r for r in rows if r["classifier"] == name]
        base = next(r["accuracy"] for r in mine if r["dropped"] == "none")
        dropped = [r for r in mine if r["dropped"] != "none"]
        deltas = [100 * (r["accuracy"] - base) for r in dropped]
        ax.barh(
            [r["dropped"] for r in dropped],
            deltas,
            color=[WINE if d < 0 else CYAN for d in deltas],
        )
        ax.axvline(0, color=NAVY, lw=1)
        ax.set_xlabel("Validation accuracy change (pts)")
        _style(ax, f"{name}: remove one feature (base {100 * base:.0f}%)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    if MASTER.exists():
        pareto(json.loads(MASTER.read_text(encoding="utf-8")), OUT / "final_pareto.png")
        made.append("final_pareto.png")
    if ANALYSIS.exists():
        analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
        if "premise" in analysis:
            premise(analysis, OUT / "premise.png")
            made.append("premise.png")
        if analysis.get("axis"):
            axis_gain(analysis, OUT / "axis_gain.png")
            made.append("axis_gain.png")
        if ablation(analysis, OUT / "ablation.png"):
            made.append("ablation.png")
    if not made:
        sys.exit("No results yet: run final_benchmark.py and analyze_final_results.py first.")
    print(f"[OK] wrote {', '.join(made)} to {OUT}")


if __name__ == "__main__":
    main()

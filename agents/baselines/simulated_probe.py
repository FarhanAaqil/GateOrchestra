"""
agents/baselines/simulated_probe.py
=====================================
Day 4 -- Calibrated CoT-SC probe simulator.

Replaces the flat mock_probe_agent everywhere in scripts/.
Still no real LLM calls -- that's Week 3.
But the behavior is calibrated to match real CoT-SC patterns:

  Task type      Consistency    Tokens       Notes
  -----------    -----------    ------       -----
  factoid        0.80 - 1.00    60 - 180     High agreement, cheap
  multi-hop      0.45 - 0.80    180 - 380    Harder, more spread
  parallel       0.30 - 0.65    220 - 450    Multi-part, low agreement

Calibration source: MASBench paper Table 3, approximated.

Usage:
    from agents.baselines.simulated_probe import SimulatedProbe

    probe = SimulatedProbe(seed=42)
    result = probe.run(task)          # -> ProbeResult
"""

from __future__ import annotations

import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Allow running as script from any directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.schemas import ProbeResult, Task

# ─────────────────────────────────────────────────────────────────────────────
# Calibration profiles per task source
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _Profile:
    """Calibration profile for a task type."""

    consistency_mean: float
    consistency_std: float
    token_min: int
    token_max: int
    # Probability MAS would beat probe on this type (used in label generation)
    mas_beats_probe_rate: float


_PROFILES: dict[str, _Profile] = {
    "synthetic_factoid": _Profile(
        consistency_mean=0.88,
        consistency_std=0.08,
        token_min=60,
        token_max=180,
        mas_beats_probe_rate=0.05,  # MAS rarely adds value here
    ),
    "synthetic_multihop": _Profile(
        consistency_mean=0.62,
        consistency_std=0.13,
        token_min=180,
        token_max=380,
        mas_beats_probe_rate=0.38,  # MAS helps ~38% of the time
    ),
    "synthetic_parallel": _Profile(
        consistency_mean=0.47,
        consistency_std=0.12,
        token_min=220,
        token_max=450,
        mas_beats_probe_rate=0.55,  # MAS helps ~55% of the time
    ),
}

_DEFAULT_PROFILE = _Profile(
    consistency_mean=0.65,
    consistency_std=0.15,
    token_min=100,
    token_max=350,
    mas_beats_probe_rate=0.30,
)

N_SAMPLES = 5  # Number of CoT-SC samples per probe run


# ─────────────────────────────────────────────────────────────────────────────
# Answer pool helpers
# ─────────────────────────────────────────────────────────────────────────────


def _wrong_variants(correct_answer: str) -> list[str]:
    """Generate plausible wrong answers for the consistency simulation."""
    base = correct_answer.strip()
    variants = [
        base + " (incorrect)",
        "I don't know",
        "Unable to determine",
    ]
    # Numeric twist: if answer looks numeric, perturb it
    nums = re.findall(r"\d+", base)
    for n in nums[:2]:
        variants.append(base.replace(n, str(int(n) + random.randint(1, 10)), 1))
    return variants


def _generate_samples(
    correct_answer: str,
    consistency: float,
    n: int,
    rng: random.Random,
) -> list[str]:
    """
    Generate n CoT-SC sample outputs.
    `consistency` fraction will give the correct answer.
    """
    samples = []
    wrongs = _wrong_variants(correct_answer)
    for _ in range(n):
        if rng.random() < consistency:
            samples.append(correct_answer)
        else:
            samples.append(rng.choice(wrongs))
    rng.shuffle(samples)
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SimulatedProbe:
    """
    Calibrated CoT-SC probe simulator.

    Produces ProbeResult objects that match real CoT-SC statistics
    without making LLM calls. Used in all scripts until Week 3.

    Args:
        seed: Random seed for reproducibility. Default 42.
        n_samples: Number of CoT-SC samples to simulate. Default 5.
    """

    seed: int = 42
    n_samples: int = N_SAMPLES
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def _get_profile(self, task: Task) -> _Profile:
        return _PROFILES.get(task.source_dataset or "", _DEFAULT_PROFILE)

    def run(self, task: Task) -> ProbeResult:
        """Simulate a CoT-SC probe run for a task.

        Returns:
            A ProbeResult with calibrated consistency and token counts.
        """
        profile = self._get_profile(task)

        # --- Consistency score ---
        consistency = self._rng.gauss(profile.consistency_mean, profile.consistency_std)
        consistency = max(0.1, min(1.0, consistency))

        # --- Token count ---
        tokens = self._rng.randint(profile.token_min, profile.token_max)

        # --- Answer ---
        correct = task.ground_truth or "Unknown"

        # Probe gets it right if consistency >= 0.5 (majority vote)
        probe_is_correct = consistency >= 0.5
        _probe_answer = correct if probe_is_correct else self._rng.choice(_wrong_variants(correct))

        # --- Raw CoT-SC samples ---
        raw = _generate_samples(correct, consistency, self.n_samples, self._rng)

        # Recompute exact consistency from samples (cleaner than using raw float)
        majority = max(set(raw), key=raw.count)
        actual_consistency = raw.count(majority) / len(raw)

        return ProbeResult(
            task_id=task.task_id,
            answer=majority,
            consistency_score=round(actual_consistency, 3),
            tokens_used=tokens,
            raw_outputs=raw,
            model_name="SimulatedProbe-v1",
            latency_ms=round(tokens * 0.8 + self._rng.gauss(50, 10), 1),
        )

    def mas_beats_probe(self, task: Task) -> bool:
        """
        Simulate whether the real MAS would beat the probe on this task.
        Used in label generation (Day 6).
        """
        profile = self._get_profile(task)
        return self._rng.random() < profile.mas_beats_probe_rate


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function (matches ProbeAgentFn signature for pipeline injection)
# ─────────────────────────────────────────────────────────────────────────────

_default_probe = SimulatedProbe(seed=42)


def simulated_probe_agent(task: Task) -> ProbeResult:
    """Drop-in replacement for mock_probe_agent. Stateful singleton."""
    return _default_probe.run(task)


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from shared.data_loader import load_split

    print("=" * 72)
    print("  GateOrchestra -- Simulated Probe Agent (Day 4)")
    print("=" * 72)

    probe = SimulatedProbe(seed=42)
    tasks = load_split("val")

    # Sample 3 from each type for comparison
    by_type: dict[str, list] = {}
    for t in tasks:
        src = t.source_dataset or "unknown"
        by_type.setdefault(src, []).append(t)

    print(
        f"\n  {'task_id':<22} {'type':<22} {'consist':>7}  {'tokens':>6}  {'correct?':>9}  answer[:30]"
    )
    print(f"  {'-'*22} {'-'*22} {'-'*7}  {'-'*6}  {'-'*9}  {'-'*30}")

    total_consist: dict[str, list[float]] = {}
    total_tokens: dict[str, list[int]] = {}

    for src, src_tasks in sorted(by_type.items()):
        sample = src_tasks[:5]
        for t in sample:
            result = probe.run(t)
            gt = t.ground_truth or "?"
            is_correct = result.answer.lower().strip() == gt.lower().strip()
            correct_str = "YES" if is_correct else "no"
            ans_short = result.answer[:28] + ".." if len(result.answer) > 28 else result.answer
            short_src = src.replace("synthetic_", "")
            print(
                f"  {t.task_id:<22} {short_src:<22} {result.consistency_score:>7.3f}  "
                f"{result.tokens_used:>6}  {correct_str:>9}  {ans_short}"
            )
            total_consist.setdefault(src, []).append(result.consistency_score)
            total_tokens.setdefault(src, []).append(result.tokens_used)
        print()

    print("  Calibration summary:")
    print(f"  {'Type':<25} {'Avg Consistency':>16}  {'Avg Tokens':>10}")
    print(f"  {'-'*25} {'-'*16}  {'-'*10}")
    for src in sorted(total_consist):
        ac = sum(total_consist[src]) / len(total_consist[src])
        at = sum(total_tokens[src]) / len(total_tokens[src])
        short = src.replace("synthetic_", "")
        print(f"  {short:<25} {ac:>16.3f}  {at:>10.0f}")

    print("\n[OK] Day 4 complete. Simulated probe ready. Replaces flat mock.")

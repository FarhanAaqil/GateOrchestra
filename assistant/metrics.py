"""
assistant/metrics.py
====================
Operational numbers from the usage events the pipeline logs.

summarize() feeds the console's Admin page; prometheus() renders the same
numbers in Prometheus text format for a scraper.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

from assistant.store import Store


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return float(ordered[index])


def summarize(store: Store, hours: float = 24.0) -> dict[str, Any]:
    since = time.time() - hours * 3600
    turns = store.events("chat", since=since)
    errors = store.events("error", since=since)
    blocked = store.events("blocked", since=since)
    live = [t for t in turns if not t.get("cached")]
    latency = [float(t.get("latency_ms") or 0) for t in live]
    tokens = Counter()
    for t in live:
        for stage, value in (t.get("tokens") or {}).items():
            tokens[stage] += int(value or 0)
    total = len(turns)
    return {
        "window_hours": hours,
        "turns": total,
        "cached": total - len(live),
        "cache_hit_rate": (total - len(live)) / total if total else 0.0,
        "errors": len(errors),
        "error_codes": dict(Counter(e.get("code", "unknown") for e in errors)),
        "blocked": len(blocked),
        "tokens": dict(tokens),
        "total_tokens": sum(tokens.values()),
        "avg_tokens_per_turn": sum(tokens.values()) / len(live) if live else 0.0,
        "latency_ms": {"p50": _percentile(latency, 0.5), "p95": _percentile(latency, 0.95)},
        "gates": dict(Counter(t.get("gate", "unknown") for t in turns)),
        "decisions": dict(Counter(t.get("decision", "unknown") for t in turns)),
        "task_types": dict(Counter(t.get("task_type", "unknown") for t in turns)),
        "recent": [
            {
                k: t.get(k)
                for k in (
                    "ts",
                    "gate",
                    "decision",
                    "task_type",
                    "total_tokens",
                    "latency_ms",
                    "cached",
                )
            }
            for t in turns[:25]
        ],
    }


def prometheus(store: Store, hours: float = 24.0) -> str:
    s = summarize(store, hours)
    lines = [
        "# HELP gateorchestra_turns Chat turns in the window.",
        "# TYPE gateorchestra_turns gauge",
        f"gateorchestra_turns {s['turns']}",
        "# HELP gateorchestra_cache_hits Turns answered from the cache.",
        "# TYPE gateorchestra_cache_hits gauge",
        f"gateorchestra_cache_hits {s['cached']}",
        "# HELP gateorchestra_errors Failed turns in the window.",
        "# TYPE gateorchestra_errors gauge",
        f"gateorchestra_errors {s['errors']}",
        "# HELP gateorchestra_blocked Messages refused by the prompt guard.",
        "# TYPE gateorchestra_blocked gauge",
        f"gateorchestra_blocked {s['blocked']}",
        "# HELP gateorchestra_tokens Tokens spent by stage.",
        "# TYPE gateorchestra_tokens gauge",
        *(
            f'gateorchestra_tokens{{stage="{stage}"}} {value}'
            for stage, value in sorted(s["tokens"].items())
        ),
        "# HELP gateorchestra_latency_ms Answer latency percentiles.",
        "# TYPE gateorchestra_latency_ms gauge",
        f'gateorchestra_latency_ms{{quantile="0.5"}} {s["latency_ms"]["p50"]}',
        f'gateorchestra_latency_ms{{quantile="0.95"}} {s["latency_ms"]["p95"]}',
        "# HELP gateorchestra_gate_turns Turns routed by each gate.",
        "# TYPE gateorchestra_gate_turns gauge",
        *(
            f'gateorchestra_gate_turns{{gate="{gate}"}} {n}'
            for gate, n in sorted(s["gates"].items())
        ),
    ]
    return "\n".join(lines) + "\n"

"""
scripts/demo_run.py
===================
End-to-end demo: runs GateOrchestra pipeline on 10 mock tasks.

Run with:
    python scripts/demo_run.py

Output: gate decisions and token summary printed + logs/demo_token_log.json written.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make sure repo root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gate.random_gate import RandomGate
from gate.rule_based_gate import RuleBasedGate
from integration.pipeline import run_batch
from shared.config import K_DEFAULT, LOGS_DIR
from shared.token_logger import TokenAccountant
from tests.mocks.mock_dataset import get_mock_tasks
from tests.mocks.mock_orchestrator import mock_orchestrator
from tests.mocks.mock_probe_agent import mock_probe_agent

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


def main():
    tasks = get_mock_tasks()
    print(f"\n{'='*60}")
    print(f"  GateOrchestra Demo — {len(tasks)} tasks")
    print(f"{'='*60}\n")

    accountant = TokenAccountant()

    for gate_cls, method_name in [
        (RuleBasedGate(), "RuleBasedGate"),
        (RandomGate(escalation_rate=0.4, seed=42), "RandomGate"),
    ]:
        print(f"\n── {method_name} ──────────────────────────────────")
        results = run_batch(
            tasks,
            gate_cls,
            mock_probe_agent,
            mock_orchestrator,
            accountant,
            k=K_DEFAULT,
            method=method_name,
        )

        n_stop = sum(1 for r in results if r.gate_decision and r.gate_decision.decision == "STOP")
        n_escalate = sum(
            1 for r in results if r.gate_decision and r.gate_decision.decision == "ESCALATE"
        )
        avg_tokens = sum(r.tokens_spent for r in results) / len(results)

        print(f"  STOP:     {n_stop}/{len(tasks)}")
        print(f"  ESCALATE: {n_escalate}/{len(tasks)}")
        print(f"  Avg tokens/task: {avg_tokens:.0f}")

        for r in results:
            dec = r.gate_decision.decision if r.gate_decision else "N/A"
            correct = "✓" if r.is_correct else ("✗" if r.is_correct is False else "?")
            print(f"    {r.task_id:<18} {dec:<10} tokens={r.tokens_spent:>4}  {correct}")

    # Save token log
    log_path = LOGS_DIR / "demo_token_log.json"
    accountant.save_to_json(log_path)
    print(f"\n✅ Token log saved to {log_path}")
    print(f"   Total methods: {list(accountant.get_total_by_method())}")


if __name__ == "__main__":
    main()

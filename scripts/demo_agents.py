"""
scripts/demo_agents.py
======================
Demo script showing Person 2's components in action:
  1. CoT-SC Probe Agent
  2. MAS Orchestrator (ReAct, Debate, Reflexion)
  3. Baselines (CoT-SC-only & Always-MAS)
  4. Token Accounting

Run from the repo root with:
    python scripts/demo_agents.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.schemas import Task
from shared.token_logger import TokenAccountant
from agents.probe_agent import ProbeAgent
from agents.orchestrator import MASOrchestrator
from agents.baselines import run_cot_sc_baseline, run_always_mas_baseline


def mock_llm(prompt: str, temperature: float, budget: int) -> tuple[str, int]:
    """Mock LLM caller for offline demo execution."""
    if "Eiffel" in prompt or "currency" in prompt:
        return "Thinking step by step: Eiffel tower is in Paris, France. France uses Euro.\nFinal Answer: Euro", 45
    elif "Tokyo" in prompt:
        return "Comparing cities: Tokyo has ~37 million in metro area.\nFinal Answer: Tokyo", 38
    elif "15 * 8" in prompt or "15" in prompt:
        return "Step 1: 10*8=80. Step 2: 5*8=40. Total=120.\nFinal Answer: 120", 30
    return "Analyzing input...\nFinal Answer: 42", 25


def main():
    print("=" * 65)
    print(" GateOrchestra - Person 2 Agent Systems Demo (Weeks 1-6)")
    print("=" * 65)

    accountant = TokenAccountant()

    # -------------------------------------------------------------
    # 1. Probe Agent (CoT-SC)
    # -------------------------------------------------------------
    print("\n[1] Running Probe Agent (CoT-SC)...")
    task1 = Task(
        task_id="demo_probe_01",
        question="What is 15 * 8?",
        ground_truth="120",
    )
    probe = ProbeAgent(n_samples=5, llm_caller=mock_llm)
    probe_res = probe.run(task1)
    print(f"    Task ID:           {probe_res.task_id}")
    print(f"    Majority Answer:   {probe_res.answer}")
    print(f"    Consistency Score: {probe_res.consistency_score * 100:.1f}%")
    print(f"    Tokens Consumed:   {probe_res.tokens_used}")

    # -------------------------------------------------------------
    # 2. MAS Orchestrator (Sub-Agent Pool)
    # -------------------------------------------------------------
    print("\n[2] Running MAS Orchestrator (ReAct / Debate / Reflexion)...")
    orch = MASOrchestrator(llm_caller=mock_llm)

    # Multi-hop task (triggers ReAct)
    task_react = Task(
        task_id="demo_react_01",
        question="Which country's capital has the Eiffel Tower and what currency does it use?",
        depth_score=4,
        context="Eiffel tower is in Paris, France. France uses Euro.",
        ground_truth="Euro",
    )
    ans_react, tok_react = orch.run(task_react, token_budget=500)
    strategy1 = orch.select_strategy(task_react)
    print(f"    Task: Multi-hop (depth=4) -> Selected Strategy: [{strategy1.upper()}]")
    print(f"    Answer: {ans_react} | Tokens spent: {tok_react}")

    # Parallel task (triggers Debate)
    task_debate = Task(
        task_id="demo_debate_01",
        question="Compare population of Tokyo vs London.",
        parallel_score=3,
        ground_truth="Tokyo",
    )
    ans_debate, tok_debate = orch.run(task_debate, token_budget=600)
    strategy2 = orch.select_strategy(task_debate)
    print(f"    Task: Parallel (parallel=3) -> Selected Strategy: [{strategy2.upper()}]")
    print(f"    Answer: {ans_debate} | Tokens spent: {tok_debate}")

    # -------------------------------------------------------------
    # 3. Baselines (CoT-SC-only & Always-MAS)
    # -------------------------------------------------------------
    print("\n[3] Running Evaluation Baselines...")
    cot_result = run_cot_sc_baseline(task1, probe_fn=probe, accountant=accountant)
    print(f"    [CoT-SC-only] Predicted: '{cot_result.predicted_answer}' | Correct: {cot_result.is_correct} | Tokens: {cot_result.tokens_spent}")

    mas_result = run_always_mas_baseline(task1, orchestrator_fn=orch, accountant=accountant, token_budget=1000)
    print(f"    [Always-MAS]  Predicted: '{mas_result.predicted_answer}' | Correct: {mas_result.is_correct} | Tokens: {mas_result.tokens_spent}")

    # -------------------------------------------------------------
    # 4. Token Accounting Summary
    # -------------------------------------------------------------
    print("\n[4] Token Accountant Summary:")
    for method, total in accountant.get_total_by_method().items():
        print(f"    Method: {method:<15} Total Tokens: {total}")

    print("\n" + "=" * 65)
    print(" Demo completed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    main()

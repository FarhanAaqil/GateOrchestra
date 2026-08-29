"""
agents/__init__.py
==================
Agents module for GateOrchestra (Person 2).
Exposes the ProbeAgent, MAS Orchestrator, LinUCBRouter, Providers, and Baselines.
"""

from agents.baselines import (
    run_always_mas_baseline,
    run_always_mas_batch,
    run_cot_sc_baseline,
    run_cot_sc_batch,
)
from agents.orchestrator import (
    DebateAgent,
    LinUCBRouter,
    MASOrchestrator,
    ReActAgent,
    ReflexionAgent,
    orchestrator,
)
from agents.probe_agent import ProbeAgent, probe_agent
from agents.providers import (
    call_groq,
    call_ollama,
    default_llm_caller,
    get_llm_caller,
)

__all__ = [
    "ProbeAgent",
    "probe_agent",
    "MASOrchestrator",
    "orchestrator",
    "LinUCBRouter",
    "ReActAgent",
    "DebateAgent",
    "ReflexionAgent",
    "call_ollama",
    "call_groq",
    "get_llm_caller",
    "default_llm_caller",
    "run_cot_sc_baseline",
    "run_cot_sc_batch",
    "run_always_mas_baseline",
    "run_always_mas_batch",
]

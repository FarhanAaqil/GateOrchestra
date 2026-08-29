"""
agents/orchestrator/__init__.py
===============================
Orchestrator package for GateOrchestra (Person 2).
"""

from agents.orchestrator.bandit_router import LinUCBRouter
from agents.orchestrator.orchestrator import MASOrchestrator, orchestrator
from agents.orchestrator.sub_agents import DebateAgent, ReActAgent, ReflexionAgent

__all__ = [
    "MASOrchestrator",
    "orchestrator",
    "LinUCBRouter",
    "ReActAgent",
    "DebateAgent",
    "ReflexionAgent",
]

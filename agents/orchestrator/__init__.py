"""
agents/orchestrator/__init__.py
===============================
Orchestrator package for GateOrchestra (Person 2).
"""

from agents.orchestrator.bandit_router import LinUCBRouter
from agents.orchestrator.orchestrator import (
    MASOrchestrator,
    get_default_mas_orchestrator,
    orchestrator,
)
from agents.orchestrator.sub_agents import DebateAgent, ReActAgent, ReflexionAgent

__all__ = [
    "MASOrchestrator",
    "get_default_mas_orchestrator",
    "orchestrator",
    "LinUCBRouter",
    "ReActAgent",
    "DebateAgent",
    "ReflexionAgent",
]

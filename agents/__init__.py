"""
agents/__init__.py
==================
Agents module for GateOrchestra (Person 2).
Exposes the ProbeAgent and baselines/orchestrator interfaces.
"""

from agents.probe_agent import ProbeAgent, probe_agent

__all__ = ["ProbeAgent", "probe_agent"]

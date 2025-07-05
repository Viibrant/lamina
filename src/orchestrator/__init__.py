"""Orchestrator layer for Lamina.
Coordinates interactions between agents and the main application.
"""

from .dispatcher import dispatch

__all__ = [
    "dispatch",
]

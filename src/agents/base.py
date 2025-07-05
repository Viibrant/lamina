from abc import ABC, abstractmethod
from typing import Any
from src.models import AgentRequest, AgentResponse


class BaseAgent(ABC):
    """
    Base class for all agents in Lamina.
    """

    name: str = "base"

    @abstractmethod
    def run(self, request: AgentRequest) -> AgentResponse:
        """
        Execute the agent on a task.
        Returns an AgentResponse containing:
        - output (str): final output
        - metadata (AgentMetadata): extra info like tokens used, tools called, etc
        - steps (optional list[str]): explanation or trace of steps taken"""
        pass

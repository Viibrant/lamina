from abc import ABC, abstractmethod
from typing import Any
from src.models import AgentRequest, AgentResponse


class BaseAgent(ABC):
    """
    Base class for all agents in Lamina.
    """

    name: str = "base"

    def __init__(self, llm_client: Any = None):
        """
        Initialise the agent with an optional LLM client.

        Args:
            llm_client (Any): Optional LLM client for generating responses.
        """
        self.llm_client = llm_client

    @abstractmethod
    def run(self, request: AgentRequest) -> AgentResponse:
        """
        Execute the agent on a task.
        Returns an AgentResponse containing:
        - output (str): final output
        - metadata (AgentMetadata): extra info like tokens used, tools called, etc
        - steps (optional list[str]): explanation or trace of steps taken"""
        pass

from abc import ABC, ABCMeta, abstractmethod
from typing import Any
from src.models import AgentRequest, AgentResponse


class AgentMeta(ABCMeta):
    """
    Metaclass that auto-registers all concrete Agent subclasses.
    """

    REGISTRY: list[type["BaseAgent"]] = []

    def __new__(cls: type, name: str, bases: tuple, attrs: dict) -> type:
        new_cls = super().__new__(cls, name, bases, attrs)
        if not attrs.get("abstract", False) and name != "BaseAgent":
            AgentMeta.REGISTRY.append(new_cls)
        return new_cls


class BaseAgent(ABC, metaclass=AgentMeta):
    """
    Base class for all agents in Lamina.
    Automatically registered via AgentMeta.
    """

    abstract = True
    name: str = "base"
    description: str

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

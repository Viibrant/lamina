from typing import Generic, TypeVar

from pydantic import BaseModel


class AgentRequest(BaseModel):
    """Request model for agent actions.
    Contains the action to be performed and any additional context needed.
    """

    input: str
    context: dict[str, str] = {}


class AgentMetadata(BaseModel):
    """Metadata about an agent's execution.
    Contains information like tokens used, tools called, duration, etc."""

    agent_name: str
    model: str | None = None
    tokens_used: int | None = None
    tools_called: list[str] = []
    duration_ms: int | None = None
    success: bool = True


class AgentResponse(BaseModel):
    """Response model for agent actions.
    Contains the output of the agent's execution and any metadata."""

    output: str
    metadata: AgentMetadata
    steps: list[str] | None = None  # Optional trace of steps taken by the agent


T = TypeVar("T", bound=BaseModel)


class LLMResponse(BaseModel, Generic[T]):
    """
    Represents a response from an LLM call, containing the model's output and metadata.
    """

    data: T
    metadata: AgentMetadata

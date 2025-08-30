from pydantic import BaseModel

from src.models import AgentResponse, AgentMetadata


class ExecutionStep(BaseModel):
    agent_name: str
    input: str


class ExecutionGraph:
    def __init__(self, nodes: list[ExecutionStep] | None = None):
        self.nodes = nodes or []


class Executor:
    async def run(self, graph: ExecutionGraph) -> AgentResponse:
        """
        Execute the given execution graph by running each step sequentially.

        Args:
            graph (ExecutionGraph): The execution graph containing steps to execute.

        Returns:
            AgentResponse: The final response after executing all steps.
        """
        metadata = AgentMetadata(agent_name="executor", model="gpt-4o-mini")
        output = ""

        for step in graph.nodes:
            # Simulate running each agent step
            output += f"Executed {step.agent_name} with input: {step.input}\n"
            metadata.tools_called.append(step.agent_name)

        return AgentResponse(output=output, metadata=metadata)

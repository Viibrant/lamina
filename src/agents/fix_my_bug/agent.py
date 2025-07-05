from src.agents.base import BaseAgent
from src.models import AgentRequest, AgentResponse, AgentMetadata


class FixMyBugAgent(BaseAgent):
    """
    An agent that helps fix bugs in code.
    """

    name: str = "fix_my_bug"

    def run(self, request: AgentRequest) -> AgentResponse:
        """
        Execute the agent on a bug-fixing task.

        Returns a dict with:
        - output (str): final output
        - metadata (AgentMetadata): extra info like tokens used, tools called, etc
        - steps (optional list[str]): explanation or trace of steps taken
        """
        # TODO: Implement logic
        output = f"Fixed bug in code: {request.input}"
        metadata = AgentMetadata(
            agent_name=FixMyBugAgent.name,
            tokens_used=0,
            tools_called=[],
        )
        steps = [
            f"Received input: {request.input}",
            "Processed input to fix bug",
            f"Output generated: {output}",
        ]
        return AgentResponse(output=output, metadata=metadata, steps=steps)

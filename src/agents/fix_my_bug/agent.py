from src.agents.base import BaseAgent
from src.agents.fix_my_bug.models import CodeFix
from src.agents.fix_my_bug.prompts import BUGFIX_PROMPT
from src.llm.client import LLMClient
from src.models import AgentRequest, AgentResponse, LLMResponse


class FixMyBugAgent(BaseAgent):
    """
    An agent that helps fix bugs in code.
    """

    name: str = "fix_my_bug"

    def __init__(self):
        """
        Initialise the FixMyBugAgent.

        Args:
            llm_client (LLMClient): Optional LLM client for generating responses.
        """
        self.llm_client = LLMClient(model_name="gpt-4o-mini")

    def run(self, request: AgentRequest) -> AgentResponse:
        if not request.input:
            raise ValueError("Input cannot be empty")

        prompt = BUGFIX_PROMPT.format(input=request.input)

        response: LLMResponse[CodeFix] = self.llm_client.call_with_schema(
            prompt=prompt,
            schema=CodeFix,
            system="You are a bug-fixing agent that provides clear and concise solutions to coding issues.",
        )

        response.metadata.agent_name = self.name

        fixed = response.data.fixed_code
        if not fixed:
            raise ValueError("No fixed_code returned by the agent")

        explanation = response.data.explanation

        return AgentResponse(
            output=fixed,
            metadata=response.metadata,
            steps=[explanation],
        )

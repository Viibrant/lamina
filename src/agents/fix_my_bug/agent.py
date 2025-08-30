from src.agents.base import BaseAgent
from src.agents.fix_my_bug.models import CodeFix
from src.agents.fix_my_bug.prompts import BUGFIX_PROMPT
from src.llm.client import LLMClient
from src.models import AgentRequest, AgentResponse, LLMResponse
from loguru import logger


class FixMyBugAgent(BaseAgent):
    """
    An agent that helps fix bugs in code.
    """

    name: str = "fix_my_bug"
    description: str = "Fixes broken code given an error or bug"

    def __init__(self):
        """
        Initialise the FixMyBugAgent.

        Args:
            llm_client (LLMClient): Optional LLM client for generating responses.
        """
        self.llm_client = LLMClient(model_name="gpt-4o-mini")

    def run(self, request: AgentRequest) -> AgentResponse:
        logger.debug(f"FixMyBugAgent received request: {request}")
        if not request.input:
            logger.error("Input cannot be empty")
            raise ValueError("Input cannot be empty")

        prompt = BUGFIX_PROMPT.format(input=request.input)
        logger.debug(f"Generated bugfix prompt: {prompt!r}")

        response: LLMResponse[CodeFix] = self.llm_client.call_with_schema(
            prompt=prompt,
            schema=CodeFix,
            system="You are a bug-fixing agent that provides clear and concise solutions to coding issues.",
        )

        logger.debug(f"LLM response: {response}")

        response.metadata.agent_name = self.name

        fixed = response.data.fixed_code
        if not fixed:
            logger.error("No fixed_code returned by the agent")
            raise ValueError("No fixed_code returned by the agent")

        explanation = response.data.explanation

        logger.info(f"Explanation: {explanation!r}")

        return AgentResponse(
            output=fixed,
            metadata=response.metadata,
            steps=[explanation],
        )

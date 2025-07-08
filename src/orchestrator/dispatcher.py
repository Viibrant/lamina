from loguru import logger

from src.agents import FixMyBugAgent
from src.models import AgentRequest, AgentResponse
from src.orchestrator.errors import NoMatchingAgentError

AGENT_REGISTRY = {
    FixMyBugAgent.name: FixMyBugAgent,
}


async def dispatch(request: AgentRequest) -> AgentResponse:
    """
    Decide which agent should handle the request
    and execute it.

    Args:
        request (AgentRequest): The request containing the action and parameters.
    Returns:
        AgentResponse: The response from the agent after execution.
    Raises:
        NoMatchingAgentError: If no agent matches the request.
    """
    logger.info(f"Dispatching request: {request}")
    input_text = request.input.lower()

    if "bug" in input_text or "error" in input_text:
        logger.info("Matched FixMyBugAgent for input.")
        agent = AGENT_REGISTRY["fix_my_bug"]
        response: AgentResponse = agent().run(request)
        logger.info(f"Agent response: {response}")
        return response

    logger.error(f"No suitable agent found for request: {request}")
    raise NoMatchingAgentError(f"No suitable agent found for request: {request}")

from src.models import AgentRequest, AgentResponse
from src.orchestrator.errors import NoMatchingAgentError
from src.orchestrator.intents import classify_intent
from src.agents.base import AgentMeta

from loguru import logger


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

    agents = {cls.name: cls for cls in AgentMeta.REGISTRY}
    decision = classify_intent(request.input, list(agents.keys()))

    if not decision.actionable or not decision.agent_name:
        logger.warning(f"No actionable agent found for request: {request.input}")
        raise NoMatchingAgentError(decision.reason)

    logger.info(
        f"Dispatching to agent: {decision.agent_name} for request: {request.input}"
    )
    agent_cls = agents.get(decision.agent_name)
    if not agent_cls:
        raise NoMatchingAgentError(f"Unknown agent: {decision.agent_name}")

    agent = agent_cls()
    return agent.run(request)

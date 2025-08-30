from loguru import logger

from src.agents.base import AgentMeta
from src.models import AgentRequest, AgentResponse
from src.orchestrator.errors import NoMatchingAgentError
from src.orchestrator.execution import Executor
from src.orchestrator.intents import IntentClassifier


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

    logger.info(f"Dispatching request: {request.input}")

    # Get all registered agents
    agents = {cls.name: cls for cls in AgentMeta.REGISTRY}
    agent_instances = [cls() for cls in agents.values()]
    # Instantiate the intent classifier
    intent_classifier = IntentClassifier()
    # Classify the request to find the appropriate agent
    decision = intent_classifier(request.input, agent_instances)
    logger.debug(f"Decision made: {decision}")

    # If the decision is not actionable or no agent is specified, raise an error
    # This means the request cannot be handled by any agent.
    if not decision.actionable or not decision.agent_name:
        logger.warning(f"No actionable agent found for request: {request.input}")
        raise NoMatchingAgentError(decision.reason)

    logger.info(
        f"Dispatching to agent: {decision.agent_name} for request: {request.input}"
    )
    # Get the agent class based on the decision
    # If the agent name is not found in the registry, raise an error.
    agent_cls = agents.get(decision.agent_name)
    if not agent_cls:
        raise NoMatchingAgentError(f"Unknown agent: {decision.agent_name}")

    # If the decision is to use the planner, we need to run the planner agent
    # and then execute the plan using the executor.
    if decision.agent_name == "planner":
        # Instantiate the planner agent and run it
        planner = agents["planner"]()
        plan = planner.run(request)
        logger.info(f"Generated execution plan with {len(plan.steps)} steps")
        executor = Executor()
        result = await executor.run(plan)
        return result
    else:
        agent = agent_cls()
        return agent.run(request)

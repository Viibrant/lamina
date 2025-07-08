from pydantic import BaseModel
from src.llm.client import LLMClient

llm = LLMClient(model_name="gpt-4o-mini")


class DispatchDecision(BaseModel):
    actionable: bool
    agent_name: str | None = None
    reason: str


def classify_intent(input_text: str, available_agents: list[str]) -> DispatchDecision:
    """
    Classify the intent of the input text to determine if it is actionable and which agent should handle it.

    Args:
        input_text (str): The input text to classify.
        available_agents (list[str]): List of available agent names.
    Returns:
        DispatchDecision: A structured decision containing:
            - actionable (bool): Whether the input is actionable.
            - agent_name (str | None): The name of the agent to handle the request, or None if not actionable.
            - reason (str): Explanation for the decision.
    """
    prompt = f"""
    You are a request classifier. Your job is to determine if an input is actionable and if so, which agent should handle it.

    Available agents: {available_agents}

    Input: "{input_text}"
    """
    return llm.call_with_schema(
        prompt=prompt,
        schema=DispatchDecision,
        system="You classify input for agent dispatch.",
    ).data

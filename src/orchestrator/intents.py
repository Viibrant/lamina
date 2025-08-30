from collections.abc import Sequence

from pydantic import BaseModel

from src.agents.base import BaseAgent
from src.llm.client import LLMClient
from src.tools.base import tool_schema_from_model

llm = LLMClient(model_name="gpt-4o-mini")


class DispatchDecision(BaseModel):
    """
    Represents the classifier's decision about whether a request is actionable,
    and if so, which agent should handle it.
    """

    actionable: bool
    agent_name: str | None = None
    reason: str


class IntentClassifier:
    """
    Handles routing a user request to the appropriate agent using an LLM.
    """

    SYSTEM_PROMPT = (
        "You classify user requests and decide which agent, if any, should handle them."
    )

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient(model_name="gpt-4o-mini")

    def __call__(
        self, user_input: str, agents: Sequence[BaseAgent]
    ) -> DispatchDecision:
        """
        Classify the user's intent and return a structured decision about agent routing.

        Args:
            user_input: The raw input text from the user.
            agents: The list of available agents to choose from.

        Returns:
            DispatchDecision: The structured decision result.
        """
        prompt = self._build_prompt(user_input, agents)

        schema = tool_schema_from_model(DispatchDecision, name="dispatch_decision")

        response = self.llm.chat_completion(
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            functions=[schema],
            function_call={"name": "dispatch_decision"},
        )

        return DispatchDecision(**response.function_call.arguments)

    def _build_prompt(self, text: str, agents: Sequence[BaseAgent]) -> str:
        """
        Build a clean, readable prompt to pass to the classifier model.

        Args:
            text: The user input.
            agents: The available agents.

        Returns:
            A formatted prompt string.
        """
        agent_descriptions = "\n".join(
            f"- **{a.name}**: {getattr(a, 'description', 'No description')}"
            for a in agents
        )
        return (
            "Decide whether the request is actionable. "
            "If it is, choose the best agent and explain why.\n\n"
            f"Available agents:\n{agent_descriptions}\n\n"
            f'User input:\n"""\n{text}\n"""'
        )

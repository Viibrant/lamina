from src.models import AgentRequest
from src.orchestrator.execution import ExecutionGraph, ExecutionStep
from src.llm.client import LLMClient
from src.tools.base import tool_schema_from_model


class PlannerAgent:
    """
    PlannerAgent coordinates multi-step executions by planning tasks across agents.
    It breaks down user requests into a sequence of steps, each calling a specific agent.
    """

    name: str = "planner"
    description: str = "Plans multi-step executions by coordinating other agents."

    def __init__(self) -> None:
        self.llm = LLMClient(model_name="gpt-4o")

    def plan(self, request: AgentRequest) -> ExecutionGraph:
        # Build prompt
        prompt = (
            "You are a planning agent. Break down the user's request into a sequence of steps.\n"
            "Each step should call a named agent and provide the input it needs.\n"
            "Only use agents you know about, and keep the steps minimal.\n"
            f"User request:\n{request.input}"
        )

        # Get schema for array of ExecutionStep
        schema = tool_schema_from_model(
            ExecutionStep, name="execution_plan", is_array=True
        )

        # Call the LLM
        response = self.llm.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a task planner that breaks down requests into agent steps.",
                },
                {"role": "user", "content": prompt},
            ],
            functions=[schema],
            function_call={"name": "execution_plan"},
        )

        steps = [ExecutionStep(**step) for step in response.function_call.arguments]
        return ExecutionGraph(steps=steps)
    
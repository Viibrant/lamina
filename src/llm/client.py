import json
import time
from pathlib import Path
from typing import TypeVar

import litellm
from loguru import logger
from pydantic import BaseModel

from src.models import AgentMetadata, LLMResponse

T = TypeVar("T", bound=BaseModel)

LOG_PATH = Path(__file__).parent / "logs" / "llm_calls.jsonl"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


class LLMClient:
    """
    A client for interacting with LLMs.
    """

    def __init__(self, model_name="openai/gpt-4o"):
        self.model_name = model_name

    def chat_completion(self, **kwargs) -> object:
        """
        Generate a chat completion using the specified model.
        """
        logger.debug(f"Calling LLM with model: {self.model_name} | kwargs: {kwargs}")
        response = litellm.completion(model=self.model_name, **kwargs)

        # Use .content if available, otherwise fallback to str(response)
        content = getattr(response, "content", None)
        if content is None:
            content = str(response)
        logger.debug(f"LLM response: {content!r}")
        return response

    def call(self, prompt: str, system: str | None = None) -> str:
        """
        Generate text from the LLM based on the given prompt.
        """

        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})

        logger.debug(f"Calling LLM with prompt: {prompt!r} | system: {system!r}")
        response = litellm.completion(
            model=self.model_name,
            messages=messages,
        )

        # Use .content if available, otherwise fallback to str(response)
        content = getattr(response, "content", None)
        if content is None:
            content = str(response)
        logger.debug(f"LLM response: {content!r}")
        return content

    def call_with_schema(
        self,
        prompt: str,
        schema: type[T],
        system: str | None = None,
        agent_name: str = "",
    ) -> LLMResponse:
        """
        Call the LLM and validate the response against a given Pydantic schema.

        Args:
            prompt (str): The user prompt to send to the LLM.
            schema (type[T]): A Pydantic model class to use for validating the structured response.
            system (str | None): Optional system message to guide the LLM's behaviour.

        Returns:
            T: An instance of the provided Pydantic schema populated with the model's response.

        Example:
        ```python
            class Answer(BaseModel):
                answer: str

            result = client.call_with_schema("What's 2 + 2?", Answer)
            print(result.answer)  # '4'
        ```
        """
        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})

        logger.debug(
            f"Calling LLM with schema. Prompt: {prompt!r} | system: {system!r} | schema: {schema.__name__}"
        )
        litellm.enable_json_schema_validation = True

        start = time.time()
        llm_raw = litellm.completion(
            model=self.model_name,
            messages=messages,
            response_format=schema,
        )
        duration_ms = int((time.time() - start) * 1000)

        parsed: T = schema.model_validate_json(llm_raw.choices[0].message.content)
        meta = AgentMetadata(
            agent_name=agent_name or "",
            model=llm_raw.model,
            tokens_used=llm_raw.usage.total_tokens,
            tools_called=[],
            duration_ms=duration_ms,
        )

        logger.info(f"LLM response parsed in {duration_ms}ms")

        # Save the raw response to a log file
        log_entry = {
            "timestamp": time.time(),
            "model": self.model_name,
            "prompt": prompt,
            "system": system,
            "response": llm_raw.choices[0].message.content,
            "parsed": parsed.model_dump(),
            "schema": schema.__name__,
            "agent_name": agent_name,
            "duration_ms": duration_ms,
        }
        with LOG_PATH.open("a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return LLMResponse(data=parsed, metadata=meta)

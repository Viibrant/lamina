import pytest
from pydantic import BaseModel

from src.llm.client import LLMClient


class DummySchema(BaseModel):
    city: str


@pytest.mark.vcr(record_mode="once")
def test_success():
    client = LLMClient(model_name="gpt-4o-mini")
    assert client.model_name == "gpt-4o-mini"

    response = client.call_with_schema(
        prompt="What is the capital of France?",
        schema=DummySchema,
        system="You are a helpful assistant.",
    )

    assert response.data.city == "Paris"
    assert response.metadata.agent_name == ""

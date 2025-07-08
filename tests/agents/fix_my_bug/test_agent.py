import pytest
from src.agents.fix_my_bug import FixMyBugAgent
from src.models import AgentRequest


@pytest.mark.vcr(record_mode="new_episodes")
def test_successful_bug_fix():
    agent = FixMyBugAgent()
    request = AgentRequest(
        input="""
My function is supposed to return the sum of two numbers, but it doesn't work:
```python
def add_numbers(a, b):
    return a - b
```
"""
    )
    response = agent.run(request)

    assert response.output is not None, "The agent should return a response"
    assert "a + b" in response.output, (
        "The fixed code should use addition instead of subtraction"
    )
    assert response.metadata is not None, "The agent should return metadata"

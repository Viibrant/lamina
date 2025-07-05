from fastapi import FastAPI
from src.models import AgentRequest
from src.orchestrator import dispatch
from src.orchestrator.errors import NoMatchingAgentError

app = FastAPI()


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/agent")
async def agent_action(request: AgentRequest):
    # TODO: Handle the agent action based on the request
    try:
        response = await dispatch(request)
        return response
    except NoMatchingAgentError as e:
        return {"error": str(e)}, 404
    except Exception as e:
        return {"error": "An unexpected error occurred", "details": str(e)}, 500

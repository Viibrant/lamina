from fastapi import FastAPI
from src.models import AgentRequest
from src.orchestrator import dispatch

app = FastAPI()


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/agent")
async def agent_action(request: AgentRequest):
    # TODO: Handle the agent action based on the request
    response = await dispatch(request)
    return response

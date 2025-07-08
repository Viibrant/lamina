from fastapi import FastAPI
from loguru import logger

from src.models import AgentRequest
from src.orchestrator import dispatch
from src.orchestrator.errors import NoMatchingAgentError

app = FastAPI()


@app.get("/health")
async def health_check():
    logger.info("Health check endpoint called")
    return {"status": "ok"}


@app.post("/agent")
async def agent_action(request: AgentRequest):
    logger.info(f"Received /agent request: {request}")
    try:
        response = await dispatch(request)
        logger.info(f"Dispatch response: {response}")
        return response
    except NoMatchingAgentError as e:
        logger.error(f"NoMatchingAgentError: {e}")
        return {"error": str(e)}, 404
    except Exception as e:
        logger.exception("Unexpected error in /agent endpoint")
        return {"error": "An unexpected error occurred", "details": str(e)}, 500

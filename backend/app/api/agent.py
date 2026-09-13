from pathlib import Path

from fastapi import APIRouter, File, Form, Query, Request, UploadFile

from backend.app.agent.schemas import AgentRequest, AgentResponse, AgentRunTrace
from backend.app.api.uploads import store_temporary_upload
from backend.app.models.errors import InvalidInputError

router = APIRouter()


@router.get("/executions", response_model=list[AgentRunTrace])
def list_agent_executions(
    request: Request, limit: int = Query(default=20, ge=1, le=50)
):
    return request.app.state.agent_traces.list(limit)


@router.post("/run", response_model=AgentResponse)
async def run_agent(
    request: Request,
    image: UploadFile = File(...),
    message: str = Form(..., min_length=1, max_length=8000),
    max_steps: int = Form(default=6, ge=1, le=10),
    max_new_tokens: int = Form(default=256, ge=64, le=512),
):
    temporary: Path | None = None
    try:
        if not message.strip():
            raise InvalidInputError("Message cannot be empty")
        temporary = await store_temporary_upload(request.app.state.settings, image)
        agent_request = AgentRequest(
            message=message,
            image_path=str(temporary),
            max_steps=max_steps,
            max_new_tokens=max_new_tokens,
        )
        return await request.app.state.vision_agent.run(agent_request)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

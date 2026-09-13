from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from backend.app.agent.schemas import (
    AgentErrorDetail,
    AgentObservation,
    AgentStep,
    AgentToolCall,
    AgentToolDefinition,
)
from backend.app.schemas.tool_result import Artifact


class AgentState(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    run_id: str
    user_message: str
    original_image_path: str
    tool_definitions: list[AgentToolDefinition]
    max_steps: int
    max_new_tokens: int
    step_count: int = 0
    observations: list[AgentObservation] = Field(default_factory=list)
    tool_calls: list[AgentToolCall] = Field(default_factory=list)
    steps: list[AgentStep] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    call_signatures: set[str] = Field(default_factory=set)
    final_answer: str | None = None
    started_at: str
    finished_at: str | None = None
    duration_ms: float | None = None
    status: str = "RUNNING"
    error: AgentErrorDetail | None = None

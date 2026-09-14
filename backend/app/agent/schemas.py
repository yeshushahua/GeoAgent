from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, model_validator

from backend.app.schemas.tool_result import Artifact, ToolResult
from backend.app.agent.aggregation import WorkflowSummary


class AgentToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    input_schema: dict[str, JsonValue]


class AgentToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["tool_call"]
    tool_name: str = Field(min_length=1)
    arguments: dict[str, JsonValue]


class AgentFinal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["final"]
    answer: str = Field(min_length=1)


AgentDecision = Annotated[AgentToolCall | AgentFinal, Field(discriminator="type")]
AGENT_DECISION_ADAPTER = TypeAdapter(AgentDecision)


class AgentErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str = Field(min_length=1)
    message: str = Field(min_length=1)


class AgentStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int = Field(ge=1)
    decision_type: Literal["tool_call", "final", "blocked"]
    tool_name: str | None = None
    arguments_summary: dict[str, JsonValue] = Field(default_factory=dict)
    observation_summary: dict[str, JsonValue] = Field(default_factory=dict)
    success: bool | None = None
    execution_id: str | None = None
    duration_ms: float = Field(default=0, ge=0)
    planner_duration_ms: float = Field(default=0, ge=0)
    tool_duration_ms: float = Field(default=0, ge=0)
    model_load_duration_ms: float = Field(default=0, ge=0)
    prompt_encoding_duration_ms: float = Field(default=0, ge=0)
    model_inference_duration_ms: float = Field(default=0, ge=0)
    tool_overhead_ms: float = Field(default=0, ge=0)
    artifacts: list[Artifact] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    error_type: str | None = None
    state_transition: str


class AgentObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step: int = Field(ge=1)
    tool_name: str
    result: ToolResult


class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=8000)
    image_path: str = Field(min_length=1)
    max_steps: int = Field(default=6, ge=1, le=10)
    max_new_tokens: int = Field(default=256, ge=64, le=512)

    @model_validator(mode="after")
    def normalize_message(self):
        self.message = self.message.strip()
        if not self.message:
            raise ValueError("Message cannot be empty")
        return self


class AgentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    device: str
    step_count: int = Field(ge=0)
    tool_call_count: int = Field(ge=0)
    total_duration_ms: float = Field(ge=0)
    planner_duration_ms: float = Field(ge=0)
    tool_duration_ms: float = Field(ge=0)
    framework_overhead_ms: float = Field(ge=0)
    model_load_duration_ms: float = Field(default=0, ge=0)
    agent_model_load_duration_ms: float = Field(default=0, ge=0)
    tool_model_load_duration_ms: float = Field(default=0, ge=0)
    framework_runtime_overhead_ms: float = Field(default=0, ge=0)
    prompt_encoding_duration_ms: float = Field(default=0, ge=0)
    model_inference_duration_ms: float = Field(default=0, ge=0)
    tool_overhead_ms: float = Field(default=0, ge=0)
    allocated_vram_gib: float = Field(ge=0)
    peak_vram_gib: float = Field(ge=0)
    successful_steps: int = Field(default=0, ge=0)
    failed_steps: int = Field(default=0, ge=0)
    workflow_total_ms: float = Field(default=0, ge=0)


class AgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    success: bool
    answer: str = ""
    run_id: str = Field(min_length=1)
    status: Literal["COMPLETED", "FAILED"]
    steps: list[AgentStep] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    workflow: WorkflowSummary
    metadata: AgentMetadata
    error: AgentErrorDetail | None = None

    @model_validator(mode="after")
    def consistent_outcome(self):
        if self.success and (self.error is not None or not self.answer):
            raise ValueError("Successful AgentResponse requires an answer and no error")
        if not self.success and self.error is None:
            raise ValueError("Failed AgentResponse requires a structured error")
        return self


class AgentRunTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    success: bool
    started_at: str
    finished_at: str
    duration_ms: float = Field(ge=0)
    step_count: int = Field(ge=0)
    tool_sequence: list[str] = Field(default_factory=list)
    artifact_count: int = Field(ge=0)
    prompt_length: int = Field(ge=0)
    error_type: str | None = None

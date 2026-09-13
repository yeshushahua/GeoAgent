from __future__ import annotations

from collections import deque
from threading import Lock

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class ToolExecutionTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")
    execution_id: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    success: bool
    started_at: str
    finished_at: str
    duration_ms: float = Field(ge=0)
    arguments_summary: dict[str, JsonValue] = Field(default_factory=dict)
    prompt_length: int | None = Field(default=None, ge=0)
    error_type: str | None = None


class ToolTraceStore:
    def __init__(self, capacity: int = 50):
        self._items: deque[ToolExecutionTrace] = deque(maxlen=capacity)
        self._lock = Lock()

    def add(self, trace: ToolExecutionTrace) -> None:
        with self._lock:
            self._items.appendleft(trace)

    def list(self, limit: int = 20) -> list[ToolExecutionTrace]:
        with self._lock:
            return list(self._items)[:limit]

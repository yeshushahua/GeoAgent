from __future__ import annotations

from abc import ABC, abstractmethod
import re
from typing import ClassVar

from pydantic import BaseModel

from backend.app.schemas.tool_result import ToolResult
from backend.app.tools.context import ToolContext

TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


class BaseTool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    category: ClassVar[str]
    version: ClassVar[str] = "1.0"
    input_schema: ClassVar[type[BaseModel]]
    requires_gpu: ClassVar[bool] = False
    requires_model: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if getattr(cls, "__abstractmethods__", None):
            return
        name = getattr(cls, "name", "")
        if not TOOL_NAME_PATTERN.fullmatch(name):
            raise TypeError(f"Tool name must be stable lowercase snake_case: {name!r}")
        schema = getattr(cls, "input_schema", None)
        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            raise TypeError("Tool input_schema must be a Pydantic BaseModel class")

    def definition(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "version": self.version,
            "input_schema": self.input_schema.model_json_schema(),
            "requires_gpu": self.requires_gpu,
            "requires_model": self.requires_model,
        }

    @abstractmethod
    async def execute(
        self, inputs: BaseModel, context: ToolContext, execution_id: str
    ) -> ToolResult:
        raise NotImplementedError

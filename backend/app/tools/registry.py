from __future__ import annotations

from backend.app.tools.base import BaseTool


class DuplicateToolError(ValueError):
    pass


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise DuplicateToolError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> BaseTool:
        try:
            return self._tools.pop(name)
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> list[dict]:
        return [self._tools[name].definition() for name in sorted(self._tools)]

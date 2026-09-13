from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import time
from uuid import uuid4

from pydantic import ValidationError

from backend.app.models.errors import VlmError
from backend.app.detection.errors import DetectorError
from backend.app.schemas.tool_result import ToolError, ToolResult
from backend.app.tools.context import ToolContext
from backend.app.tools.errors import ToolExecutionError
from backend.app.tools.registry import ToolRegistry
from backend.app.tools.trace import ToolExecutionTrace


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, context: ToolContext):
        self.registry = registry
        self.context = context

    async def execute(self, tool_name: str, arguments: dict) -> ToolResult:
        execution_id = str(uuid4())
        started_at = _utc_now()
        timer = time.perf_counter()
        tool = None
        result: ToolResult
        try:
            tool = self.registry.get(tool_name)
            inputs = tool.input_schema.model_validate(arguments)
            async with asyncio.timeout(self.context.settings.tool_timeout_seconds):
                result = await tool.execute(inputs, self.context, execution_id)
            if result.tool != tool_name:
                raise RuntimeError("Tool returned a mismatched tool name")
        except KeyError:
            result = self._failure(tool_name, "TOOL_NOT_FOUND", f"Unknown tool: {tool_name}")
        except ValidationError as exc:
            fields = [f"{'.'.join(map(str, item['loc']))}: {item['msg']}" for item in exc.errors()]
            result = self._failure(
                tool_name, "INVALID_TOOL_INPUT", "Tool input validation failed", {"fields": fields}
            )
        except TimeoutError:
            result = self._failure(tool_name, "TOOL_TIMEOUT", "Tool execution timed out")
        except (VlmError, DetectorError, ToolExecutionError) as exc:
            self.context.logger.exception("[%s] %s failed", execution_id, tool_name)
            result = self._failure(tool_name, exc.code, str(exc))
        except Exception:
            self.context.logger.exception("[%s] Unexpected %s failure", execution_id, tool_name)
            result = self._failure(
                tool_name, "TOOL_EXECUTION_FAILED", "Tool execution failed; see server log"
            )
        finished_at = _utc_now()
        duration_ms = round((time.perf_counter() - timer) * 1000, 2)
        result.metadata.update({
            "execution_id": execution_id,
            "duration_ms": duration_ms,
            "started_at": started_at,
            "finished_at": finished_at,
        })
        prompt = arguments.get("prompt")
        self.context.trace_store.add(ToolExecutionTrace(
            execution_id=execution_id,
            tool=tool_name,
            success=result.success,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            prompt_length=len(prompt) if isinstance(prompt, str) else None,
            error_type=result.error.type if result.error else None,
        ))
        self.context.logger.info(
            "[%s] Tool %s completed success=%s duration_ms=%.2f",
            execution_id, tool_name, result.success, duration_ms,
        )
        return result

    @staticmethod
    def _failure(tool: str, code: str, message: str, details: dict | None = None) -> ToolResult:
        return ToolResult(
            success=False,
            tool=tool,
            error=ToolError(code=code, type=code, message=message, details=details or {}),
        )

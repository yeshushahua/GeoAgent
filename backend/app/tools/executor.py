from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
import time
from uuid import uuid4

from pydantic import ValidationError

from backend.app.models.errors import VlmError
from backend.app.geo_errors import GeoError
from backend.app.raster_errors import RasterError
from backend.app.detection.errors import DetectorError
from backend.app.open_vocabulary.errors import OpenVocabularyError
from backend.app.segmentation.errors import SegmentationError
from backend.app.schemas.tool_result import ToolError, ToolResult
from backend.app.tools.context import ToolContext
from backend.app.tools.errors import ToolExecutionError
from backend.app.tools.registry import ToolRegistry
from backend.app.tools.trace import ToolExecutionTrace


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def summarize_tool_arguments(arguments: dict) -> dict:
    """Keep useful parameters while removing paths, prompts, and binary content."""
    summary = {}
    for key, value in arguments.items():
        if key in {"image_path", "raster_path", "vector_path", "mask_path"} and value is not None:
            summary[key] = Path(str(value)).name
        elif key == "prompt" and isinstance(value, str):
            summary["prompt_length"] = len(value)
        elif isinstance(value, bytes):
            summary[key] = f"<{len(value)} bytes>"
        else:
            summary[key] = value
    return summary


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, context: ToolContext):
        self.registry = registry
        self.context = context

    async def execute(self, tool_name: str, arguments: dict) -> ToolResult:
        execution_id = str(uuid4())
        started_at = _utc_now()
        timer = time.perf_counter()
        tool = None
        arguments_summary = summarize_tool_arguments(arguments)
        result: ToolResult
        try:
            tool = self.registry.get(tool_name)
            inputs = tool.input_schema.model_validate(arguments)
            arguments_summary = summarize_tool_arguments(inputs.model_dump(mode="json"))
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
        except (
            VlmError, DetectorError, OpenVocabularyError, SegmentationError, ToolExecutionError,
            RasterError, GeoError,
        ) as exc:
            self.context.logger.exception("[%s] %s failed", execution_id, tool_name)
            result = self._failure(tool_name, exc.code, str(exc))
        except Exception:
            self.context.logger.exception("[%s] Unexpected %s failure", execution_id, tool_name)
            result = self._failure(
                tool_name, "TOOL_EXECUTION_FAILED", "Tool execution failed; see server log"
            )
        finished_at = _utc_now()
        duration_ms = round((time.perf_counter() - timer) * 1000, 2)
        inference_ms = result.metadata.get(
            "inference_ms",
            result.metadata.get("detector_inference_ms", result.metadata.get("latency_ms", 0)),
        )
        model_load_ms = result.metadata.get("model_load_ms", 0)
        prompt_encoding_ms = result.metadata.get("prompt_encoding_ms", 0)
        numeric_inference_ms = float(inference_ms) if isinstance(inference_ms, (int, float)) else 0.0
        numeric_load_ms = float(model_load_ms) if isinstance(model_load_ms, (int, float)) else 0.0
        numeric_prompt_ms = (
            float(prompt_encoding_ms) if isinstance(prompt_encoding_ms, (int, float)) else 0.0
        )
        result.metadata.update({
            "execution_id": execution_id,
            "duration_ms": duration_ms,
            "started_at": started_at,
            "finished_at": finished_at,
            "arguments_summary": arguments_summary,
            "model_load_ms": round(numeric_load_ms, 2),
            "inference_ms": round(numeric_inference_ms, 2),
            "prompt_encoding_ms": round(numeric_prompt_ms, 2),
            "tool_overhead_ms": round(max(
                0.0,
                duration_ms - numeric_load_ms - numeric_prompt_ms - numeric_inference_ms,
            ), 2),
        })
        prompt = arguments.get("prompt")
        self.context.trace_store.add(ToolExecutionTrace(
            execution_id=execution_id,
            tool=tool_name,
            success=result.success,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            arguments_summary=arguments_summary,
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

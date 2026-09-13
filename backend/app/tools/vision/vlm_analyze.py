from pathlib import Path

import anyio
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.models.errors import ModelLoadError
from backend.app.models.manager import ModelState
from backend.app.schemas.tool_result import ToolResult
from backend.app.tools.base import BaseTool
from backend.app.tools.context import ToolContext
from backend.app.tools.image_io import open_supported_image


class AnalyzeImageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_path: Path = Field(description="Reference to a PNG, JPEG, or WEBP image.")
    prompt: str = Field(min_length=1, max_length=8000, description="Question about the image.")
    max_new_tokens: int = Field(default=256, ge=64, le=512)

    @field_validator("prompt")
    @classmethod
    def prompt_must_contain_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Prompt cannot be empty")
        return value


class AnalyzeImageTool(BaseTool):
    name = "analyze_image"
    description = "Analyze an RGB image using the configured vision-language model and answer a natural-language question about the image."
    category = "vision"
    input_schema = AnalyzeImageInput
    requires_gpu = True
    requires_model = "Qwen3-VL-4B-Instruct"

    async def execute(self, inputs, context, execution_id):
        _, image, _ = open_supported_image(context, inputs.image_path)
        try:
            state = context.model_manager.state
            loaded_this_call = state == ModelState.UNLOADED
            if state == ModelState.UNLOADED:
                context.logger.info("[%s] Auto-loading Qwen3-VL for analyze_image", execution_id)
                await anyio.to_thread.run_sync(context.model_manager.load_model)
            elif state == ModelState.ERROR:
                status = context.model_manager.status()
                raise ModelLoadError(
                    f"Model is in ERROR state: {status.get('last_error') or 'unknown load error'}"
                )
            inference = await anyio.to_thread.run_sync(
                context.model_manager.infer,
                image,
                inputs.prompt,
                inputs.max_new_tokens,
            )
            return ToolResult(
                success=True,
                tool=self.name,
                data={"answer": inference.text, "image": inference.image.model_dump(mode="json")},
                metadata={
                    "model": inference.model,
                    "device": inference.device,
                    "dtype": inference.dtype,
                    "latency_ms": inference.latency_ms,
                    "model_load_ms": (
                        float(context.model_manager.status().get("load_time_s") or 0) * 1000
                        if loaded_this_call else 0
                    ),
                    "gpu_allocated_gb": inference.gpu.allocated_gb,
                    "gpu_reserved_gb": inference.gpu.reserved_gb,
                    "gpu_peak_gb": inference.gpu.peak_allocated_gb,
                    "max_new_tokens": inference.generation.max_new_tokens,
                },
            )
        finally:
            image.close()

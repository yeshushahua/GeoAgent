from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from backend.app.schemas.tool_result import ToolResult
from backend.app.tools.base import BaseTool
from backend.app.tools.context import ToolContext
from backend.app.tools.image_io import open_supported_image


class InspectImageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_path: Path = Field(description="Workflow artifact ID (preferred) or a PNG, JPEG, or WEBP image reference.")


class InspectImageTool(BaseTool):
    name = "inspect_image"
    description = "Inspect an RGB image and return its dimensions, format, mode, file size, and aspect ratio."
    category = "vision"
    input_schema = InspectImageInput

    async def execute(self, inputs, context, execution_id):
        path, image, image_format = open_supported_image(context, inputs.image_path)
        try:
            width, height = image.size
            return ToolResult(
                success=True,
                tool=self.name,
                data={
                    "width": width,
                    "height": height,
                    "mode": image.mode,
                    "format": image_format,
                    "file_size": path.stat().st_size,
                    "aspect_ratio": round(width / height, 6),
                },
            )
        finally:
            image.close()

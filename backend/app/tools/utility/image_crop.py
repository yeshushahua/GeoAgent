from datetime import datetime, timezone
from pathlib import Path

import anyio
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.schemas.tool_result import Artifact, ToolResult
from backend.app.tools.base import BaseTool
from backend.app.tools.context import ToolContext
from backend.app.tools.errors import InvalidCropError
from backend.app.tools.image_io import open_supported_image


class CropImageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_path: Path = Field(description="Workflow artifact ID (preferred) or a PNG, JPEG, or WEBP image reference.")
    x1: int = Field(ge=0, description="Left crop coordinate in pixels.")
    y1: int = Field(ge=0, description="Top crop coordinate in pixels.")
    x2: int = Field(gt=0, description="Exclusive right crop coordinate in pixels.")
    y2: int = Field(gt=0, description="Exclusive bottom crop coordinate in pixels.")

    @model_validator(mode="after")
    def ordered_coordinates(self):
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("x2/y2 must be greater than x1/y1")
        return self


class CropImageTool(BaseTool):
    name = "crop_image"
    description = (
        "Crop a rectangular pixel region and return it as an artifact. A named corner "
        "quarter means one quadrant from a 2-by-2 split: use floor(width / 2) and "
        "floor(height / 2) as the integer midpoints, never width / 4 and height / 4."
    )
    category = "utility"
    input_schema = CropImageInput

    async def execute(self, inputs, context, execution_id):
        _, image, _ = open_supported_image(context, inputs.image_path)
        try:
            if inputs.x2 > image.width or inputs.y2 > image.height:
                raise InvalidCropError(
                    f"Crop coordinates exceed image bounds {image.width}x{image.height}"
                )
            output_dir = context.tool_output_dir(
                execution_id, datetime.now(timezone.utc).strftime("%Y%m%d")
            )
            target = output_dir / "crop.png"
            cropped = image.crop((inputs.x1, inputs.y1, inputs.x2, inputs.y2)).convert("RGB")
            try:
                await anyio.to_thread.run_sync(cropped.save, target, "PNG")
            finally:
                cropped.close()
            return ToolResult(
                success=True,
                tool=self.name,
                data={"width": inputs.x2 - inputs.x1, "height": inputs.y2 - inputs.y1},
                artifacts=[Artifact(kind="image", path=str(target), mime_type="image/png")],
            )
        finally:
            image.close()

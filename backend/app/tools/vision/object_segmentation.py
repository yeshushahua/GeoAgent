from __future__ import annotations

from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import anyio
import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field

from backend.app.detection.schemas import BoundingBox
from backend.app.schemas.tool_result import Artifact, ToolResult
from backend.app.segmentation.errors import SegmentationLoadError
from backend.app.tools.base import BaseTool
from backend.app.tools.context import ToolContext
from backend.app.tools.image_io import open_supported_image


class SegmentObjectsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_path: Path = Field(description="Reference to a PNG, JPEG, or WEBP image.")
    boxes: list[BoundingBox] = Field(
        min_length=1,
        max_length=50,
        description=(
            "Pixel bounding boxes supplied by a previous detection observation. "
            "Each box must use x1, y1, x2, y2 in the same image coordinate space."
        ),
    )


def _color(index: int) -> tuple[int, int, int]:
    return (45 + index * 83 % 190, 45 + index * 113 % 190, 45 + index * 151 % 190)


def _overlay(image: Image.Image, masks: list[np.ndarray], segments) -> Image.Image:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    for index, mask in enumerate(masks):
        color = np.asarray(_color(index), dtype=np.float32)
        pixels[mask] = pixels[mask] * 0.52 + color * 0.48
    rendered = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(rendered)
    line_width = max(2, round(min(image.size) / 250))
    for index, segment in enumerate(segments):
        box = segment.bbox
        color = _color(index)
        draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline=color, width=line_width)
        draw.text((box.x1 + 3, max(0, box.y1 + 3)), segment.segment_id, fill=color)
    return rendered


class SegmentObjectsTool(BaseTool):
    name = "segment_objects"
    description = (
        "Precisely segment one or more detected objects with SAM 2.1 Base using explicit "
        "pixel bounding-box prompts. Returns per-instance mask pixel area and image-area ratio, "
        "individual PNG masks, and a combined overlay. Boxes must come from a prior observation."
    )
    category = "vision"
    input_schema = SegmentObjectsInput
    requires_gpu = True
    requires_model = "SAM-2.1-Base"

    async def execute(self, inputs, context: ToolContext, execution_id: str) -> ToolResult:
        if context.segmentation_manager is None:
            raise SegmentationLoadError("Segmentation manager is not configured")
        path, image, _ = open_supported_image(context, inputs.image_path)
        try:
            run = await anyio.to_thread.run_sync(partial(
                context.segmentation_manager.predict, path, inputs.boxes
            ))
            output_dir = context.tool_output_dir(
                execution_id, datetime.now(timezone.utc).strftime("%Y%m%d")
            )
            mask_dir = output_dir / "masks"
            mask_dir.mkdir()
            artifacts = []
            data = run.prediction.model_dump(mode="json")
            for item, mask in zip(data["segments"], run.masks):
                target = mask_dir / f"{item['segment_id']}.png"
                mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
                await anyio.to_thread.run_sync(partial(mask_image.save, target, "PNG"))
                mask_image.close()
                item["mask_artifact_path"] = str(target)
                artifacts.append(Artifact(kind="mask", path=str(target), mime_type="image/png"))
            overlay_path = output_dir / "segmentation-overlay.png"
            rendered = _overlay(image, run.masks, run.prediction.segments)
            await anyio.to_thread.run_sync(partial(rendered.save, overlay_path, "PNG"))
            rendered.close()
            artifacts.append(Artifact(kind="image", path=str(overlay_path), mime_type="image/png"))
            data["overlay_artifact_path"] = str(overlay_path)
            return ToolResult(
                success=True,
                tool=self.name,
                data=data,
                artifacts=artifacts,
                metadata={
                    "model": run.model,
                    "device": run.device,
                    "model_load_time_s": run.load_time_s,
                    "inference_ms": run.inference_ms,
                    "model_load_count": context.segmentation_manager.load_count,
                    "gpu_allocated_gb": run.gpu.allocated_gb,
                    "gpu_reserved_gb": run.gpu.reserved_gb,
                    "gpu_peak_gb": run.gpu.peak_allocated_gb,
                    "prompt_type": "bbox",
                    "offline": True,
                },
            )
        finally:
            image.close()

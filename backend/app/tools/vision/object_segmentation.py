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
from backend.app.segmentation.errors import SegmentationError, SegmentationLoadError
from backend.app.tools.base import BaseTool
from backend.app.tools.context import ToolContext
from backend.app.tools.image_io import open_supported_image
from backend.app.tools.errors import InvalidBBoxError, NoValidBoxesError


class SegmentObjectsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_path: Path = Field(description="Workflow artifact ID (preferred) or a PNG, JPEG, or WEBP image reference.")
    boxes: list[BoundingBox] | None = Field(
        default=None,
        max_length=50,
        description=(
            "Pixel bounding boxes supplied by a previous detection observation. "
            "Each box must use x1, y1, x2, y2 in the same image coordinate space."
        ),
    )
    detection_ids: list[str] | None = Field(
        default=None,
        min_length=1,
        max_length=50,
        description=(
            "Workflow detection IDs to segment. VisionAgent resolves these IDs to exact boxes; "
            "direct Tool/API callers should provide boxes."
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
        "workflow detection IDs or pixel bounding-box prompts. Returns per-instance mask pixel area and image-area ratio, "
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
            if not inputs.boxes:
                raise NoValidBoxesError(
                    "detection_ids require VisionAgent workflow resolution; no boxes were supplied"
                )
            indexed_boxes = []
            failures = []
            for input_index, box in enumerate(inputs.boxes):
                if box.x2 > image.width or box.y2 > image.height:
                    failures.append({
                        "input_index": input_index,
                        "bbox": box.model_dump(mode="json"),
                        "code": "INVALID_BBOX",
                        "message": f"Bounding box exceeds image bounds {image.width}x{image.height}",
                    })
                else:
                    indexed_boxes.append((input_index, box))
            if not indexed_boxes:
                raise InvalidBBoxError(
                    f"All bounding boxes exceed image bounds {image.width}x{image.height}"
                )
            load_count_before = context.segmentation_manager.load_count
            completed = []
            last_run = None
            inference_total_ms = 0.0
            try:
                batch_boxes = [box for _, box in indexed_boxes]
                run = await anyio.to_thread.run_sync(partial(
                    context.segmentation_manager.predict, path, batch_boxes
                ))
                last_run = run
                inference_total_ms += run.inference_ms
                completed.extend(
                    (input_index, segment, mask)
                    for (input_index, _), segment, mask in zip(
                        indexed_boxes, run.prediction.segments, run.masks
                    )
                )
            except SegmentationError as batch_error:
                if len(indexed_boxes) == 1:
                    raise
                for input_index, box in indexed_boxes:
                    try:
                        run = await anyio.to_thread.run_sync(partial(
                            context.segmentation_manager.predict, path, [box]
                        ))
                        last_run = run
                        inference_total_ms += run.inference_ms
                        completed.append((input_index, run.prediction.segments[0], run.masks[0]))
                    except SegmentationError as exc:
                        failures.append({
                            "input_index": input_index,
                            "bbox": box.model_dump(mode="json"),
                            "code": exc.code,
                            "message": str(exc),
                        })
                if not completed:
                    raise batch_error
            output_dir = context.tool_output_dir(
                execution_id, datetime.now(timezone.utc).strftime("%Y%m%d")
            )
            mask_dir = output_dir / "masks"
            mask_dir.mkdir()
            artifacts = []
            data = {
                "image_width": image.width,
                "image_height": image.height,
                "segment_count": len(completed),
                "segments": [],
                "failures": failures,
                "partial_failure": bool(failures),
            }
            rendered_segments = []
            rendered_masks = []
            for output_index, (input_index, segment, mask) in enumerate(completed, start=1):
                item = segment.model_dump(mode="json")
                item["segment_id"] = f"segment-{output_index:03d}"
                item["input_index"] = input_index
                target = mask_dir / f"{item['segment_id']}.png"
                mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
                await anyio.to_thread.run_sync(partial(mask_image.save, target, "PNG"))
                mask_image.close()
                item["mask_artifact_path"] = str(target)
                data["segments"].append(item)
                artifacts.append(Artifact(kind="mask", path=str(target), mime_type="image/png"))
            overlay_path = output_dir / "segmentation-overlay.png"
            rendered_segments = [segment for _, segment, _ in completed]
            rendered_masks = [mask for _, _, mask in completed]
            rendered = _overlay(image, rendered_masks, rendered_segments)
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
                    "model": last_run.model,
                    "device": last_run.device,
                    "model_load_time_s": last_run.load_time_s,
                    "model_load_ms": (
                        (last_run.load_time_s or 0) * 1000
                        if context.segmentation_manager.load_count > load_count_before else 0
                    ),
                    "inference_ms": round(inference_total_ms, 2),
                    "model_load_count": context.segmentation_manager.load_count,
                    "gpu_allocated_gb": last_run.gpu.allocated_gb,
                    "gpu_reserved_gb": last_run.gpu.reserved_gb,
                    "gpu_peak_gb": last_run.gpu.peak_allocated_gb,
                    "prompt_type": "bbox",
                    "offline": True,
                },
            )
        finally:
            image.close()

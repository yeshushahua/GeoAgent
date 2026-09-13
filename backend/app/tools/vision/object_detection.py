from __future__ import annotations

from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import anyio
from PIL import ImageDraw
from pydantic import BaseModel, ConfigDict, Field

from backend.app.detection.errors import DetectorLoadError
from backend.app.schemas.tool_result import Artifact, ToolResult
from backend.app.tools.base import BaseTool
from backend.app.tools.context import ToolContext
from backend.app.tools.image_io import open_supported_image


class DetectObjectsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_path: Path = Field(description="Reference to a PNG, JPEG, or WEBP image.")
    confidence: float = Field(
        default=0.25,
        ge=0.01,
        le=1,
        description="Minimum confidence score from 0.01 to 1. Defaults to 0.25.",
    )
    iou_threshold: float = Field(
        default=0.45,
        ge=0.01,
        le=1,
        description="Non-maximum suppression IoU threshold. Defaults to 0.45.",
    )
    classes: list[int | str] | None = Field(
        default=None,
        min_length=1,
        description=(
            "Optional COCO class IDs or exact English COCO class names. Omit to detect all "
            "supported classes. This is closed-set detection and cannot accept arbitrary labels."
        ),
    )


def _color(class_id: int) -> tuple[int, int, int]:
    return (
        48 + (class_id * 67) % 192,
        48 + (class_id * 97) % 192,
        48 + (class_id * 137) % 192,
    )


def _annotate(image, detections):
    annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    line_width = max(2, round(min(image.size) / 250))
    for item in detections:
        box = item.bbox
        color = _color(item.class_id)
        coordinates = (box.x1, box.y1, box.x2, box.y2)
        draw.rectangle(coordinates, outline=color, width=line_width)
        label = f"{item.class_name} {item.confidence:.2f}"
        left, top, right, bottom = draw.textbbox((box.x1, box.y1), label)
        label_height = bottom - top + 4
        label_top = max(0, box.y1 - label_height)
        label_right = min(image.width, box.x1 + (right - left) + 6)
        draw.rectangle((box.x1, label_top, label_right, box.y1), fill=color)
        draw.text((box.x1 + 3, label_top + 2), label, fill="white")
    return annotated


class DetectObjectsTool(BaseTool):
    name = "detect_objects"
    description = (
        "Detect known COCO object classes with YOLO11s. Returns exact class counts, "
        "confidence scores, and pixel bounding boxes plus an annotated image. Use this for "
        "object detection, counting, localization, or confidence requests. It is closed-set "
        "and cannot detect arbitrary user-defined categories."
    )
    category = "vision"
    input_schema = DetectObjectsInput
    requires_gpu = True
    requires_model = "YOLO11s-COCO"

    async def execute(self, inputs, context: ToolContext, execution_id: str) -> ToolResult:
        if context.detector_manager is None:
            raise DetectorLoadError("Detector manager is not configured")
        path, image, _ = open_supported_image(context, inputs.image_path)
        try:
            run = await anyio.to_thread.run_sync(partial(
                context.detector_manager.predict,
                path,
                confidence=inputs.confidence,
                iou_threshold=inputs.iou_threshold,
                classes=inputs.classes,
            ))
            output_dir = context.tool_output_dir(
                execution_id, datetime.now(timezone.utc).strftime("%Y%m%d")
            )
            target = output_dir / "annotated.jpg"
            annotated = _annotate(image, run.prediction.detections)
            try:
                await anyio.to_thread.run_sync(partial(
                    annotated.save, target, "JPEG", quality=92, optimize=True
                ))
            finally:
                annotated.close()
            data = run.prediction.model_dump(mode="json")
            return ToolResult(
                success=True,
                tool=self.name,
                data=data,
                artifacts=[Artifact(kind="image", path=str(target), mime_type="image/jpeg")],
                metadata={
                    "model": run.model,
                    "device": run.device,
                    "detector_load_time_s": run.load_time_s,
                    "detector_inference_ms": run.inference_ms,
                    "detector_load_count": context.detector_manager.load_count,
                    "gpu_allocated_gb": run.gpu.allocated_gb,
                    "gpu_reserved_gb": run.gpu.reserved_gb,
                    "gpu_peak_gb": run.gpu.peak_allocated_gb,
                },
            )
        finally:
            image.close()

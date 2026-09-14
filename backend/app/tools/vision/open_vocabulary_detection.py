from __future__ import annotations

from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import anyio
from PIL import ImageDraw
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.open_vocabulary.errors import OpenVocabularyLoadError
from backend.app.schemas.tool_result import Artifact, ToolResult
from backend.app.tools.base import BaseTool
from backend.app.tools.context import ToolContext
from backend.app.tools.image_io import open_supported_image


class DetectOpenVocabularyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_path: Path = Field(description="Workflow artifact ID (preferred) or a PNG, JPEG, or WEBP image reference.")
    classes: list[str] = Field(
        min_length=1,
        max_length=32,
        description=(
            "English text labels for the targets to locate, such as "
            "['yellow safety helmet', 'tower crane']. Labels are open vocabulary."
        ),
    )
    confidence: float = Field(default=0.25, ge=0.01, le=1)
    iou_threshold: float = Field(default=0.45, ge=0.01, le=1)

    @field_validator("classes")
    @classmethod
    def validate_classes(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value or len(value) > 100 for value in cleaned):
            raise ValueError("Each class must contain 1 to 100 characters")
        return cleaned


def _color(index: int) -> tuple[int, int, int]:
    return (48 + index * 67 % 192, 48 + index * 97 % 192, 48 + index * 137 % 192)


def _annotate(image, detections):
    annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    width = max(2, round(min(image.size) / 250))
    for index, item in enumerate(detections):
        box = item.bbox
        color = _color(index)
        draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline=color, width=width)
        label = f"{item.class_name} {item.confidence:.2f}"
        left, top, right, bottom = draw.textbbox((box.x1, box.y1), label)
        label_height = bottom - top + 4
        label_top = max(0, box.y1 - label_height)
        label_right = min(image.width, box.x1 + right - left + 6)
        draw.rectangle((box.x1, label_top, label_right, box.y1), fill=color)
        draw.text((box.x1 + 3, label_top + 2), label, fill="white")
    return annotated


class DetectOpenVocabularyTool(BaseTool):
    name = "detect_open_vocab"
    description = (
        "Locate arbitrary user-specified English text categories with YOLOE-26s. Returns "
        "detection IDs, exact pixel bounding boxes, confidence, counts, and an annotated image. "
        "Use for targets outside or more specific than COCO. Do not use it for precise masks."
    )
    category = "vision"
    input_schema = DetectOpenVocabularyInput
    requires_gpu = True
    requires_model = "YOLOE-26s-seg"

    async def execute(self, inputs, context: ToolContext, execution_id: str) -> ToolResult:
        if context.open_vocab_manager is None:
            raise OpenVocabularyLoadError("Open-vocabulary manager is not configured")
        path, image, _ = open_supported_image(context, inputs.image_path)
        try:
            load_count_before = context.open_vocab_manager.load_count
            run = await anyio.to_thread.run_sync(partial(
                context.open_vocab_manager.predict,
                path,
                classes=inputs.classes,
                confidence=inputs.confidence,
                iou_threshold=inputs.iou_threshold,
            ))
            output_dir = context.tool_output_dir(
                execution_id, datetime.now(timezone.utc).strftime("%Y%m%d")
            )
            target = output_dir / "open-vocab-annotated.jpg"
            annotated = _annotate(image, run.prediction.detections)
            try:
                await anyio.to_thread.run_sync(partial(
                    annotated.save, target, "JPEG", quality=92, optimize=True
                ))
            finally:
                annotated.close()
            data = run.prediction.model_dump(mode="json")
            data["source_image_path"] = str(path)
            return ToolResult(
                success=True,
                tool=self.name,
                data=data,
                artifacts=[Artifact(kind="image", path=str(target), mime_type="image/jpeg")],
                metadata={
                    "model": run.model,
                    "text_encoder": run.text_encoder,
                    "device": run.device,
                    "model_load_time_s": run.load_time_s,
                    "model_load_ms": (
                        (run.load_time_s or 0) * 1000
                        if context.open_vocab_manager.load_count > load_count_before else 0
                    ),
                    "effective_prompts": run.effective_prompts,
                    "prompt_encoding_ms": run.prompt_encoding_ms,
                    "inference_ms": run.inference_ms,
                    "model_load_count": context.open_vocab_manager.load_count,
                    "gpu_allocated_gb": run.gpu.allocated_gb,
                    "gpu_reserved_gb": run.gpu.reserved_gb,
                    "gpu_peak_gb": run.gpu.peak_allocated_gb,
                    "offline": True,
                },
            )
        finally:
            image.close()

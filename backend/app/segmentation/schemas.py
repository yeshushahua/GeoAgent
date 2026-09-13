from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.detection.schemas import BoundingBox
from backend.app.schemas.inference import GpuMemory


class SegmentInstance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    segment_id: str = Field(pattern=r"^segment-[0-9]{3}$")
    bbox: BoundingBox
    mask_area_pixels: int = Field(ge=0)
    mask_area_ratio: float = Field(ge=0, le=1)


class SegmentationPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    segment_count: int = Field(ge=0)
    segments: list[SegmentInstance] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent(self):
        if self.segment_count != len(self.segments):
            raise ValueError("segment_count must match segments")
        for item in self.segments:
            if item.bbox.x2 > self.image_width or item.bbox.y2 > self.image_height:
                raise ValueError("Bounding box exceeds image bounds")
        return self


@dataclass(frozen=True)
class SegmentationRun:
    model: str
    device: str
    load_time_s: float | None
    inference_ms: float
    prediction: SegmentationPrediction
    masks: list[np.ndarray]
    gpu: GpuMemory

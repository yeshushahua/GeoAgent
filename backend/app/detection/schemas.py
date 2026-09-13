from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.schemas.inference import GpuMemory


class BoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float = Field(gt=0)
    y2: float = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Bounding box must have positive width and height")
        return self


class Detection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    class_id: int = Field(ge=0)
    class_name: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    bbox: BoundingBox


class DetectionPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    detection_count: int = Field(ge=0)
    class_counts: dict[str, int] = Field(default_factory=dict)
    detections: list[Detection] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_count(self):
        if self.detection_count != len(self.detections):
            raise ValueError("detection_count must match detections")
        if sum(self.class_counts.values()) != self.detection_count:
            raise ValueError("class_counts must match detection_count")
        for item in self.detections:
            if item.bbox.x2 > self.image_width or item.bbox.y2 > self.image_height:
                raise ValueError("Bounding box exceeds image bounds")
        return self


class DetectorRun(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    device: str
    load_time_s: float | None = Field(default=None, ge=0)
    inference_ms: float = Field(ge=0)
    prediction: DetectionPrediction
    gpu: GpuMemory

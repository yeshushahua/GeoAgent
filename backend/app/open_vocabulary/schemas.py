from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.detection.schemas import BoundingBox
from backend.app.schemas.inference import GpuMemory


class OpenVocabularyDetection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    detection_id: str = Field(pattern=r"^detection-[0-9]{3}$")
    class_name: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    bbox: BoundingBox


class OpenVocabularyPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    requested_classes: list[str] = Field(min_length=1)
    detection_count: int = Field(ge=0)
    class_counts: dict[str, int] = Field(default_factory=dict)
    detections: list[OpenVocabularyDetection] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent(self):
        if self.detection_count != len(self.detections):
            raise ValueError("detection_count must match detections")
        if sum(self.class_counts.values()) != self.detection_count:
            raise ValueError("class_counts must match detection_count")
        for item in self.detections:
            if item.bbox.x2 > self.image_width or item.bbox.y2 > self.image_height:
                raise ValueError("Bounding box exceeds image bounds")
        return self


class OpenVocabularyRun(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    text_encoder: str
    device: str
    load_time_s: float | None = Field(default=None, ge=0)
    effective_prompts: list[str] = Field(default_factory=list)
    prompt_encoding_ms: float = Field(ge=0)
    inference_ms: float = Field(ge=0)
    prediction: OpenVocabularyPrediction
    gpu: GpuMemory

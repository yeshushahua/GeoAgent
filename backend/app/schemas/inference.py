from pydantic import BaseModel, ConfigDict, Field, model_validator


class ImageInfo(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    mode: str
    format: str
    preprocessing_width: int = Field(gt=0)
    preprocessing_height: int = Field(gt=0)
    resized: bool


class GenerationInfo(BaseModel):
    max_new_tokens: int = Field(ge=64, le=512)


class GpuMemory(BaseModel):
    allocated_gb: float = Field(ge=0)
    reserved_gb: float = Field(ge=0)
    peak_allocated_gb: float = Field(ge=0)


class InferenceError(BaseModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class InferenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    success: bool
    model: str
    text: str = ""
    latency_ms: float = Field(ge=0)
    device: str
    dtype: str
    image: ImageInfo
    generation: GenerationInfo
    gpu: GpuMemory
    error: InferenceError | None = None

    @model_validator(mode="after")
    def validate_outcome(self):
        if self.success and self.error is not None:
            raise ValueError("Successful inference cannot contain an error")
        if not self.success and self.error is None:
            raise ValueError("Failed inference requires an error")
        return self


class TextGenerationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)
    model: str
    latency_ms: float = Field(ge=0)
    device: str
    dtype: str
    gpu: GpuMemory
    max_new_tokens: int = Field(ge=64, le=512)


class ApiError(BaseModel):
    error: InferenceError

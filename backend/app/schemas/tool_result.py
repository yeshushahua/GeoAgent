from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class ToolError(BaseModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    details: dict[str, JsonValue] = Field(default_factory=dict)


class Artifact(BaseModel):
    kind: str = Field(min_length=1, description="image, mask, geojson, geotiff, json, chart, ...")
    path: str = Field(min_length=1)
    mime_type: str | None = None


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    success: bool
    tool: str = Field(min_length=1)
    data: dict[str, JsonValue] = Field(default_factory=dict)
    artifacts: list[Artifact] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    error: ToolError | None = None

    @model_validator(mode="after")
    def consistent_error(self):
        if self.success and self.error is not None:
            raise ValueError("Successful results must not contain an error")
        if not self.success and self.error is None:
            raise ValueError("Failed results must contain a structured error")
        return self

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from backend.app.detection.schemas import BoundingBox
from backend.app.schemas.tool_result import ToolError, ToolResult

if TYPE_CHECKING:
    from backend.app.agent.schemas import AgentToolCall


ArtifactType = Literal[
    "original_image", "cropped_image", "detection_overlay",
    "segmentation_overlay", "mask", "image",
]


class WorkflowArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_id: str = Field(pattern=r"^[a-z][a-z-]*-[0-9]{3}$")
    artifact_type: ArtifactType
    role: Literal["analysis_source", "visualization"]
    source_tool: str
    path: str
    parent_artifact_id: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    created_step: int = Field(ge=0)


class WorkflowArtifactView(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_id: str
    artifact_type: ArtifactType
    role: Literal["analysis_source", "visualization"]
    source_tool: str
    basename: str
    parent_artifact_id: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    created_step: int = Field(ge=0)


class WorkflowDetection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    detection_id: str = Field(pattern=r"^det-[0-9]{3}$")
    source_detection_id: str | None = None
    source_artifact_id: str
    class_name: str
    confidence: float = Field(ge=0, le=1)
    bbox: BoundingBox
    segmented: bool = False
    created_step: int = Field(ge=1)


class WorkflowSegmentation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    segmentation_id: str = Field(pattern=r"^seg-[0-9]{3}$")
    source_detection_id: str | None = None
    source_artifact_id: str
    bbox: BoundingBox
    success: bool
    mask_artifact_id: str | None = None
    overlay_artifact_id: str | None = None
    mask_area_pixels: int | None = Field(default=None, ge=0)
    mask_area_ratio: float | None = Field(default=None, ge=0, le=1)
    error_code: str | None = None
    created_step: int = Field(ge=1)


class WorkflowProgress(BaseModel):
    model_config = ConfigDict(extra="forbid")
    completed_actions: list[str] = Field(default_factory=list)
    failed_actions: list[str] = Field(default_factory=list)
    pending_goals: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class WorkflowDependencyError(RuntimeError):
    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


class WorkflowController:
    """Maintains workflow identity and dependencies without adding a workflow engine."""

    @staticmethod
    def initialize(state, image_path: str) -> None:
        state.workflow_artifacts.append(WorkflowArtifact(
            artifact_id="original-image-001",
            artifact_type="original_image",
            role="analysis_source",
            source_tool="user_upload",
            path=image_path,
            created_step=0,
        ))
        state.original_artifact_id = "original-image-001"
        state.active_image_artifact_id = "original-image-001"
        state.pending_goals = ["Complete every requested task before returning the final answer"]

    @staticmethod
    def artifact(state, artifact_id: str) -> WorkflowArtifact | None:
        return next(
            (item for item in state.workflow_artifacts if item.artifact_id == artifact_id), None
        )

    @classmethod
    def artifact_for_path(cls, state, path: str) -> WorkflowArtifact | None:
        target = Path(path).resolve()
        matches = [
            item for item in state.workflow_artifacts if Path(item.path).resolve() == target
        ]
        return matches[0] if len(matches) == 1 else None

    @classmethod
    def resolve_image(cls, state, reference: str) -> WorkflowArtifact:
        aliases = {
            "original_image": state.original_artifact_id,
            "original-image": state.original_artifact_id,
            "active_image": state.active_image_artifact_id,
            "active-image": state.active_image_artifact_id,
        }
        artifact_id = aliases.get(reference, reference)
        artifact = cls.artifact(state, artifact_id)
        if artifact is not None:
            if artifact.role != "analysis_source":
                raise WorkflowDependencyError(
                    "INVALID_ARTIFACT_ROLE",
                    f"Artifact {artifact_id} is display-only and cannot be a model input",
                    {"artifact_id": artifact_id, "role": artifact.role},
                )
            return artifact
        by_path = cls.artifact_for_path(state, reference)
        if by_path is not None:
            return by_path
        basename_matches = [
            item for item in state.workflow_artifacts
            if Path(item.path).name == reference and item.role == "analysis_source"
        ]
        if len(basename_matches) == 1:
            return basename_matches[0]
        raise WorkflowDependencyError(
            "ARTIFACT_NOT_FOUND",
            f"Workflow artifact does not exist: {reference}",
            {
                "requested": Path(reference).name,
                "available_artifact_ids": [
                    item.artifact_id for item in state.workflow_artifacts
                    if item.role == "analysis_source"
                ],
            },
        )

    @classmethod
    def prepare_call(cls, state, call: AgentToolCall) -> tuple[AgentToolCall, str | None]:
        arguments = dict(call.arguments)
        source_artifact_id = None
        if "image_path" in arguments and arguments["image_path"] is not None:
            source = cls.resolve_image(state, str(arguments["image_path"]))
            source_artifact_id = source.artifact_id
            arguments["image_path"] = source.path
            if call.tool_name == "crop_image" and not {
                "width", "height"
            }.issubset(source.metadata):
                raise WorkflowDependencyError(
                    "IMAGE_METADATA_REQUIRED",
                    "Inspect the source artifact before computing crop coordinates",
                    {"artifact_id": source.artifact_id, "required_tool": "inspect_image"},
                )
        if call.tool_name == "segment_objects" and arguments.get("detection_ids"):
            detection_ids = list(arguments["detection_ids"])
            detections = []
            missing = []
            for detection_id in detection_ids:
                detection = next(
                    (item for item in state.detections if item.detection_id == detection_id), None
                )
                if detection is None:
                    missing.append(detection_id)
                else:
                    detections.append(detection)
            if missing:
                raise WorkflowDependencyError(
                    "DETECTION_NOT_FOUND",
                    "One or more workflow detections do not exist",
                    {"missing_detection_ids": missing},
                )
            sources = {item.source_artifact_id for item in detections}
            if len(sources) != 1 or source_artifact_id not in sources:
                raise WorkflowDependencyError(
                    "DEPENDENCY_MISMATCH",
                    "Segmentation detections and image artifact must share one source",
                    {
                        "image_artifact_id": source_artifact_id,
                        "detection_source_artifact_ids": sorted(sources),
                    },
                )
            arguments["boxes"] = [item.bbox.model_dump(mode="json") for item in detections]
            arguments.pop("detection_ids", None)
        return call.model_copy(update={"arguments": arguments}), source_artifact_id

    @staticmethod
    def call_signature(
        original: AgentToolCall, prepared: AgentToolCall | None, source_artifact_id: str | None
    ) -> str:
        arguments = dict((prepared or original).arguments)
        if source_artifact_id:
            arguments["image_path"] = source_artifact_id
        classes = arguments.get("classes")
        if isinstance(classes, list):
            arguments["classes"] = sorted(str(item).strip().casefold() for item in classes)
        detection_ids = original.arguments.get("detection_ids")
        if isinstance(detection_ids, list):
            arguments["detection_ids"] = sorted(str(item) for item in detection_ids)
            arguments.pop("boxes", None)
        boxes = arguments.get("boxes")
        if isinstance(boxes, list):
            normalized = []
            for box in boxes:
                if isinstance(box, dict):
                    normalized.append(tuple(round(float(box[key]), 4) for key in ("x1", "y1", "x2", "y2")))
            arguments["boxes"] = sorted(normalized)
        return json.dumps(
            {"tool_name": original.tool_name, "arguments": arguments},
            sort_keys=True, ensure_ascii=False,
        )

    @staticmethod
    def dependency_failure(tool_name: str, error: WorkflowDependencyError) -> ToolResult:
        return ToolResult(
            success=False,
            tool=tool_name,
            error=ToolError(
                code=error.code, type=error.code, message=str(error), details=error.details
            ),
        )

    @staticmethod
    def _next_id(state, prefix: str) -> str:
        counters = {
            "crop": sum(item.artifact_type == "cropped_image" for item in state.workflow_artifacts),
            "detection-overlay": sum(item.artifact_type == "detection_overlay" for item in state.workflow_artifacts),
            "segmentation-overlay": sum(item.artifact_type == "segmentation_overlay" for item in state.workflow_artifacts),
            "mask": sum(item.artifact_type == "mask" for item in state.workflow_artifacts),
            "image": sum(item.artifact_type == "image" for item in state.workflow_artifacts),
        }
        return f"{prefix}-{counters.get(prefix, 0) + 1:03d}"

    @classmethod
    def _register_artifact(
        cls, state, artifact, artifact_type: ArtifactType, role: str,
        source_tool: str, parent_id: str | None, step: int, metadata: dict | None = None,
    ) -> WorkflowArtifact:
        prefix = {
            "cropped_image": "crop",
            "detection_overlay": "detection-overlay",
            "segmentation_overlay": "segmentation-overlay",
            "mask": "mask",
            "image": "image",
        }[artifact_type]
        record = WorkflowArtifact(
            artifact_id=cls._next_id(state, prefix),
            artifact_type=artifact_type,
            role=role,
            source_tool=source_tool,
            path=artifact.path,
            parent_artifact_id=parent_id,
            metadata=metadata or {},
            created_step=step,
        )
        state.workflow_artifacts.append(record)
        return record

    @classmethod
    def apply_result(
        cls, state, original_call: AgentToolCall, result: ToolResult,
        source_artifact_id: str | None, step: int,
    ) -> None:
        action = f"{original_call.tool_name}@{source_artifact_id or 'workflow'}"
        if not result.success:
            state.failed_actions.append(action)
            if result.error:
                state.warnings.append(f"{result.error.code}: {result.error.message}")
            return
        state.completed_actions.append(action)
        if original_call.tool_name == "inspect_image" and source_artifact_id:
            artifact = cls.artifact(state, source_artifact_id)
            if artifact:
                artifact.metadata.update({
                    key: result.data[key] for key in ("width", "height", "format", "mode")
                    if key in result.data
                })
        elif original_call.tool_name == "crop_image" and result.artifacts:
            crop = cls._register_artifact(
                state, result.artifacts[0], "cropped_image", "analysis_source",
                original_call.tool_name, source_artifact_id, step,
                {key: result.data[key] for key in ("width", "height") if key in result.data},
            )
            state.active_image_artifact_id = crop.artifact_id
        elif original_call.tool_name in {"detect_objects", "detect_open_vocab"}:
            if result.artifacts:
                cls._register_artifact(
                    state, result.artifacts[0], "detection_overlay", "visualization",
                    original_call.tool_name, source_artifact_id, step,
                )
            requested = result.data.get("requested_classes")
            if not requested:
                requested = list(result.data.get("class_counts", {}))
            for category in requested or []:
                if category not in state.requested_categories:
                    state.requested_categories.append(category)
            for item in result.data.get("detections", []):
                state.detections.append(WorkflowDetection(
                    detection_id=f"det-{len(state.detections) + 1:03d}",
                    source_detection_id=item.get("detection_id"),
                    source_artifact_id=source_artifact_id or state.active_image_artifact_id,
                    class_name=item["class_name"],
                    confidence=item["confidence"],
                    bbox=BoundingBox.model_validate(item["bbox"]),
                    created_step=step,
                ))
        elif original_call.tool_name == "segment_objects":
            masks_by_path = {}
            overlay_id = None
            for artifact in result.artifacts:
                if artifact.kind == "mask":
                    record = cls._register_artifact(
                        state, artifact, "mask", "visualization", original_call.tool_name,
                        source_artifact_id, step,
                    )
                    masks_by_path[str(Path(artifact.path).resolve())] = record.artifact_id
                else:
                    record = cls._register_artifact(
                        state, artifact, "segmentation_overlay", "visualization",
                        original_call.tool_name, source_artifact_id, step,
                    )
                    overlay_id = record.artifact_id
            detection_ids = original_call.arguments.get("detection_ids") or []
            for index, item in enumerate(result.data.get("segments", [])):
                input_index = int(item.get("input_index", index))
                detection_id = detection_ids[input_index] if input_index < len(detection_ids) else None
                if detection_id is None:
                    detection_id = cls._match_detection(state, source_artifact_id, item["bbox"])
                mask_id = masks_by_path.get(str(Path(item["mask_artifact_path"]).resolve()))
                state.segmentations.append(WorkflowSegmentation(
                    segmentation_id=f"seg-{len(state.segmentations) + 1:03d}",
                    source_detection_id=detection_id,
                    source_artifact_id=source_artifact_id or state.active_image_artifact_id,
                    bbox=BoundingBox.model_validate(item["bbox"]),
                    success=True,
                    mask_artifact_id=mask_id,
                    overlay_artifact_id=overlay_id,
                    mask_area_pixels=item["mask_area_pixels"],
                    mask_area_ratio=item["mask_area_ratio"],
                    created_step=step,
                ))
                for detection in state.detections:
                    if detection.detection_id == detection_id:
                        detection.segmented = True
            for failure in result.data.get("failures", []):
                input_index = int(failure.get("input_index", -1))
                detection_id = detection_ids[input_index] if 0 <= input_index < len(detection_ids) else None
                box = failure.get("bbox")
                if box:
                    state.segmentations.append(WorkflowSegmentation(
                        segmentation_id=f"seg-{len(state.segmentations) + 1:03d}",
                        source_detection_id=detection_id,
                        source_artifact_id=source_artifact_id or state.active_image_artifact_id,
                        bbox=BoundingBox.model_validate(box),
                        success=False,
                        error_code=failure.get("code", "SEGMENTATION_FAILED"),
                        created_step=step,
                    ))
                state.warnings.append(
                    f"Segmentation failed for {detection_id or f'input {input_index}'}: "
                    f"{failure.get('code', 'SEGMENTATION_FAILED')}"
                )

    @staticmethod
    def _match_detection(state, source_artifact_id: str | None, bbox: dict) -> str | None:
        target = tuple(round(float(bbox[key]), 2) for key in ("x1", "y1", "x2", "y2"))
        for item in state.detections:
            current = tuple(round(float(getattr(item.bbox, key)), 2) for key in ("x1", "y1", "x2", "y2"))
            if item.source_artifact_id == source_artifact_id and current == target:
                return item.detection_id
        return None

    @classmethod
    def public_artifacts(cls, state) -> list[WorkflowArtifactView]:
        return [WorkflowArtifactView(
            artifact_id=item.artifact_id,
            artifact_type=item.artifact_type,
            role=item.role,
            source_tool=item.source_tool,
            basename=Path(item.path).name,
            parent_artifact_id=item.parent_artifact_id,
            metadata=item.metadata,
            created_step=item.created_step,
        ) for item in state.workflow_artifacts]

    @classmethod
    def compact_observations(cls, state) -> list[dict]:
        compact = []
        for observation in state.observations[-state.max_steps:]:
            result = observation.result
            item: dict[str, Any] = {
                "step": observation.step,
                "tool": observation.tool_name,
                "success": result.success,
            }
            if not result.success:
                item["error"] = result.error.model_dump(mode="json") if result.error else None
            elif result.tool == "inspect_image":
                item["image"] = {key: result.data.get(key) for key in ("width", "height", "format", "mode")}
            elif result.tool == "crop_image":
                item["active_image_artifact_id"] = state.active_image_artifact_id
                item["size"] = {key: result.data.get(key) for key in ("width", "height")}
            elif result.tool in {"detect_objects", "detect_open_vocab"}:
                source_path = result.data.get("source_image_path")
                source = cls.artifact_for_path(state, source_path) if source_path else None
                records = [
                    det for det in state.detections
                    if source is not None and det.source_artifact_id == source.artifact_id
                    and det.created_step == observation.step
                ]
                item.update({
                    "source_artifact_id": source.artifact_id if source else None,
                    "class_counts": result.data.get("class_counts", {}),
                    "requested_classes": result.data.get("requested_classes", []),
                    "detection_ids": [det.detection_id for det in records],
                })
            elif result.tool == "segment_objects":
                item.update({
                    "segment_count": result.data.get("segment_count", 0),
                    "failed_count": len(result.data.get("failures", [])),
                    "segmentation_ids": [
                        seg.segmentation_id for seg in state.segmentations
                        if seg.created_step == observation.step
                    ],
                })
            elif result.tool == "analyze_image":
                answer = result.data.get("answer", "")
                item["answer"] = answer[:2000]
            compact.append(item)
        return compact

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from backend.app.agent.workflow import (
    WorkflowArtifactView,
    WorkflowController,
    WorkflowDetection,
    WorkflowProgress,
    WorkflowSegmentation,
)


class CategoryAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    detected: int = Field(ge=0)
    segmented: int = Field(ge=0)
    failed_segmentations: int = Field(ge=0)
    confidences: list[float] = Field(default_factory=list)
    instance_mask_area_pixels: list[int] = Field(default_factory=list)
    instance_area_sum_pixels: int = Field(default=0, ge=0)
    union_mask_area_pixels: int = Field(default=0, ge=0)
    union_mask_area_ratio: float = Field(default=0, ge=0, le=1)
    source_artifact_ids: list[str] = Field(default_factory=list)


class WorkflowSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    original_image_artifact_id: str
    active_image_artifact_id: str
    artifacts: list[WorkflowArtifactView] = Field(default_factory=list)
    detections: list[WorkflowDetection] = Field(default_factory=list)
    segmentations: list[WorkflowSegmentation] = Field(default_factory=list)
    categories: dict[str, CategoryAggregate] = Field(default_factory=dict)
    progress: WorkflowProgress
    successful_steps: int = Field(ge=0)
    failed_steps: int = Field(ge=0)


class ResultAggregator:
    @classmethod
    def aggregate(cls, state) -> WorkflowSummary:
        category_names = list(state.requested_categories)
        for detection in state.detections:
            if detection.class_name not in category_names:
                category_names.append(detection.class_name)
        segmentations_by_detection = defaultdict(list)
        for segmentation in state.segmentations:
            if segmentation.source_detection_id:
                segmentations_by_detection[segmentation.source_detection_id].append(segmentation)
        categories = {}
        for category in category_names:
            detections = [item for item in state.detections if item.class_name == category]
            successful = []
            failed = []
            for detection in detections:
                for segmentation in segmentations_by_detection[detection.detection_id]:
                    (successful if segmentation.success else failed).append(segmentation)
            instance_areas = [item.mask_area_pixels or 0 for item in successful]
            union_pixels, union_ratio = cls._union_area(state, successful)
            categories[category] = CategoryAggregate(
                detected=len(detections),
                segmented=len(successful),
                failed_segmentations=len(failed),
                confidences=[item.confidence for item in detections],
                instance_mask_area_pixels=instance_areas,
                instance_area_sum_pixels=sum(instance_areas),
                union_mask_area_pixels=union_pixels,
                union_mask_area_ratio=union_ratio,
                source_artifact_ids=sorted({item.source_artifact_id for item in detections}),
            )
        return WorkflowSummary(
            original_image_artifact_id=state.original_artifact_id,
            active_image_artifact_id=state.active_image_artifact_id,
            artifacts=WorkflowController.public_artifacts(state),
            detections=state.detections,
            segmentations=state.segmentations,
            categories=categories,
            progress=WorkflowProgress(
                completed_actions=state.completed_actions,
                failed_actions=state.failed_actions,
                pending_goals=state.pending_goals,
                warnings=state.warnings,
            ),
            successful_steps=sum(step.success is True for step in state.steps),
            failed_steps=sum(step.success is False for step in state.steps),
        )

    @staticmethod
    def _union_area(state, segmentations: list[WorkflowSegmentation]) -> tuple[int, float]:
        grouped: dict[str, list[np.ndarray]] = defaultdict(list)
        for segmentation in segmentations:
            if not segmentation.mask_artifact_id:
                continue
            artifact = WorkflowController.artifact(state, segmentation.mask_artifact_id)
            if artifact is None or not Path(artifact.path).is_file():
                continue
            with Image.open(artifact.path) as image:
                grouped[segmentation.source_artifact_id].append(
                    np.asarray(image.convert("L")) > 0
                )
        total_union = 0
        total_source_pixels = 0
        for source_id, masks in grouped.items():
            if not masks:
                continue
            union = np.logical_or.reduce(masks)
            total_union += int(union.sum())
            total_source_pixels += int(union.size)
        ratio = total_union / total_source_pixels if total_source_pixels else 0.0
        return total_union, round(ratio, 8)

    @staticmethod
    def render_chinese(summary: WorkflowSummary) -> str:
        if not summary.categories:
            return ""
        lines = ["结构化工作流结果："]
        has_masks = False
        for name, item in summary.categories.items():
            text = (
                f"- {name}：检测到 {item.detected} 个"
                if item.detected else f"- {name}：未检测到"
            )
            if item.segmented or item.failed_segmentations:
                text += f"，成功分割 {item.segmented} 个"
                if item.failed_segmentations:
                    text += f"，{item.failed_segmentations} 个分割失败"
            if item.confidences:
                text += "；检测置信度 " + ", ".join(f"{value:.3f}" for value in item.confidences)
            if item.segmented:
                has_masks = True
                text += f"；联合掩膜面积占当前来源图像 {item.union_mask_area_ratio:.2%}"
            lines.append(text + "。")
        if has_masks:
            lines.append("面积占比基于图像像素计算，不代表平方米、公顷或真实地理面积。")
        return "\n".join(lines)

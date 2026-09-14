import logging
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from backend.app.agent.aggregation import ResultAggregator
from backend.app.agent.prompts import build_planner_prompt
from backend.app.agent.schemas import AgentRequest, AgentToolCall
from backend.app.agent.service import VisionAgent
from backend.app.agent.state import AgentState
from backend.app.agent.trace import AgentTraceStore
from backend.app.agent.workflow import (
    WorkflowArtifact,
    WorkflowController,
    WorkflowDependencyError,
    WorkflowDetection,
    WorkflowSegmentation,
)
from backend.app.detection.schemas import BoundingBox
from backend.app.open_vocabulary.schemas import OpenVocabularyDetection
from backend.app.schemas.tool_result import Artifact, ToolResult
from backend.app.segmentation.errors import SegmentationInferenceError
from backend.app.tools import build_tool_system
from backend.app.tools.context import ToolContext
from backend.app.tools.executor import ToolExecutor
from backend.app.tools.registry import ToolRegistry
from backend.app.tools.trace import ToolTraceStore
from backend.app.tools.vision.object_segmentation import SegmentObjectsTool
from backend.tests.test_agent_service import FakeAgentManager, ScriptedPlanner, tool_call
from backend.tests.test_phase5_tools import (
    FakeOpenVocabularyManager,
    FakeSegmentationManager,
)


def _source(settings, size=(100, 60)):
    path = settings.project_root / "phase6.png"
    Image.new("RGB", size, "gold").save(path)
    return path


def _state(path: Path):
    state = AgentState(
        run_id="run-phase6",
        user_message="workflow",
        original_image_path=str(path),
        tool_definitions=[],
        max_steps=8,
        max_new_tokens=128,
        started_at="2026-01-01T00:00:00+00:00",
    )
    WorkflowController.initialize(state, str(path))
    return state


def _agent(settings, planner, open_vocab=None, segmentation=None):
    manager = FakeAgentManager(settings)
    registry, executor, _ = build_tool_system(
        settings, manager, logging.getLogger("phase6-test"),
        open_vocab_manager=open_vocab,
        segmentation_manager=segmentation,
    )
    return VisionAgent(
        registry, executor, manager, AgentTraceStore(50), logging.getLogger("phase6-test"),
        repair_attempts=1, planner=planner,
    )


def test_artifact_registration_parent_and_active_image(settings):
    source = _source(settings)
    crop_path = settings.output_dir / "crop.png"
    crop_path.parent.mkdir(parents=True)
    Image.new("RGB", (50, 60), "gold").save(crop_path)
    state = _state(source)
    call = AgentToolCall(
        type="tool_call", tool_name="crop_image",
        arguments={"image_path": "original-image-001", "x1": 0, "y1": 0, "x2": 50, "y2": 60},
    )
    WorkflowController.apply_result(
        state, call,
        ToolResult(
            success=True, tool="crop_image", data={"width": 50, "height": 60},
            artifacts=[Artifact(kind="image", path=str(crop_path), mime_type="image/png")],
        ),
        "original-image-001", 1,
    )
    crop = WorkflowController.artifact(state, "crop-001")
    assert crop.parent_artifact_id == "original-image-001"
    assert crop.role == "analysis_source"
    assert state.active_image_artifact_id == "crop-001"

    overlay_path = settings.output_dir / "detected.jpg"
    overlay_path.write_bytes(b"preview")
    WorkflowController.apply_result(
        state,
        AgentToolCall(type="tool_call", tool_name="detect_open_vocab", arguments={}),
        ToolResult(
            success=True, tool="detect_open_vocab",
            data={"requested_classes": ["helmet"], "class_counts": {}, "detections": []},
            artifacts=[Artifact(kind="image", path=str(overlay_path), mime_type="image/jpeg")],
        ),
        "crop-001", 2,
    )
    assert state.active_image_artifact_id == "crop-001"
    assert WorkflowController.artifact(state, "detection-overlay-001").role == "visualization"


@pytest.mark.anyio
async def test_multiclass_batch_linking_zero_category_and_aggregation(settings):
    source = _source(settings)
    detections = [
        OpenVocabularyDetection(
            detection_id="detection-001", class_name="yellow helmet", confidence=0.91,
            bbox=BoundingBox(x1=5, y1=4, x2=25, y2=30),
        ),
        OpenVocabularyDetection(
            detection_id="detection-002", class_name="reflective safety vest", confidence=0.84,
            bbox=BoundingBox(x1=40, y1=8, x2=75, y2=50),
        ),
    ]
    open_vocab = FakeOpenVocabularyManager(detections)
    segmentation = FakeSegmentationManager()

    def segment_all(state):
        return tool_call("segment_objects", {
            "image_path": state.active_image_artifact_id,
            "detection_ids": [item.detection_id for item in state.detections],
        })

    planner = ScriptedPlanner([
        tool_call("detect_open_vocab", {
            "image_path": "active_image",
            "classes": ["yellow helmet", "reflective safety vest", "excavator"],
        }),
        segment_all,
        '{"type":"final","answer":"已完成检测和分割。"}',
    ])
    response = await _agent(settings, planner, open_vocab, segmentation).run(AgentRequest(
        message="找到三类目标并分割", image_path=str(source), max_steps=6
    ))
    assert response.success
    assert open_vocab.calls[0][1] == [
        "yellow helmet", "reflective safety vest", "excavator"
    ]
    assert len(open_vocab.calls) == len(segmentation.calls) == 1
    assert segmentation.calls[0][0] == source
    assert len(segmentation.calls[0][1]) == 2
    assert [item.source_detection_id for item in response.workflow.segmentations] == [
        "det-001", "det-002"
    ]
    assert response.workflow.categories["yellow helmet"].detected == 1
    assert response.workflow.categories["reflective safety vest"].segmented == 1
    assert response.workflow.categories["excavator"].detected == 0
    assert response.workflow.categories["excavator"].segmented == 0
    segment_step = next(step for step in response.steps if step.tool_name == "segment_objects")
    assert segment_step.arguments_summary["image_path"] == "original-image-001"
    assert segment_step.arguments_summary["detection_ids"] == ["det-001", "det-002"]
    assert "boxes" not in segment_step.arguments_summary
    assert "excavator" in response.answer


@pytest.mark.anyio
async def test_inspect_crop_detect_segment_keeps_one_analysis_source(settings):
    source = _source(settings, (100, 60))
    open_vocab = FakeOpenVocabularyManager([
        OpenVocabularyDetection(
            detection_id="detection-001", class_name="yellow helmet", confidence=0.88,
            bbox=BoundingBox(x1=5, y1=4, x2=25, y2=30),
        )
    ])
    segmentation = FakeSegmentationManager()

    def segment_latest(state):
        return tool_call("segment_objects", {
            "image_path": "active_image",
            "detection_ids": [item.detection_id for item in state.detections],
        })

    planner = ScriptedPlanner([
        tool_call("inspect_image", {"image_path": "original_image"}),
        tool_call("crop_image", {
            "image_path": "original-image-001", "x1": 0, "y1": 0, "x2": 50, "y2": 60,
        }),
        tool_call("detect_open_vocab", {
            "image_path": "active_image", "classes": ["yellow helmet"],
        }),
        segment_latest,
        '{"type":"final","answer":"左半部分处理完成。"}',
    ])
    response = await _agent(settings, planner, open_vocab, segmentation).run(AgentRequest(
        message="检查、裁剪左半部分、检测并分割", image_path=str(source), max_steps=6
    ))
    sequence = [step.tool_name for step in response.steps if step.tool_name]
    assert response.success
    assert sequence == [
        "inspect_image", "crop_image", "detect_open_vocab", "segment_objects"
    ]
    assert Path(open_vocab.calls[0][0]) == Path(segmentation.calls[0][0])
    assert response.workflow.active_image_artifact_id == "crop-001"
    detect_step = next(step for step in response.steps if step.tool_name == "detect_open_vocab")
    segment_step = next(step for step in response.steps if step.tool_name == "segment_objects")
    assert detect_step.arguments_summary["image_path"] == "crop-001"
    assert segment_step.arguments_summary["image_path"] == "crop-001"
    assert response.workflow.detections[0].source_artifact_id == "crop-001"
    assert response.workflow.segmentations[0].source_artifact_id == "crop-001"
    assert all("path" not in item.model_dump() for item in response.workflow.artifacts)


@pytest.mark.anyio
async def test_invalid_artifact_becomes_observation_and_agent_recovers(settings):
    source = _source(settings)
    planner = ScriptedPlanner([
        tool_call("inspect_image", {"image_path": "crop-999"}),
        tool_call("inspect_image", {"image_path": "original-image-001"}),
        '{"type":"final","answer":"图片尺寸已确认。"}',
    ])
    response = await _agent(settings, planner).run(AgentRequest(
        message="检查图片", image_path=str(source), max_steps=4
    ))
    assert response.success
    assert response.steps[0].success is False
    assert response.steps[0].error_type == "ARTIFACT_NOT_FOUND"
    assert response.steps[0].observation_summary["error"]["code"] == "ARTIFACT_NOT_FOUND"
    assert response.steps[1].success is True
    assert response.metadata.failed_steps == 1
    assert any("ARTIFACT_NOT_FOUND" in warning for warning in response.workflow.progress.warnings)


def test_duplicate_signature_normalizes_classes_boxes_and_source(settings):
    source = _source(settings)
    state = _state(source)
    first = AgentToolCall(type="tool_call", tool_name="detect_open_vocab", arguments={
        "image_path": "original-image-001", "classes": ["Helmet", " vest "],
    })
    second = AgentToolCall(type="tool_call", tool_name="detect_open_vocab", arguments={
        "image_path": str(source), "classes": ["VEST", "helmet"],
    })
    prepared1, source1 = WorkflowController.prepare_call(state, first)
    prepared2, source2 = WorkflowController.prepare_call(state, second)
    assert WorkflowController.call_signature(first, prepared1, source1) == (
        WorkflowController.call_signature(second, prepared2, source2)
    )
    crop_path = settings.project_root / "another.png"
    Image.new("RGB", (50, 60)).save(crop_path)
    state.workflow_artifacts.append(WorkflowArtifact(
        artifact_id="crop-001", artifact_type="cropped_image", role="analysis_source",
        source_tool="crop_image", path=str(crop_path),
        parent_artifact_id="original-image-001", created_step=1,
    ))
    third = first.model_copy(update={
        "arguments": {"image_path": "crop-001", "classes": ["helmet", "vest"]}
    })
    prepared3, source3 = WorkflowController.prepare_call(state, third)
    assert WorkflowController.call_signature(first, prepared1, source1) != (
        WorkflowController.call_signature(third, prepared3, source3)
    )


def test_crop_requires_source_metadata_before_coordinates(settings):
    source = _source(settings)
    state = _state(source)
    call = AgentToolCall(type="tool_call", tool_name="crop_image", arguments={
        "image_path": "original-image-001", "x1": 0, "y1": 0, "x2": 50, "y2": 30,
    })
    with pytest.raises(WorkflowDependencyError) as raised:
        WorkflowController.prepare_call(state, call)
    assert getattr(raised.value, "code", None) == "IMAGE_METADATA_REQUIRED"
    WorkflowController.artifact(state, "original-image-001").metadata.update({
        "width": 100, "height": 60,
    })
    prepared, source_id = WorkflowController.prepare_call(state, call)
    assert source_id == "original-image-001"
    assert prepared.arguments["image_path"] == str(source)


@pytest.mark.anyio
async def test_same_crop_can_retry_after_metadata_dependency_is_filled(settings):
    source = _source(settings, (100, 60))
    crop = tool_call("crop_image", {
        "image_path": "original-image-001", "x1": 0, "y1": 0, "x2": 50, "y2": 30,
    })
    planner = ScriptedPlanner([
        crop,
        tool_call("inspect_image", {"image_path": "original-image-001"}),
        crop,
        '{"type":"final","answer":"依赖补齐后裁剪完成。"}',
    ])
    response = await _agent(settings, planner).run(AgentRequest(
        message="裁剪图像区域", image_path=str(source), max_steps=5
    ))
    assert response.success
    assert response.steps[0].error_type == "IMAGE_METADATA_REQUIRED"
    assert response.steps[1].tool_name == "inspect_image" and response.steps[1].success
    assert response.steps[2].tool_name == "crop_image" and response.steps[2].success
    assert response.workflow.active_image_artifact_id == "crop-001"
    assert all(step.error_type != "DUPLICATE_TOOL_CALL" for step in response.steps)


def test_result_aggregator_uses_union_mask_area(settings):
    source = _source(settings, (4, 4))
    state = _state(source)
    boxes = [BoundingBox(x1=0, y1=0, x2=2, y2=2), BoundingBox(x1=1, y1=0, x2=3, y2=2)]
    for index, box in enumerate(boxes, start=1):
        state.detections.append(WorkflowDetection(
            detection_id=f"det-{index:03d}", source_artifact_id="original-image-001",
            class_name="helmet", confidence=0.9 - index / 10, bbox=box,
            segmented=True, created_step=1,
        ))
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[0:2, index - 1:index + 1] = 255
        mask_path = settings.output_dir / f"mask-{index}.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(mask_path)
        state.workflow_artifacts.append(WorkflowArtifact(
            artifact_id=f"mask-{index:03d}", artifact_type="mask", role="visualization",
            source_tool="segment_objects", path=str(mask_path),
            parent_artifact_id="original-image-001", created_step=2,
        ))
        state.segmentations.append(WorkflowSegmentation(
            segmentation_id=f"seg-{index:03d}", source_detection_id=f"det-{index:03d}",
            source_artifact_id="original-image-001", bbox=box, success=True,
            mask_artifact_id=f"mask-{index:03d}", mask_area_pixels=4,
            mask_area_ratio=0.25, created_step=2,
        ))
    state.requested_categories = ["helmet"]
    category = ResultAggregator.aggregate(state).categories["helmet"]
    assert category.instance_area_sum_pixels == 8
    assert category.union_mask_area_pixels == 6
    assert category.union_mask_area_ratio == 0.375


class PartialFailureSegmentationManager(FakeSegmentationManager):
    def predict(self, path, boxes):
        if len(boxes) > 1:
            raise SegmentationInferenceError("batch failed")
        if boxes[0].x1 >= 40:
            raise SegmentationInferenceError("one bbox failed")
        return super().predict(path, boxes)


@pytest.mark.anyio
async def test_segment_tool_recovers_other_instances_after_one_failure(settings):
    source = _source(settings)
    manager = PartialFailureSegmentationManager()
    registry = ToolRegistry()
    registry.register(SegmentObjectsTool())
    context = ToolContext(
        settings=settings, model_manager=FakeAgentManager(settings),
        logger=logging.getLogger("phase6-partial"), trace_store=ToolTraceStore(20),
        segmentation_manager=manager,
    )
    result = await ToolExecutor(registry, context).execute("segment_objects", {
        "image_path": str(source),
        "boxes": [
            {"x1": 5, "y1": 4, "x2": 25, "y2": 30},
            {"x1": 40, "y1": 8, "x2": 75, "y2": 50},
        ],
    })
    assert result.success
    assert result.data["segment_count"] == 1
    assert result.data["partial_failure"] is True
    assert result.data["segments"][0]["input_index"] == 0
    assert result.data["failures"][0]["input_index"] == 1
    assert result.data["failures"][0]["code"] == "SEGMENTATION_INFERENCE_FAILED"


@pytest.mark.anyio
async def test_agent_keeps_detection_when_one_segmentation_fails(settings):
    source = _source(settings)
    detections = [
        OpenVocabularyDetection(
            detection_id="detection-001", class_name="helmet", confidence=0.9,
            bbox=BoundingBox(x1=5, y1=4, x2=25, y2=30),
        ),
        OpenVocabularyDetection(
            detection_id="detection-002", class_name="vest", confidence=0.8,
            bbox=BoundingBox(x1=40, y1=8, x2=75, y2=50),
        ),
    ]
    open_vocab = FakeOpenVocabularyManager(detections)
    segmentation = PartialFailureSegmentationManager()

    def segment_all(state):
        return tool_call("segment_objects", {
            "image_path": "active_image",
            "detection_ids": [item.detection_id for item in state.detections],
        })

    planner = ScriptedPlanner([
        tool_call("detect_open_vocab", {
            "image_path": "active_image", "classes": ["helmet", "vest"],
        }),
        segment_all,
        '{"type":"final","answer":"已完成，部分目标分割失败。"}',
    ])
    response = await _agent(settings, planner, open_vocab, segmentation).run(AgentRequest(
        message="检测并分割", image_path=str(source), max_steps=5
    ))
    assert response.success
    assert response.workflow.categories["helmet"].segmented == 1
    assert response.workflow.categories["vest"].detected == 1
    assert response.workflow.categories["vest"].segmented == 0
    assert response.workflow.categories["vest"].failed_segmentations == 1
    assert any("det-002" in warning for warning in response.workflow.progress.warnings)
    assert "分割失败" in response.answer


@pytest.mark.anyio
async def test_planner_receives_compact_ids_without_absolute_paths(settings):
    source = _source(settings)
    open_vocab = FakeOpenVocabularyManager([
        OpenVocabularyDetection(
            detection_id="detection-001", class_name="helmet", confidence=0.8,
            bbox=BoundingBox(x1=5, y1=4, x2=25, y2=30),
        )
    ])
    planner = ScriptedPlanner([
        tool_call("detect_open_vocab", {
            "image_path": "original-image-001", "classes": ["helmet"]
        }),
        '{"type":"final","answer":"完成。"}',
    ])
    agent = _agent(settings, planner, open_vocab)
    response = await agent.run(AgentRequest(message="检测", image_path=str(source)))
    assert response.success
    prompt = build_planner_prompt(
        AgentState.model_validate({
            **_state(source).model_dump(),
            "observations": [],
        })
    )
    assert str(source) not in prompt
    assert "original-image-001" in prompt
    assert response.workflow.detections[0].detection_id == "det-001"

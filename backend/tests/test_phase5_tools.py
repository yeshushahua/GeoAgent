import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from backend.app.detection.schemas import BoundingBox
from backend.app.open_vocabulary.schemas import (
    OpenVocabularyDetection,
    OpenVocabularyPrediction,
    OpenVocabularyRun,
)
from backend.app.schemas.inference import GpuMemory
from backend.app.segmentation.schemas import SegmentationPrediction, SegmentationRun, SegmentInstance
from backend.app.tools.context import ToolContext
from backend.app.tools.executor import ToolExecutor
from backend.app.tools.registry import ToolRegistry
from backend.app.tools.trace import ToolTraceStore
from backend.app.tools.vision.object_segmentation import SegmentObjectsInput, SegmentObjectsTool
from backend.app.tools.vision.open_vocabulary_detection import (
    DetectOpenVocabularyInput,
    DetectOpenVocabularyTool,
)
from backend.tests.test_tool_system import FakeManager


class FakeOpenVocabularyManager:
    def __init__(self, detections=None):
        self.detections = detections or []
        self.calls = []
        self.load_count = 1

    def predict(self, path, classes, confidence=0.25, iou_threshold=0.45):
        self.calls.append((Path(path), classes, confidence, iou_threshold))
        with Image.open(path) as image:
            width, height = image.size
        counts = {}
        for item in self.detections:
            counts[item.class_name] = counts.get(item.class_name, 0) + 1
        return OpenVocabularyRun(
            model="yoloe-26s-seg", text_encoder="mobileclip2_b.ts", device="cuda:0",
            load_time_s=0.3, prompt_encoding_ms=12.0, inference_ms=8.0,
            prediction=OpenVocabularyPrediction(
                image_width=width, image_height=height, requested_classes=classes,
                detection_count=len(self.detections), class_counts=counts,
                detections=self.detections,
            ),
            gpu=GpuMemory(allocated_gb=0.5, reserved_gb=0.6, peak_allocated_gb=0.8),
        )


class FakeSegmentationManager:
    def __init__(self):
        self.calls = []
        self.load_count = 1

    def predict(self, path, boxes):
        self.calls.append((Path(path), boxes))
        with Image.open(path) as image:
            width, height = image.size
        masks = []
        segments = []
        for index, box in enumerate(boxes, start=1):
            mask = np.zeros((height, width), dtype=bool)
            left, top = int(box.x1), int(box.y1)
            right, bottom = int(box.x2), int(box.y2)
            mask[top:bottom, left:right] = True
            area = int(mask.sum())
            masks.append(mask)
            segments.append(SegmentInstance(
                segment_id=f"segment-{index:03d}", bbox=box,
                mask_area_pixels=area, mask_area_ratio=area / (width * height),
            ))
        return SegmentationRun(
            model="sam2.1_b", device="cuda:0", load_time_s=0.2, inference_ms=20.0,
            prediction=SegmentationPrediction(
                image_width=width, image_height=height,
                segment_count=len(segments), segments=segments,
            ),
            masks=masks,
            gpu=GpuMemory(allocated_gb=0.7, reserved_gb=0.8, peak_allocated_gb=1.0),
        )


def source(settings, size=(100, 60)):
    path = settings.project_root / "phase5.png"
    Image.new("RGB", size, "gold").save(path)
    return path


def context(settings, open_vocab=None, segmentation=None):
    return ToolContext(
        settings=settings,
        model_manager=FakeManager(),
        logger=logging.getLogger("test"),
        trace_store=ToolTraceStore(50),
        open_vocab_manager=open_vocab,
        segmentation_manager=segmentation,
    )


def test_phase5_tool_schemas_accept_free_text_and_require_boxes():
    open_schema = DetectOpenVocabularyTool().definition()["input_schema"]
    assert open_schema["required"] == ["image_path", "classes"]
    assert open_schema["properties"]["confidence"]["default"] == 0.25
    assert DetectOpenVocabularyInput.model_validate({
        "image_path": "x.png", "classes": ["yellow safety helmet", "tower crane"]
    }).classes == ["yellow safety helmet", "tower crane"]
    with pytest.raises(ValueError):
        DetectOpenVocabularyInput.model_validate({"image_path": "x.png", "classes": [" "]})
    segment_schema = SegmentObjectsTool().definition()["input_schema"]
    assert segment_schema["required"] == ["image_path", "boxes"]
    with pytest.raises(ValueError):
        SegmentObjectsInput.model_validate({"image_path": "x.png", "boxes": []})


@pytest.mark.anyio
async def test_open_vocab_tool_structured_output_artifact_and_zero_result(settings):
    image_path = source(settings)
    manager = FakeOpenVocabularyManager([
        OpenVocabularyDetection(
            detection_id="detection-001", class_name="yellow safety helmet",
            confidence=0.91, bbox=BoundingBox(x1=5, y1=4, x2=45, y2=40),
        )
    ])
    registry = ToolRegistry()
    registry.register(DetectOpenVocabularyTool())
    tool_context = context(settings, open_vocab=manager)
    executor = ToolExecutor(registry, tool_context)
    result = await executor.execute("detect_open_vocab", {
        "image_path": str(image_path), "classes": ["yellow safety helmet"],
        "confidence": 0.3, "iou_threshold": 0.5,
    })
    assert result.success and result.data["detection_count"] == 1
    assert result.data["detections"][0]["detection_id"] == "detection-001"
    assert manager.calls[0][1:] == (["yellow safety helmet"], 0.3, 0.5)
    assert Path(result.artifacts[0].path).name == "open-vocab-annotated.jpg"
    assert result.metadata["model_load_ms"] == 0
    assert result.metadata["inference_ms"] == 8.0
    assert result.metadata["tool_overhead_ms"] >= 0
    assert "base64" not in json.dumps(result.model_dump())
    manager.detections = []
    empty = await executor.execute("detect_open_vocab", {
        "image_path": str(image_path), "classes": ["nonexistent custom target"],
    })
    assert empty.success and empty.data["detection_count"] == 0


@pytest.mark.anyio
async def test_segment_tool_writes_per_instance_masks_overlay_and_area(settings):
    image_path = source(settings)
    manager = FakeSegmentationManager()
    registry = ToolRegistry()
    registry.register(SegmentObjectsTool())
    tool_context = context(settings, segmentation=manager)
    result = await ToolExecutor(registry, tool_context).execute("segment_objects", {
        "image_path": str(image_path),
        "boxes": [
            {"x1": 5, "y1": 4, "x2": 45, "y2": 40},
            {"x1": 50, "y1": 10, "x2": 90, "y2": 50},
        ],
    })
    assert result.success and result.data["segment_count"] == 2
    assert len(result.artifacts) == 3
    assert [item.kind for item in result.artifacts] == ["mask", "mask", "image"]
    assert Path(result.artifacts[-1].path).name == "segmentation-overlay.png"
    for item in result.data["segments"]:
        assert item["mask_area_pixels"] > 0
        assert 0 < item["mask_area_ratio"] <= 1
        assert Path(item["mask_artifact_path"]).is_file()
    with Image.open(result.artifacts[0].path) as mask:
        assert mask.mode == "L" and mask.size == (100, 60)
    assert manager.calls[0][1][0] == BoundingBox(x1=5, y1=4, x2=45, y2=40)
    assert "base64" not in json.dumps(result.model_dump())


@pytest.mark.anyio
async def test_phase5_tools_missing_image_fails_safely(settings):
    missing = str(settings.project_root / "missing.png")
    for tool, manager, arguments in (
        (DetectOpenVocabularyTool(), FakeOpenVocabularyManager(), {
            "image_path": missing, "classes": ["custom target"],
        }),
        (SegmentObjectsTool(), FakeSegmentationManager(), {
            "image_path": missing, "boxes": [{"x1": 1, "y1": 1, "x2": 2, "y2": 2}],
        }),
    ):
        registry = ToolRegistry()
        registry.register(tool)
        ctx = context(
            settings,
            open_vocab=manager if tool.name == "detect_open_vocab" else None,
            segmentation=manager if tool.name == "segment_objects" else None,
        )
        result = await ToolExecutor(registry, ctx).execute(tool.name, arguments)
        assert not result.success and result.error.code == "INVALID_IMAGE"

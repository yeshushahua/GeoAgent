import json
import logging
from pathlib import Path

from PIL import Image
import pytest

from backend.app.detection.schemas import BoundingBox, Detection, DetectionPrediction, DetectorRun
from backend.app.schemas.inference import GpuMemory
from backend.app.tools.context import ToolContext
from backend.app.tools.executor import ToolExecutor
from backend.app.tools.registry import ToolRegistry
from backend.app.tools.trace import ToolTraceStore
from backend.app.tools.vision.object_detection import DetectObjectsInput, DetectObjectsTool
from backend.tests.test_tool_system import FakeManager


class FakeDetector:
    def __init__(self, detections=None):
        self.detections = detections or []
        self.calls = []
        self.load_count = 1

    def predict(self, path, confidence=0.25, iou_threshold=0.45, classes=None):
        self.calls.append((path, confidence, iou_threshold, classes))
        with Image.open(path) as opened:
            image_width, image_height = opened.size
        counts = {}
        for item in self.detections:
            counts[item.class_name] = counts.get(item.class_name, 0) + 1
        prediction = DetectionPrediction(
            image_width=image_width,
            image_height=image_height,
            detection_count=len(self.detections),
            class_counts=counts,
            detections=self.detections,
        )
        return DetectorRun(
            model="yolo11s", device="cuda:0", load_time_s=0.2, inference_ms=4.5,
            prediction=prediction,
            gpu=GpuMemory(allocated_gb=0.3, reserved_gb=0.4, peak_allocated_gb=0.5),
        )


def context(settings, detector):
    return ToolContext(
        settings, FakeManager(), logging.getLogger("test"), ToolTraceStore(50), detector
    )


def source(settings):
    path = settings.project_root / "detect.png"
    Image.new("RGB", (100, 60), "white").save(path)
    return path


def test_detect_objects_schema_defaults_and_bounds():
    schema = DetectObjectsTool().definition()["input_schema"]
    assert schema["properties"]["confidence"]["default"] == 0.25
    assert schema["properties"]["iou_threshold"]["default"] == 0.45
    assert "classes" not in schema["required"]
    parsed = DetectObjectsInput.model_validate({"image_path": "x.png", "classes": [0, "bus"]})
    assert parsed.classes == [0, "bus"]
    with pytest.raises(ValueError):
        DetectObjectsInput.model_validate({"image_path": "x.png", "confidence": 1.1})


@pytest.mark.anyio
async def test_detect_objects_structured_output_artifact_and_safe_trace(settings):
    image_path = source(settings)
    detector = FakeDetector([
        Detection(
            class_id=0, class_name="person", confidence=0.91,
            bbox=BoundingBox(x1=5, y1=4, x2=45, y2=55),
        ),
        Detection(
            class_id=5, class_name="bus", confidence=0.82,
            bbox=BoundingBox(x1=40, y1=8, x2=98, y2=58),
        ),
    ])
    registry = ToolRegistry()
    registry.register(DetectObjectsTool())
    tool_context = context(settings, detector)
    result = await ToolExecutor(registry, tool_context).execute(
        "detect_objects",
        {
            "image_path": str(image_path), "confidence": 0.4,
            "iou_threshold": 0.5, "classes": ["person", "bus"],
        },
    )
    assert result.success and result.data["detection_count"] == 2
    assert result.data["class_counts"] == {"person": 1, "bus": 1}
    assert all(0 <= item["confidence"] <= 1 for item in result.data["detections"])
    artifact = Path(result.artifacts[0].path)
    assert artifact.is_file() and artifact.name == "annotated.jpg"
    with Image.open(artifact) as annotated:
        assert annotated.size == (100, 60)
    assert detector.calls[0][1:] == (0.4, 0.5, ["person", "bus"])
    trace = tool_context.trace_store.list()[0].model_dump()
    assert "base64" not in json.dumps(result.model_dump())
    assert "base64" not in json.dumps(trace)


@pytest.mark.anyio
async def test_detect_objects_empty_and_missing_image(settings):
    registry = ToolRegistry()
    registry.register(DetectObjectsTool())
    detector = FakeDetector()
    executor = ToolExecutor(registry, context(settings, detector))
    empty = await executor.execute("detect_objects", {"image_path": str(source(settings))})
    assert empty.success and empty.data["detection_count"] == 0
    assert Path(empty.artifacts[0].path).is_file()
    missing = await executor.execute(
        "detect_objects", {"image_path": str(settings.project_root / "missing.jpg")}
    )
    assert not missing.success and missing.error.code == "INVALID_IMAGE"

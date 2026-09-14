import io
import json
import logging

from PIL import Image

from backend.app.detection.schemas import BoundingBox
from backend.app.open_vocabulary.schemas import OpenVocabularyDetection
from backend.app.tools import build_tool_system
from backend.tests.test_phase5_tools import (
    FakeOpenVocabularyManager,
    FakeSegmentationManager,
)


def image_bytes(size=(100, 60)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "gold").save(buffer, "PNG")
    return buffer.getvalue()


def install(client):
    open_vocab = FakeOpenVocabularyManager([
        OpenVocabularyDetection(
            detection_id="detection-001", class_name="yellow safety helmet",
            confidence=0.91, bbox=BoundingBox(x1=5, y1=4, x2=45, y2=40),
        )
    ])
    segmentation = FakeSegmentationManager()
    registry, executor, traces = build_tool_system(
        client.app.state.settings,
        client.app.state.model_manager,
        logging.getLogger("test"),
        client.app.state.detector_manager,
        open_vocab,
        segmentation,
    )
    client.app.state.open_vocab_manager = open_vocab
    client.app.state.segmentation_manager = segmentation
    client.app.state.tool_registry = registry
    client.app.state.tool_executor = executor
    client.app.state.tool_traces = traces
    return open_vocab, segmentation


def test_phase5_status_endpoints(client):
    assert client.get("/api/v1/models/open-vocabulary/status").status_code == 200
    assert client.get("/api/v1/models/open-vocabulary/status").json()["model"] == "yoloe-26s-seg"
    assert client.get("/api/v1/models/segmentation/status").status_code == 200
    assert client.get("/api/v1/models/segmentation/status").json()["model"] == "sam2.1_b"


def test_phase5_tool_api_parses_classes_and_boxes(client):
    open_vocab, segmentation = install(client)
    detected = client.post(
        "/api/v1/tools/detect_open_vocab/execute",
        files={"image": ("scene.png", image_bytes(), "image/png")},
        data={"classes": json.dumps(["yellow safety helmet"]), "confidence": "0.3"},
    )
    assert detected.status_code == 200
    assert detected.json()["data"]["detection_count"] == 1
    assert open_vocab.calls[0][1] == ["yellow safety helmet"]
    assert open_vocab.calls[0][2:] == (0.3, 0.45)
    executed = detected.json()["metadata"]["arguments_summary"]
    assert executed["classes"] == ["yellow safety helmet"]
    assert executed["confidence"] == 0.3
    assert executed["iou_threshold"] == 0.45
    assert executed["image_path"]
    assert "\\" not in executed["image_path"] and "/" not in executed["image_path"]
    trace = client.get("/api/v1/tools/executions", params={"limit": 1}).json()[0]
    assert trace["arguments_summary"] == executed
    segmented = client.post(
        "/api/v1/tools/segment_objects/execute",
        files={"image": ("scene.png", image_bytes(), "image/png")},
        data={"boxes": json.dumps([{"x1": 5, "y1": 4, "x2": 45, "y2": 40}])},
    )
    assert segmented.status_code == 200
    body = segmented.json()
    assert body["data"]["segment_count"] == 1
    assert [item["type"] for item in body["artifacts"]] == ["mask", "image"]
    assert segmentation.calls[0][1][0] == BoundingBox(x1=5, y1=4, x2=45, y2=40)


def test_open_vocab_api_defaults_match_executor_defaults(client):
    open_vocab, _ = install(client)
    response = client.post(
        "/api/v1/tools/detect_open_vocab/execute",
        files={"image": ("scene.png", image_bytes(), "image/png")},
        data={"classes": json.dumps(["a person wearing a yellow helmet"])},
    )
    assert response.status_code == 200
    assert open_vocab.calls[0][1:] == (
        ["a person wearing a yellow helmet"], 0.25, 0.45
    )
    assert response.json()["metadata"]["arguments_summary"] == {
        "image_path": response.json()["metadata"]["arguments_summary"]["image_path"],
        "classes": ["a person wearing a yellow helmet"],
        "confidence": 0.25,
        "iou_threshold": 0.45,
    }


def test_phase5_tool_api_rejects_bad_inputs(client):
    install(client)
    bad_classes = client.post(
        "/api/v1/tools/detect_open_vocab/execute",
        files={"image": ("scene.png", image_bytes(), "image/png")},
        data={"classes": "[]"},
    )
    assert bad_classes.status_code == 422
    assert bad_classes.json()["error"]["code"] == "INVALID_TOOL_INPUT"
    bad_boxes = client.post(
        "/api/v1/tools/segment_objects/execute",
        files={"image": ("scene.png", image_bytes(), "image/png")},
        data={"boxes": "not-json"},
    )
    assert bad_boxes.status_code == 422
    assert bad_boxes.json()["error"]["code"] == "NO_VALID_BOXES"

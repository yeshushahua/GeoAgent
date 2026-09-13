import io
import logging

from PIL import Image

from backend.app.detection.schemas import BoundingBox, Detection
from backend.app.tools import build_tool_system
from backend.tests.test_detect_objects_tool import FakeDetector


def image_bytes():
    payload = io.BytesIO()
    Image.new("RGB", (100, 60), "white").save(payload, "PNG")
    return payload.getvalue()


def test_detector_status_is_lazy(client):
    response = client.get("/api/v1/models/detector/status")
    assert response.status_code == 200
    assert response.json()["state"] == "UNLOADED"
    assert response.json()["load_count"] == 0
    assert response.json()["device"] == "cuda:0"


def test_detect_objects_through_live_shape_tool_api(client):
    detector = FakeDetector([
        Detection(
            class_id=5, class_name="bus", confidence=0.87,
            bbox=BoundingBox(x1=5, y1=4, x2=95, y2=58),
        )
    ])
    registry, executor, traces = build_tool_system(
        client.app.state.settings,
        client.app.state.model_manager,
        logging.getLogger("test"),
        detector,
    )
    client.app.state.detector_manager = detector
    client.app.state.tool_registry = registry
    client.app.state.tool_executor = executor
    client.app.state.tool_traces = traces
    response = client.post(
        "/api/v1/tools/detect_objects/execute",
        files={"image": ("scene.png", image_bytes(), "image/png")},
        data={"confidence": "0.4", "iou_threshold": "0.5", "classes": '["bus"]'},
    )
    body = response.json()
    assert response.status_code == 200 and body["success"]
    assert body["data"]["class_counts"] == {"bus": 1}
    assert body["artifacts"][0]["path"].endswith("annotated.jpg")
    assert detector.calls[0][1:] == (0.4, 0.5, ["bus"])

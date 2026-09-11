import io
import logging

from PIL import Image

from backend.app.tools import build_tool_system
from backend.tests.test_tool_system import FakeManager


def image_bytes(size=(40, 20)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "green").save(buffer, "PNG")
    return buffer.getvalue()


def install_fake_tool_system(client):
    manager = FakeManager()
    registry, executor, traces = build_tool_system(
        client.app.state.settings, manager, logging.getLogger("test")
    )
    client.app.state.model_manager = manager
    client.app.state.tool_registry = registry
    client.app.state.tool_executor = executor
    client.app.state.tool_traces = traces
    return manager


def test_tool_discovery_api(client):
    response = client.get("/api/v1/tools")
    assert response.status_code == 200
    assert [item["name"] for item in response.json()] == [
        "analyze_image", "crop_image", "inspect_image"
    ]
    detail = client.get("/api/v1/tools/analyze_image")
    assert detail.status_code == 200
    assert detail.json()["input_schema"]["required"] == ["image_path", "prompt"]
    assert client.get("/api/v1/tools/no_such_tool").status_code == 404


def test_tool_api_inspect_crop_and_trace(client):
    inspect = client.post(
        "/api/v1/tools/inspect_image/execute",
        files={"image": ("../../unsafe.png", image_bytes(), "image/png")},
    )
    assert inspect.status_code == 200
    assert inspect.json()["data"]["width"] == 40
    crop = client.post(
        "/api/v1/tools/crop_image/execute",
        files={"image": ("scene.png", image_bytes(), "image/png")},
        data={"x1": "2", "y1": "3", "x2": "20", "y2": "15"},
    )
    body = crop.json()
    assert crop.status_code == 200 and body["success"]
    assert body["artifacts"][0]["type"] == "image"
    assert "unsafe" not in body["artifacts"][0]["path"]
    traces = client.get("/api/v1/tools/executions?limit=2").json()
    assert len(traces) == 2 and traces[0]["tool"] == "crop_image"
    assert "prompt" not in traces[0]


def test_tool_api_analyze_auto_load_and_errors(client):
    manager = install_fake_tool_system(client)
    response = client.post(
        "/api/v1/tools/analyze_image/execute",
        files={"image": ("scene.png", image_bytes(), "image/png")},
        data={"prompt": "what is shown?", "max_new_tokens": "64"},
    )
    body = response.json()
    assert response.status_code == 200 and body["success"]
    assert body["data"]["answer"] == "answer: what is shown?"
    assert body["metadata"]["execution_id"]
    assert manager.load_calls == 1
    missing = client.post("/api/v1/tools/analyze_image/execute")
    assert missing.status_code == 422
    assert missing.json()["error"]["type"] == "INVALID_TOOL_INPUT"
    invalid_number = client.post(
        "/api/v1/tools/crop_image/execute",
        files={"image": ("scene.png", image_bytes(), "image/png")},
        data={"x1": "left", "y1": "0", "x2": "20", "y2": "10"},
    )
    assert invalid_number.status_code == 422
    assert invalid_number.json()["error"]["type"] == "INVALID_TOOL_INPUT"
    bad_image = client.post(
        "/api/v1/tools/inspect_image/execute",
        files={"image": ("bad.png", b"not-image", "image/png")},
    )
    assert bad_image.status_code == 415
    assert bad_image.json()["error"]["type"] == "INVALID_IMAGE"

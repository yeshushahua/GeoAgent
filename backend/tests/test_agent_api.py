import io
import logging

from PIL import Image

from backend.app.agent.service import VisionAgent
from backend.app.agent.trace import AgentTraceStore
from backend.app.tools import build_tool_system
from backend.tests.test_agent_service import FakeAgentManager, ScriptedPlanner


def image_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (24, 12), "orange").save(buffer, "PNG")
    return buffer.getvalue()


def install_fake_agent(client):
    manager = FakeAgentManager(client.app.state.settings)
    registry, executor, tool_traces = build_tool_system(
        client.app.state.settings, manager, logging.getLogger("test")
    )
    planner = ScriptedPlanner([
        lambda state: '{"type":"tool_call","tool_name":"inspect_image","arguments":{"image_path":"%s"}}' % state.original_image_path.replace("\\", "\\\\"),
        '{"type":"final","answer":"图片尺寸为 24 × 12 像素。"}',
    ])
    traces = AgentTraceStore(50)
    agent = VisionAgent(
        registry, executor, manager, traces, logging.getLogger("test"), planner=planner
    )
    client.app.state.model_manager = manager
    client.app.state.tool_registry = registry
    client.app.state.tool_executor = executor
    client.app.state.tool_traces = tool_traces
    client.app.state.vision_agent = agent
    client.app.state.agent_traces = traces


def test_agent_api_and_safe_trace(client):
    install_fake_agent(client)
    response = client.post(
        "/api/v1/agent/run",
        files={"image": ("scene.png", image_bytes(), "image/png")},
        data={"message": "告诉我图片尺寸", "max_steps": "4", "max_new_tokens": "64"},
    )
    body = response.json()
    assert response.status_code == 200 and body["success"]
    assert body["answer"] == "图片尺寸为 24 × 12 像素。"
    assert body["steps"][0]["tool_name"] == "inspect_image"
    traces = client.get("/api/v1/agent/executions?limit=5")
    assert traces.status_code == 200 and len(traces.json()) == 1
    trace_text = traces.text
    assert "告诉我图片尺寸" not in trace_text
    assert "system prompt" not in trace_text.lower()


def test_agent_api_missing_image_and_existing_tool_api(client):
    missing = client.post("/api/v1/agent/run", data={"message": "分析图片"})
    assert missing.status_code == 422
    assert missing.json()["error"]["code"] == "INVALID_REQUEST"
    blank = client.post(
        "/api/v1/agent/run",
        files={"image": ("scene.png", image_bytes(), "image/png")},
        data={"message": "   "},
    )
    assert blank.status_code == 422
    assert blank.json()["error"]["code"] == "INVALID_INPUT"
    assert client.get("/api/v1/tools").status_code == 200
    assert client.get("/api/v1/models/vlm/status").status_code == 200

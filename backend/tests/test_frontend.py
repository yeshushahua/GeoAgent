import inspect

import httpx

from frontend.app import PLACEHOLDER, _agent_markdown, _post_agent, analyze, chat, fetch_status


def test_chat_placeholder():
    text, history = chat("hello", [])
    assert text == ""
    assert history == [{"role": "user", "content": "hello"}, {"role": "assistant", "content": PLACEHOLDER}]
    assert chat("  ", history)[1] == history


def test_backend_unavailable(settings, monkeypatch):
    def unavailable(*args, **kwargs):
        raise httpx.ConnectError("backend stopped")
    monkeypatch.setattr(httpx.Client, "get", unavailable)
    title, details = fetch_status(settings)
    assert "后端不可用" in title
    assert details == {"backend": "unavailable"}


def test_analysis_boundary_uses_tool_api_only():
    source = inspect.getsource(_post_agent) + inspect.getsource(analyze)
    assert "/agent/run" in source
    assert "/tools/" not in source
    assert "/models/vlm/infer" not in source


def test_agent_panel_shows_detection_and_segmentation_summary():
    panel = _agent_markdown({
        "run_id": "run-1",
        "steps": [{
            "index": 1, "decision_type": "tool_call", "tool_name": "detect_objects",
            "success": True, "duration_ms": 12.0,
            "observation_summary": {
                "detection_count": 5, "class_counts": {"bus": 1, "person": 4}
            },
        }, {
            "index": 2, "decision_type": "tool_call", "tool_name": "detect_open_vocab",
            "success": True, "duration_ms": 20.0,
            "observation_summary": {
                "detection_count": 1, "class_counts": {"yellow safety helmet": 1}
            },
        }, {
            "index": 3, "decision_type": "tool_call", "tool_name": "segment_objects",
            "success": True, "duration_ms": 30.0,
            "observation_summary": {
                "segment_count": 1, "segments": [{"mask_area_ratio": 0.125}]
            },
        }],
        "metadata": {},
    })
    assert "检测总数：**5**" in panel
    assert "bus × 1" in panel and "person × 4" in panel
    assert "开放检测总数：**1**" in panel and "yellow safety helmet × 1" in panel
    assert "分割实例：**1**" in panel and "12.50%" in panel

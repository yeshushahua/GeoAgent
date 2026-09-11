import inspect

import httpx

from frontend.app import PLACEHOLDER, _post_tool, analyze, chat, fetch_status


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
    assert "Backend unavailable" in title
    assert details == {"backend": "unavailable"}


def test_analysis_boundary_uses_tool_api_only():
    source = inspect.getsource(_post_tool) + inspect.getsource(analyze)
    assert "/tools/{tool_name}/execute" in source
    assert "/models/vlm/infer" not in source

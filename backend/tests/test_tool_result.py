import pytest
from pydantic import ValidationError

from backend.app.schemas.tool_result import Artifact, ToolError, ToolResult


def test_success_roundtrip():
    result = ToolResult(success=True, tool="preview", data={"width": 320},
                        artifacts=[Artifact(kind="image", path="outputs/image.png")],
                        metadata={"runtime_s": 0.01})
    assert ToolResult.model_validate_json(result.model_dump_json()) == result


def test_structured_failure():
    result = ToolResult(success=False, tool="preview", error=ToolError(code="INVALID_INPUT", message="No image"))
    assert result.error.code == "INVALID_INPUT"


@pytest.mark.parametrize("data", [
    {"success": False, "tool": "test"},
    {"success": True, "tool": ""},
    {"success": "yes", "tool": "test"},
    {"success": True, "tool": "test", "error": {"code": "X", "message": "bad"}},
])
def test_invalid_results(data):
    with pytest.raises(ValidationError):
        ToolResult.model_validate(data)


def test_mutable_defaults_are_independent():
    a, b = ToolResult(success=True, tool="a"), ToolResult(success=True, tool="b")
    a.data["x"] = 1
    assert b.data == {}

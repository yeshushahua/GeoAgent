import pytest

from backend.app.agent.errors import AgentParseError, AgentValidationError, UnknownToolError
from backend.app.agent.parser import AgentOutputParser
from backend.app.agent.schemas import AgentFinal, AgentToolCall
from backend.app.tools import build_tool_system
from backend.tests.test_tool_system import FakeManager


def registry(settings):
    import logging
    return build_tool_system(settings, FakeManager(), logging.getLogger("test"))[0]


def test_parse_tool_call_and_validate_arguments(settings):
    decision = AgentOutputParser().parse(
        '{"type":"tool_call","tool_name":"inspect_image","arguments":{"image_path":"x.png"}}',
        registry(settings),
    )
    assert isinstance(decision, AgentToolCall)
    assert decision.tool_name == "inspect_image"


@pytest.mark.parametrize("raw", [
    '```json\n{"type":"final","answer":"完成"}\n```',
    'Here is the result: {"type":"final","answer":"完成"}',
])
def test_parse_fenced_or_wrapped_final(settings, raw):
    decision = AgentOutputParser().parse(raw, registry(settings))
    assert isinstance(decision, AgentFinal) and decision.answer == "完成"


def test_parser_rejects_malformed_unknown_and_bad_arguments(settings):
    parser = AgentOutputParser()
    tools = registry(settings)
    with pytest.raises(AgentParseError):
        parser.parse("not json", tools)
    with pytest.raises(UnknownToolError):
        parser.parse(
            '{"type":"tool_call","tool_name":"invent_tool","arguments":{}}', tools
        )
    with pytest.raises(AgentValidationError):
        parser.parse(
            '{"type":"tool_call","tool_name":"crop_image","arguments":{"image_path":"x"}}',
            tools,
        )

from __future__ import annotations

import json

from pydantic import ValidationError

from backend.app.agent.errors import (
    AgentParseError,
    AgentValidationError,
    UnknownToolError,
)
from backend.app.agent.schemas import (
    AGENT_DECISION_ADAPTER,
    AgentDecision,
    AgentToolCall,
)
from backend.app.tools.registry import ToolRegistry


class AgentOutputParser:
    def parse(self, raw: str, registry: ToolRegistry) -> AgentDecision:
        payload = self._extract_json(raw)
        # Small local models occasionally use the registered tool name as the
        # discriminator. Normalize that unambiguous shorthand through the runtime
        # registry; argument validation below remains identical to a canonical call.
        shorthand = payload.get("type")
        if (
            isinstance(shorthand, str)
            and registry.has(shorthand)
            and "tool_name" not in payload
        ):
            arguments = payload.get("arguments")
            if arguments is None:
                arguments = {
                    key: value for key, value in payload.items() if key != "type"
                }
            payload = {
                "type": "tool_call",
                "tool_name": shorthand,
                "arguments": arguments,
            }
        try:
            decision = AGENT_DECISION_ADAPTER.validate_python(payload)
        except ValidationError as exc:
            raise AgentValidationError("Agent decision does not match the required schema") from exc
        if isinstance(decision, AgentToolCall):
            if not registry.has(decision.tool_name):
                raise UnknownToolError(f"Agent selected unknown tool: {decision.tool_name}")
            try:
                normalized = registry.get(decision.tool_name).input_schema.model_validate(
                    decision.arguments
                ).model_dump(mode="json", exclude_none=True)
            except ValidationError as exc:
                raise AgentValidationError(
                    f"Arguments are invalid for tool {decision.tool_name}"
                ) from exc
            decision = decision.model_copy(update={"arguments": normalized})
        return decision

    @staticmethod
    def _extract_json(raw: str) -> dict:
        if not isinstance(raw, str) or not raw.strip():
            raise AgentParseError("Agent returned an empty decision")
        decoder = json.JSONDecoder()
        text = raw.strip()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        raise AgentParseError("Agent response did not contain a valid JSON object")

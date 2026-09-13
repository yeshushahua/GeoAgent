class AgentError(RuntimeError):
    code = "AGENT_ERROR"


class AgentParseError(AgentError):
    code = "AGENT_PARSE_ERROR"


class AgentValidationError(AgentError):
    code = "AGENT_VALIDATION_ERROR"


class UnknownToolError(AgentError):
    code = "UNKNOWN_TOOL"


class MaxStepsExceededError(AgentError):
    code = "MAX_STEPS_EXCEEDED"


class DuplicateToolCallError(AgentError):
    code = "DUPLICATE_TOOL_CALL"


class AgentModelError(AgentError):
    code = "AGENT_MODEL_ERROR"

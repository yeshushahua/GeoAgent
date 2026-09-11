class ToolExecutionError(RuntimeError):
    code = "TOOL_EXECUTION_FAILED"


class InvalidToolInputError(ToolExecutionError):
    code = "INVALID_TOOL_INPUT"


class InvalidCropError(ToolExecutionError):
    code = "INVALID_CROP"


class InvalidToolImageError(ToolExecutionError):
    code = "INVALID_IMAGE"

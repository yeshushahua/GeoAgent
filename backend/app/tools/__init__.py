"""Future callable tools; depend on models/services and return ToolResult."""
from backend.app.tools.context import ToolContext
from backend.app.tools.executor import ToolExecutor
from backend.app.tools.registry import ToolRegistry
from backend.app.tools.trace import ToolTraceStore
from backend.app.tools.utility import CropImageTool
from backend.app.tools.vision import AnalyzeImageTool, InspectImageTool


def build_tool_system(settings, model_manager, logger):
    registry = ToolRegistry()
    registry.register(InspectImageTool())
    registry.register(CropImageTool())
    registry.register(AnalyzeImageTool())
    traces = ToolTraceStore(settings.tool_trace_limit)
    context = ToolContext(settings, model_manager, logger, traces)
    return registry, ToolExecutor(registry, context), traces


__all__ = ["ToolContext", "ToolExecutor", "ToolRegistry", "build_tool_system"]

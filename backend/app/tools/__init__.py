"""Future callable tools; depend on models/services and return ToolResult."""
from backend.app.tools.context import ToolContext
from backend.app.tools.executor import ToolExecutor
from backend.app.tools.registry import ToolRegistry
from backend.app.tools.trace import ToolTraceStore
from backend.app.tools.utility import CropImageTool
from backend.app.tools.vision import (
    AnalyzeImageTool,
    DetectObjectsTool,
    DetectOpenVocabularyTool,
    InspectImageTool,
    SegmentObjectsTool,
)


def build_tool_registry():
    """Create the canonical registry used by the API, Agent, and manual UI."""
    registry = ToolRegistry()
    registry.register(InspectImageTool())
    registry.register(CropImageTool())
    registry.register(AnalyzeImageTool())
    registry.register(DetectObjectsTool())
    registry.register(DetectOpenVocabularyTool())
    registry.register(SegmentObjectsTool())
    return registry


def build_tool_system(
    settings, model_manager, logger, detector_manager=None,
    open_vocab_manager=None, segmentation_manager=None,
):
    if detector_manager is None:
        from backend.app.detection import DetectorManager

        detector_manager = DetectorManager(settings, logger)
    if open_vocab_manager is None:
        from backend.app.open_vocabulary import OpenVocabularyDetectorManager

        open_vocab_manager = OpenVocabularyDetectorManager(settings, logger)
    if segmentation_manager is None:
        from backend.app.segmentation import SegmentationManager

        segmentation_manager = SegmentationManager(settings, logger)
    registry = build_tool_registry()
    traces = ToolTraceStore(settings.tool_trace_limit)
    context = ToolContext(
        settings, model_manager, logger, traces, detector_manager,
        open_vocab_manager, segmentation_manager,
    )
    return registry, ToolExecutor(registry, context), traces


__all__ = [
    "ToolContext", "ToolExecutor", "ToolRegistry", "build_tool_registry", "build_tool_system"
]

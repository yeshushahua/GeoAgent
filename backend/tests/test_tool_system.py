import asyncio
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
import pytest

from backend.app.schemas.inference import GenerationInfo, GpuMemory, ImageInfo, InferenceResult
from backend.app.schemas.tool_result import ToolResult
from backend.app.tools.base import BaseTool
from backend.app.tools.context import ToolContext
from backend.app.tools.executor import ToolExecutor
from backend.app.tools.registry import DuplicateToolError, ToolRegistry
from backend.app.tools.trace import ToolTraceStore
from backend.app.tools.utility.image_crop import CropImageTool
from backend.app.tools.vision.image_metadata import InspectImageTool
from backend.app.tools.vision.vlm_analyze import AnalyzeImageTool


class EchoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int = Field(gt=0)


class EchoTool(BaseTool):
    name = "echo_value"
    description = "Return a validated positive integer."
    category = "test"
    input_schema = EchoInput

    async def execute(self, inputs, context, execution_id):
        return ToolResult(success=True, tool=self.name, data={"value": inputs.value})


class BrokenTool(BaseTool):
    name = "break_safely"
    description = "Raise an exception for executor isolation testing."
    category = "test"
    input_schema = EchoInput

    async def execute(self, inputs, context, execution_id):
        raise RuntimeError("secret traceback detail")


class SlowTool(BaseTool):
    name = "wait_too_long"
    description = "Wait longer than the configured timeout."
    category = "test"
    input_schema = EchoInput

    async def execute(self, inputs, context, execution_id):
        await asyncio.sleep(0.05)
        return ToolResult(success=True, tool=self.name)


class FakeManager:
    def __init__(self):
        self.state = "UNLOADED"
        self.load_calls = 0
        self.infer_calls = 0

    def load_model(self):
        self.load_calls += 1
        self.state = "READY"
        return self.status()

    def status(self):
        return {"state": self.state, "last_error": None}

    def infer(self, image, prompt, max_new_tokens):
        self.infer_calls += 1
        return InferenceResult(
            success=True,
            model="Qwen3-VL-4B-Instruct",
            text=f"answer: {prompt}",
            latency_ms=12.5,
            device="cuda:0",
            dtype="bfloat16",
            image=ImageInfo(
                width=image.width,
                height=image.height,
                mode=image.mode,
                format=image.format,
                preprocessing_width=image.width,
                preprocessing_height=image.height,
                resized=False,
            ),
            generation=GenerationInfo(max_new_tokens=max_new_tokens),
            gpu=GpuMemory(allocated_gb=8, reserved_gb=9, peak_allocated_gb=9),
        )


def make_context(settings, manager=None):
    import logging
    return ToolContext(settings, manager or FakeManager(), logging.getLogger("test"), ToolTraceStore(50))


def save_image(path: Path, size=(40, 20)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "navy").save(path, "PNG")
    return path


def test_registry_discovery_and_duplicate_protection():
    registry = ToolRegistry()
    registry.register(EchoTool())
    assert registry.has("echo_value")
    assert registry.get("echo_value").name == "echo_value"
    definition = registry.list_tools()[0]
    assert definition["input_schema"]["properties"]["value"]["exclusiveMinimum"] == 0
    assert definition["description"] == "Return a validated positive integer."
    with pytest.raises(DuplicateToolError):
        registry.register(EchoTool())
    assert registry.unregister("echo_value").name == "echo_value"


def test_base_tool_rejects_invalid_name():
    with pytest.raises(TypeError):
        class BadNameTool(BaseTool):
            name = "BadTool"
            description = "invalid"
            category = "test"
            input_schema = EchoInput

            async def execute(self, inputs, context, execution_id):
                return ToolResult(success=True, tool=self.name)


@pytest.mark.anyio
async def test_executor_validation_trace_ids_and_exception_isolation(settings):
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(BrokenTool())
    context = make_context(settings)
    executor = ToolExecutor(registry, context)
    first = await executor.execute("echo_value", {"value": 7})
    invalid = await executor.execute("echo_value", {"value": 0, "extra": 1})
    unknown = await executor.execute("missing_tool", {})
    broken = await executor.execute("break_safely", {"value": 1})
    assert first.success and first.data == {"value": 7}
    assert invalid.error.code == "INVALID_TOOL_INPUT"
    assert unknown.error.code == "TOOL_NOT_FOUND"
    assert broken.error.code == "TOOL_EXECUTION_FAILED"
    assert "secret" not in broken.error.message
    ids = [item.execution_id for item in context.trace_store.list(10)]
    assert len(ids) == len(set(ids)) == 4
    assert first.metadata["duration_ms"] >= 0
    assert first.metadata["started_at"] <= first.metadata["finished_at"]


@pytest.mark.anyio
async def test_executor_timeout(settings):
    settings.tool_timeout_seconds = 0.01
    registry = ToolRegistry()
    registry.register(SlowTool())
    result = await ToolExecutor(registry, make_context(settings)).execute(
        "wait_too_long", {"value": 1}
    )
    assert not result.success and result.error.code == "TOOL_TIMEOUT"


@pytest.mark.anyio
async def test_inspect_and_crop_tools(settings):
    source = save_image(settings.project_root / "sample.png")
    registry = ToolRegistry()
    registry.register(InspectImageTool())
    registry.register(CropImageTool())
    executor = ToolExecutor(registry, make_context(settings))
    inspected = await executor.execute("inspect_image", {"image_path": str(source)})
    assert inspected.data == {
        "width": 40, "height": 20, "mode": "RGB", "format": "PNG",
        "file_size": source.stat().st_size, "aspect_ratio": 2.0,
    }
    cropped = await executor.execute(
        "crop_image", {"image_path": str(source), "x1": 5, "y1": 2, "x2": 25, "y2": 12}
    )
    artifact = Path(cropped.artifacts[0].path)
    assert cropped.success and artifact.is_file()
    assert artifact.is_relative_to(settings.output_dir / "tools")
    assert artifact.name == "crop.png"
    with Image.open(artifact) as image:
        assert image.size == (20, 10)
    invalid = await executor.execute(
        "crop_image", {"image_path": str(source), "x1": 0, "y1": 0, "x2": 100, "y2": 10}
    )
    assert invalid.error.code == "INVALID_CROP"


@pytest.mark.anyio
async def test_analyze_image_wraps_manager_and_auto_loads(settings):
    source = save_image(settings.project_root / "analyze.png")
    manager = FakeManager()
    registry = ToolRegistry()
    registry.register(AnalyzeImageTool())
    result = await ToolExecutor(registry, make_context(settings, manager)).execute(
        "analyze_image",
        {"image_path": str(source), "prompt": "describe", "max_new_tokens": 64},
    )
    assert result.success and result.data["answer"] == "answer: describe"
    assert manager.load_calls == 1 and manager.infer_calls == 1
    assert result.metadata["model"] == "Qwen3-VL-4B-Instruct"
    assert result.metadata["device"] == "cuda:0"
    assert result.metadata["gpu_peak_gb"] == 9

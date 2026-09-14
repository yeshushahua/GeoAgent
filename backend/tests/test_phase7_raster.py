from __future__ import annotations

from datetime import datetime, timezone
import io
import logging
from pathlib import Path

import anyio
import numpy as np
from PIL import Image
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from backend.app.agent.schemas import AgentRequest, AgentToolCall
from backend.app.agent.service import VisionAgent
from backend.app.agent.state import AgentState
from backend.app.agent.trace import AgentTraceStore
from backend.app.agent.workflow import WorkflowController, WorkflowDependencyError
from backend.app.raster import is_raster_candidate
from backend.app.tools import build_tool_system
from backend.app.tools.context import ToolContext
from backend.app.tools.executor import ToolExecutor
from backend.app.tools.raster import (
    CropRasterTool,
    InspectRasterTool,
    RasterPreviewTool,
    RasterStatisticsTool,
)
from backend.app.tools.registry import ToolRegistry
from backend.app.tools.trace import ToolTraceStore
from backend.tests.test_agent_service import FakeAgentManager, ScriptedPlanner
from backend.tests.test_tool_system import FakeManager


def make_raster(
    path: Path,
    *,
    width: int = 128,
    height: int = 64,
    count: int = 3,
    dtype: str = "uint16",
    nodata=0,
    crs: str | None = "EPSG:32647",
    tiled: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(100, 200, 10, 10)
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": count,
        "dtype": dtype,
        "transform": transform,
        "nodata": nodata,
    }
    if crs:
        profile["crs"] = crs
    if tiled:
        profile.update(tiled=True, blockxsize=256, blockysize=256, compress="deflate")
    base = np.arange(width * height, dtype=np.float64).reshape(height, width)
    with rasterio.open(path, "w", **profile) as dst:
        for band in range(1, count + 1):
            values = (base + band * 100).astype(dtype)
            if nodata is not None:
                values[0, 0] = nodata
            dst.write(values, band)
            dst.set_band_description(band, f"Channel {band}")
    return path


def raster_bytes() -> bytes:
    data = np.arange(3 * 16 * 24, dtype=np.uint16).reshape(3, 16, 24)
    profile = {
        "driver": "GTiff", "width": 24, "height": 16, "count": 3,
        "dtype": "uint16", "crs": "EPSG:32647",
        "transform": from_origin(100, 200, 10, 10), "nodata": 0,
    }
    with MemoryFile() as memory:
        with memory.open(**profile) as dst:
            dst.write(data)
        return memory.read()


def executor(settings) -> ToolExecutor:
    registry = ToolRegistry()
    for tool in (
        InspectRasterTool(), RasterPreviewTool(), CropRasterTool(), RasterStatisticsTool()
    ):
        registry.register(tool)
    context = ToolContext(
        settings=settings,
        model_manager=FakeManager(),
        logger=logging.getLogger("phase7"),
        trace_store=ToolTraceStore(50),
    )
    return ToolExecutor(registry, context)


def state_for(path: Path) -> AgentState:
    state = AgentState(
        run_id="phase7", user_message="test", original_image_path=str(path),
        tool_definitions=[], max_steps=6, max_new_tokens=64,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    WorkflowController.initialize(state, str(path))
    return state


def call(name: str, arguments: dict) -> AgentToolCall:
    return AgentToolCall(type="tool_call", tool_name=name, arguments=arguments)


def test_geotiff_candidate_uses_extension_or_magic(settings):
    tif = make_raster(settings.project_root / "scene.tif")
    hidden = settings.project_root / "scene.upload"
    hidden.write_bytes(tif.read_bytes())
    ordinary = settings.project_root / "scene.png"
    Image.new("RGB", (4, 4)).save(ordinary)
    assert is_raster_candidate(tif)
    assert is_raster_candidate(hidden)
    assert not is_raster_candidate(ordinary)


@pytest.mark.anyio
async def test_inspect_raster_extracts_complete_metadata(settings):
    path = make_raster(settings.project_root / "metadata.tif")
    result = await executor(settings).execute("inspect_raster", {"raster_path": str(path)})
    assert result.success
    assert result.data["width"] == 128 and result.data["height"] == 64
    assert result.data["band_count"] == 3
    assert result.data["dtypes"] == ["uint16"] * 3
    assert result.data["driver"] == "GTiff"
    assert result.data["crs"] == "EPSG:32647" and result.data["epsg"] == 32647
    assert result.data["resolution_x"] == result.data["resolution_y"] == 10
    assert result.data["bounds"] == {
        "left": 100.0, "bottom": -440.0, "right": 1380.0, "top": 200.0
    }
    assert result.data["nodata"] == 0
    assert result.data["band_descriptions"] == ["Channel 1", "Channel 2", "Channel 3"]


@pytest.mark.anyio
async def test_raster_without_crs_or_nodata_is_valid(settings):
    path = make_raster(
        settings.project_root / "plain.tiff", count=1, crs=None, nodata=None
    )
    result = await executor(settings).execute("inspect_raster", {"raster_path": str(path)})
    assert result.success
    assert result.data["crs"] is None and result.data["epsg"] is None
    assert result.data["nodata"] is None


@pytest.mark.anyio
async def test_percentile_preview_downsamples_and_validates_bands(settings):
    path = make_raster(settings.project_root / "preview.tif", width=256, height=128)
    tool_executor = executor(settings)
    result = await tool_executor.execute("raster_preview", {
        "raster_path": str(path), "max_size": 64,
    })
    assert result.success
    assert result.data["bands"] == [1, 2, 3]
    assert result.data["stretch"] == "percentile"
    assert result.data["percentile_range"] == [2.0, 98.0]
    assert result.data["read_strategy"] == "downsampled_out_shape"
    with Image.open(result.artifacts[0].path) as preview:
        assert preview.mode == "RGB" and preview.size == (64, 32)
    invalid = await tool_executor.execute("raster_preview", {
        "raster_path": str(path), "bands": [4],
    })
    assert not invalid.success and invalid.error.code == "BAND_NOT_FOUND"


@pytest.mark.anyio
async def test_one_and_two_band_preview_rules(settings):
    one = make_raster(settings.project_root / "one.tif", count=1)
    two = make_raster(settings.project_root / "two.tif", count=2)
    tool_executor = executor(settings)
    grayscale = await tool_executor.execute("raster_preview", {"raster_path": str(one)})
    assert grayscale.success and grayscale.data["bands"] == [1]
    with Image.open(grayscale.artifacts[0].path) as image:
        assert image.mode == "L"
    ambiguous = await tool_executor.execute("raster_preview", {"raster_path": str(two)})
    assert not ambiguous.success and ambiguous.error.code == "BAND_NOT_FOUND"
    selected = await tool_executor.execute(
        "raster_preview", {"raster_path": str(two), "bands": [2]}
    )
    assert selected.success and selected.data["bands"] == [2]


@pytest.mark.anyio
async def test_crop_preserves_georeferencing_and_updates_transform(settings):
    path = make_raster(settings.project_root / "crop-source.tif")
    result = await executor(settings).execute("crop_raster", {
        "raster_path": str(path),
        "row_start": 10, "row_end": 30, "col_start": 20, "col_end": 70,
    })
    assert result.success
    metadata = result.data["metadata"]
    assert metadata["width"] == 50 and metadata["height"] == 20
    assert metadata["crs"] == "EPSG:32647"
    assert metadata["resolution_x"] == metadata["resolution_y"] == 10
    assert metadata["transform"] == [10.0, 0.0, 300.0, 0.0, -10.0, 100.0]
    assert metadata["bounds"] == {
        "left": 300.0, "bottom": -100.0, "right": 800.0, "top": 100.0
    }
    with rasterio.open(result.artifacts[0].path) as crop:
        assert crop.crs == rasterio.crs.CRS.from_epsg(32647)
        assert crop.count == 3 and crop.nodata == 0
        assert crop.descriptions == ("Channel 1", "Channel 2", "Channel 3")


@pytest.mark.anyio
async def test_named_quadrant_uses_each_raster_dimension(settings):
    path = make_raster(settings.project_root / "named-window.tif")
    result = await executor(settings).execute("crop_raster", {
        "raster_path": str(path), "region": "top_left_quarter",
    })
    assert result.success
    assert result.data["window"] == {
        "row_start": 0, "row_end": 32, "col_start": 0, "col_end": 64,
        "region": "top_left_quarter",
    }
    assert result.data["metadata"]["width"] == 64
    assert result.data["metadata"]["height"] == 32


@pytest.mark.anyio
async def test_crop_rejects_empty_and_out_of_bounds_windows(settings):
    path = make_raster(settings.project_root / "invalid-window.tif")
    tool_executor = executor(settings)
    empty = await tool_executor.execute("crop_raster", {
        "raster_path": str(path),
        "row_start": 2, "row_end": 2, "col_start": 1, "col_end": 4,
    })
    assert not empty.success and empty.error.code == "EMPTY_WINDOW"
    outside = await tool_executor.execute("crop_raster", {
        "raster_path": str(path),
        "row_start": 0, "row_end": 65, "col_start": 0, "col_end": 128,
    })
    assert not outside.success and outside.error.code == "INVALID_WINDOW"


@pytest.mark.anyio
async def test_statistics_use_raw_values_and_exclude_nodata_nan_inf(settings):
    path = settings.project_root / "statistics.tif"
    values = np.arange(16, dtype=np.float32).reshape(4, 4)
    values[0, 0] = -9999
    values[0, 1] = np.nan
    values[0, 2] = np.inf
    with rasterio.open(
        path, "w", driver="GTiff", width=4, height=4, count=1,
        dtype="float32", nodata=-9999, transform=from_origin(0, 4, 1, 1),
    ) as dst:
        dst.write(values, 1)
    result = await executor(settings).execute(
        "raster_statistics", {"raster_path": str(path)}
    )
    assert result.success and result.data["read_strategy"] == "block_windows"
    stats = result.data["bands"][0]
    expected = np.arange(3, 16, dtype=np.float64)
    assert stats["valid_pixel_count"] == 13 and stats["nodata_count"] == 3
    assert stats["min"] == 3 and stats["max"] == 15
    assert stats["mean"] == pytest.approx(expected.mean())
    assert stats["std"] == pytest.approx(expected.std())


@pytest.mark.anyio
async def test_large_raster_preview_and_statistics_are_bounded(settings):
    path = settings.project_root / "large.tif"
    with rasterio.open(
        path, "w", driver="GTiff", width=4096, height=4096, count=1,
        dtype="uint8", transform=from_origin(0, 4096, 1, 1),
        tiled=True, blockxsize=256, blockysize=256, compress="deflate",
    ) as dst:
        block = np.ones((256, 256), dtype=np.uint8)
        for _, window in dst.block_windows(1):
            dst.write(block, 1, window=window)
    tool_executor = executor(settings)
    preview = await tool_executor.execute("raster_preview", {
        "raster_path": str(path), "max_size": 256,
    })
    assert preview.success
    assert preview.data["preview_width"] == preview.data["preview_height"] == 256
    statistics = await tool_executor.execute(
        "raster_statistics", {"raster_path": str(path)}
    )
    assert statistics.success
    assert statistics.data["bands"][0]["block_count"] == 256
    assert statistics.data["bands"][0]["valid_pixel_count"] == 4096 * 4096


@pytest.mark.anyio
async def test_raster_artifact_dependency_and_preview_isolation(settings):
    path = make_raster(settings.project_root / "workflow.tif")
    state = state_for(path)
    assert state.original_raster_artifact_id == state.active_raster_artifact_id == "raster-001"
    assert not state.active_image_artifact_id
    preview_call = call("raster_preview", {"raster_path": "active_raster"})
    with pytest.raises(WorkflowDependencyError) as raised:
        WorkflowController.prepare_call(state, preview_call)
    assert raised.value.code == "RASTER_METADATA_REQUIRED"

    inspect_call = call("inspect_raster", {"raster_path": "active_raster"})
    prepared, source_id = WorkflowController.prepare_call(state, inspect_call)
    inspected = await executor(settings).execute("inspect_raster", prepared.arguments)
    WorkflowController.apply_result(state, inspect_call, inspected, source_id, 1)
    assert WorkflowController.artifact(state, "raster-001").metadata["crs"] == "EPSG:32647"

    prepared, source_id = WorkflowController.prepare_call(state, preview_call)
    preview = await executor(settings).execute("raster_preview", prepared.arguments)
    WorkflowController.apply_result(state, preview_call, preview, source_id, 2)
    assert state.active_image_artifact_id == "raster-preview-001"
    assert state.active_raster_artifact_id == "raster-001"
    artifact = WorkflowController.artifact(state, "raster-preview-001")
    assert artifact.role == "visualization" and artifact.parent_artifact_id == "raster-001"
    with pytest.raises(WorkflowDependencyError) as mismatch:
        WorkflowController.resolve_raster(state, "raster-preview-001")
    assert mismatch.value.code == "INVALID_ARTIFACT_ROLE"


@pytest.mark.anyio
async def test_agent_raster_crop_statistics_workflow(settings):
    path = make_raster(settings.project_root / "agent.tif")
    planner = ScriptedPlanner([
        call("inspect_raster", {"raster_path": "active_raster"}).model_dump_json(),
        call("crop_raster", {
            "raster_path": "active_raster",
            "row_start": 0, "row_end": 32, "col_start": 0, "col_end": 64,
        }).model_dump_json(),
        call("raster_statistics", {"raster_path": "active_raster"}).model_dump_json(),
        '{"type":"final","answer":"已完成遥感栅格裁剪和统计。"}',
    ])
    manager = FakeAgentManager(settings)
    registry, tool_executor, _ = build_tool_system(
        settings, manager, logging.getLogger("phase7-agent")
    )
    agent = VisionAgent(
        registry, tool_executor, manager, AgentTraceStore(50),
        logging.getLogger("phase7-agent"), planner=planner,
    )
    response = await agent.run(AgentRequest(
        message="裁剪左上四分之一区域并统计", image_path=str(path), max_steps=5
    ))
    assert response.success
    assert [step.tool_name for step in response.steps if step.tool_name] == [
        "inspect_raster", "crop_raster", "raster_statistics"
    ]
    assert response.workflow.active_raster_artifact_id == "raster-crop-001"
    assert response.workflow.raster.artifact_id == "raster-crop-001"
    assert response.workflow.raster.statistics["read_strategy"] == "block_windows"
    assert response.workflow.active_image_artifact_id == ""


def test_raster_tool_api_upload_and_preview(client):
    payload = raster_bytes()
    inspect = client.post(
        "/api/v1/tools/inspect_raster/execute",
        files={"image": ("remote.tif", payload, "image/tiff")},
    )
    assert inspect.status_code == 200
    assert inspect.json()["data"]["crs"] == "EPSG:32647"
    preview = client.post(
        "/api/v1/tools/raster_preview/execute",
        files={"image": ("remote.tif", payload, "image/tiff")},
        data={"max_size": "64", "bands": "[1,2,3]"},
    )
    assert preview.status_code == 200 and preview.json()["success"]
    assert Path(preview.json()["artifacts"][0]["path"]).is_file()


def test_corrupt_tiff_returns_structured_raster_error(client):
    response = client.post(
        "/api/v1/tools/inspect_raster/execute",
        files={"image": ("broken.tif", b"II*\x00not-a-real-tiff", "image/tiff")},
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "RASTER_OPEN_FAILED"

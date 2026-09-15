from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from backend.app.agent.schemas import AgentRequest
from backend.app.agent.service import VisionAgent
from backend.app.agent.trace import AgentTraceStore
from backend.app.schemas.tool_result import ToolResult
from backend.app.tools import build_tool_registry, build_tool_system
from backend.app.tools.context import ToolContext
from backend.app.tools.executor import ToolExecutor
from backend.app.tools.geo import (
    CalculateAreaTool,
    ExportGeoJSONTool,
    GetRasterCoordinateTool,
)
from backend.app.tools.registry import ToolRegistry
from backend.app.tools.spatial import ZonalStatisticsTool
from backend.app.tools.trace import ToolTraceStore
from backend.tests.test_agent_service import FakeAgentManager, ScriptedPlanner, tool_call
from backend.tests.test_tool_system import FakeManager


def make_geo_raster(
    path: Path, *, crs: str | None = "EPSG:32647",
    transform=None, values: np.ndarray | None = None,
) -> Path:
    array = values if values is not None else np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff", "width": array.shape[1], "height": array.shape[0],
        "count": 1, "dtype": str(array.dtype),
        "transform": transform or from_origin(500000, 2000000, 10, 10),
    }
    if crs is not None:
        profile["crs"] = crs
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)
        dst.set_band_description(1, "Value")
    return path


def geo_raster_bytes() -> bytes:
    data = np.arange(1, 17, dtype=np.float32).reshape(1, 4, 4)
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff", width=4, height=4, count=1, dtype="float32",
            crs="EPSG:32647", transform=from_origin(500000, 2000000, 10, 10),
        ) as dst:
            dst.write(data)
        return memory.read()


def geo_executor(settings) -> ToolExecutor:
    registry = ToolRegistry()
    for tool in (
        GetRasterCoordinateTool(), ExportGeoJSONTool(),
        CalculateAreaTool(), ZonalStatisticsTool(),
    ):
        registry.register(tool)
    return ToolExecutor(
        registry,
        ToolContext(
            settings=settings, model_manager=FakeManager(),
            logger=logging.getLogger("phase8"), trace_store=ToolTraceStore(50),
        ),
    )


@pytest.mark.anyio
async def test_pixel_coordinate_for_utm_and_geographic_rasters(settings):
    executor = geo_executor(settings)
    utm = make_geo_raster(settings.project_root / "utm.tif")
    result = await executor.execute("get_raster_coordinate", {
        "raster_path": str(utm), "row": 0, "col": 0,
    })
    assert result.success
    assert result.data["projected_coordinate"] == {"x": 500005.0, "y": 1999995.0}
    assert result.data["source_crs"] == "EPSG:32647"
    assert 96 < result.data["longitude"] < 102
    assert 17 < result.data["latitude"] < 19
    assert Path(result.artifacts[0].path).is_file()

    geographic = make_geo_raster(
        settings.project_root / "geographic.tif", crs="EPSG:4326",
        transform=from_origin(103, 36, 0.01, 0.01),
    )
    result = await executor.execute("get_raster_coordinate", {
        "raster_path": str(geographic), "row": 2, "col": 3,
    })
    assert result.success
    assert result.data["longitude"] == pytest.approx(103.035)
    assert result.data["latitude"] == pytest.approx(35.975)
    assert result.data["projected_coordinate"] == pytest.approx(
        {"x": 103.035, "y": 35.975}
    )


@pytest.mark.anyio
async def test_coordinate_requires_crs_and_valid_pixel(settings):
    executor = geo_executor(settings)
    no_crs = make_geo_raster(settings.project_root / "no-crs.tif", crs=None)
    missing = await executor.execute("get_raster_coordinate", {
        "raster_path": str(no_crs), "row": 0, "col": 0,
    })
    assert not missing.success and missing.error.code == "CRS_REQUIRED"
    utm = make_geo_raster(settings.project_root / "bounds.tif")
    outside = await executor.execute("get_raster_coordinate", {
        "raster_path": str(utm), "row": 4, "col": 0,
    })
    assert not outside.success and outside.error.code == "INVALID_PIXEL"


@pytest.mark.anyio
async def test_export_geojson_bbox_preserves_crs_properties_and_provenance(settings):
    raster = make_geo_raster(settings.project_root / "bbox.tif")
    result = await geo_executor(settings).execute("export_geojson", {
        "raster_path": str(raster), "bbox": [1, 1, 3, 3],
        "coordinate_space": "pixel", "properties": {"class": "field"},
    })
    assert result.success and result.artifacts[0].kind == "geojson"
    payload = json.loads(Path(result.artifacts[0].path).read_text(encoding="utf-8"))
    assert payload["type"] == "FeatureCollection"
    assert payload["crs"]["properties"]["name"] == "EPSG:32647"
    feature = payload["features"][0]
    assert feature["properties"] == {"class": "field", "source_type": "bbox"}
    assert feature["geometry"]["type"] == "Polygon"
    assert feature["geometry"]["coordinates"][0] == [
        [500010.0, 1999990.0], [500030.0, 1999990.0],
        [500030.0, 1999970.0], [500010.0, 1999970.0],
        [500010.0, 1999990.0],
    ]
    assert result.data["provenance"]["coordinate_space"] == "pixel"


@pytest.mark.anyio
async def test_export_geojson_from_raster_aligned_mask(settings):
    raster = make_geo_raster(settings.project_root / "mask-source.tif")
    mask = settings.project_root / "mask.png"
    values = np.zeros((4, 4), dtype=np.uint8)
    values[1:3, 1:3] = 255
    Image.fromarray(values).save(mask)
    result = await geo_executor(settings).execute("export_geojson", {
        "raster_path": str(raster), "mask_path": str(mask),
    })
    assert result.success
    assert result.data["geometry_type"] == "Polygon"
    assert result.data["provenance"]["selected_pixel_count"] == 4


@pytest.mark.anyio
async def test_calculate_known_projected_and_geographic_area(settings):
    executor = geo_executor(settings)
    projected = await executor.execute("calculate_area", {
        "polygon": [[500000, 2000000], [500100, 2000000],
                    [500100, 1999950], [500000, 1999950]],
        "coordinate_space": "projected", "crs": "EPSG:32647",
    })
    assert projected.success
    assert projected.data["area_m2"] == pytest.approx(5000.0)
    assert projected.data["area_ha"] == pytest.approx(0.5)
    assert projected.data["calculation_method"] == "projected_planar"
    assert Path(projected.artifacts[0].path).is_file()

    geographic = await executor.execute("calculate_area", {
        "polygon": [[103.0, 36.0], [103.001, 36.0],
                    [103.001, 35.999], [103.0, 35.999]],
        "coordinate_space": "geographic",
    })
    assert geographic.success
    assert 9000 < geographic.data["area_m2"] < 11000
    assert geographic.data["calculation_method"] == "geographic_to_local_utm"


@pytest.mark.anyio
async def test_zonal_statistics_for_known_polygon(settings):
    raster = make_geo_raster(
        settings.project_root / "zone.tif", crs="EPSG:32647",
        transform=from_origin(0, 4, 1, 1),
        values=np.arange(1, 17, dtype=np.float32).reshape(4, 4),
    )
    result = await geo_executor(settings).execute("zonal_statistics", {
        "raster_path": str(raster),
        "polygon": [[0, 4], [2, 4], [2, 2], [0, 2]],
        "coordinate_space": "projected", "crs": "EPSG:32647",
        "bands": [1],
    })
    assert result.success
    band = result.data["bands"][0]
    assert band["count"] == 4
    assert band["min"] == 1 and band["max"] == 6
    assert band["mean"] == pytest.approx(3.5)
    assert band["std"] == pytest.approx(np.std([1, 2, 5, 6]))
    assert result.data["read_strategy"] == "block_windows_geometry_mask"
    assert result.data["has_overlap"] is True


@pytest.mark.anyio
async def test_phase8_cpu_agent_workflow_and_artifact_provenance(settings):
    raster = make_geo_raster(settings.project_root / "agent-spatial.tif")
    planner = ScriptedPlanner([
        tool_call("inspect_raster", {"raster_path": "active_raster"}),
        tool_call("get_raster_coordinate", {
            "raster_path": "active_raster", "row": 0, "col": 0,
        }),
        tool_call("export_geojson", {
            "raster_path": "active_raster", "bbox": [0, 0, 2, 2],
            "coordinate_space": "pixel", "properties": {"region": "top-left"},
        }),
        tool_call("calculate_area", {"vector_path": "active_vector"}),
        '{"type":"final","answer":"已完成空间定位和真实面积计算。"}',
    ])
    manager = FakeAgentManager(settings)
    registry, executor, _ = build_tool_system(
        settings, manager, logging.getLogger("phase8-agent")
    )
    agent = VisionAgent(
        registry, executor, manager, AgentTraceStore(50),
        logging.getLogger("phase8-agent"), planner=planner,
    )
    response = await agent.run(AgentRequest(
        message="计算影像左上 2×2 像素区域的坐标与真实面积",
        image_path=str(raster), max_steps=6,
    ))
    assert response.success
    assert [step.tool_name for step in response.steps if step.tool_name] == [
        "inspect_raster", "get_raster_coordinate", "export_geojson", "calculate_area",
    ]
    assert response.workflow.active_vector_artifact_id == "vector-001"
    assert response.workflow.active_analysis_result_artifact_id == "analysis-result-002"
    vector = next(item for item in response.workflow.artifacts if item.artifact_id == "vector-001")
    assert vector.artifact_type == "vector" and vector.role == "vector_source"
    assert vector.parent_artifact_id == "raster-001"
    area = response.workflow.spatial_results[-1]
    assert area["tool"] == "calculate_area"
    assert area["data"]["area_m2"] == pytest.approx(400.0)
    assert "真实面积" in response.answer
    assert manager.load_calls == 1


def test_phase8_registry_is_cpu_only():
    tools = {item["name"]: item for item in build_tool_registry().list_tools()}
    for name in (
        "get_raster_coordinate", "export_geojson", "calculate_area", "zonal_statistics",
    ):
        assert name in tools
        assert tools[name]["requires_gpu"] is False
        assert tools[name]["requires_model"] is None


def test_phase8_tool_api_coordinate_area_and_zonal(client):
    payload = geo_raster_bytes()
    coordinate = client.post(
        "/api/v1/tools/get_raster_coordinate/execute",
        files={"image": ("geo.tif", payload, "image/tiff")},
        data={"row": "0", "col": "0"},
    )
    assert coordinate.status_code == 200
    assert coordinate.json()["data"]["projected_coordinate"] == {
        "x": 500005.0, "y": 1999995.0,
    }

    area = client.post(
        "/api/v1/tools/calculate_area/execute",
        data={
            "polygon": "[[500000,2000000],[500100,2000000],[500100,1999950],[500000,1999950]]",
            "coordinate_space": "projected", "crs": "EPSG:32647",
        },
    )
    assert area.status_code == 200
    assert area.json()["data"]["area_m2"] == pytest.approx(5000.0)

    vector = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:32647"}},
        "features": [{
            "type": "Feature", "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [[
                [500000, 2000000], [500020, 2000000], [500020, 1999980],
                [500000, 1999980], [500000, 2000000],
            ]]},
        }],
    }
    zonal = client.post(
        "/api/v1/tools/zonal_statistics/execute",
        files={
            "image": ("geo.tif", payload, "image/tiff"),
            "vector": ("zone.geojson", json.dumps(vector).encode(), "application/geo+json"),
        },
        data={"bands": "[1]"},
    )
    assert zonal.status_code == 200
    assert zonal.json()["data"]["bands"][0]["count"] == 4
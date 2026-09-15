from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Literal

import anyio
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom

from backend.app.geo_errors import (
    CrsRequiredError,
    CrsTransformError,
    GeoError,
    ZonalStatisticsError,
)
from backend.app.raster import validate_raster_path
from backend.app.raster_errors import BandNotFoundError
from backend.app.schemas.tool_result import Artifact, ToolResult
from backend.app.tools.base import BaseTool
from backend.app.tools.geo import build_geometry, load_geojson
from backend.app.tools.raster import _validated_bands


class ZonalStatisticsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    raster_path: Path = Field(description="Raster artifact ID or GeoTIFF path.")
    vector_path: Path | None = Field(default=None, description="GeoJSON vector artifact ID or path.")
    bbox: list[float] | None = Field(
        default=None,
        description="Optional bbox when a GeoJSON vector is not supplied.",
    )
    polygon: list[list[float]] | None = Field(
        default=None,
        description="Optional polygon when a GeoJSON vector is not supplied.",
    )
    coordinate_space: Literal["pixel", "projected", "geographic"] = "projected"
    crs: str | None = None
    bands: list[int] | None = Field(default=None, description="One-based bands; defaults to all.")
    all_touched: bool = False

    @model_validator(mode="after")
    def one_zone_source(self):
        supplied = sum(value is not None for value in (self.vector_path, self.bbox, self.polygon))
        if supplied != 1:
            raise ValueError("Provide exactly one of vector_path, bbox, or polygon")
        return self


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _zonal_statistics(inputs: ZonalStatisticsInput, context, target: Path) -> dict:
    raster_path = validate_raster_path(context, inputs.raster_path)
    if inputs.vector_path is not None:
        geometries, vector_crs, _ = load_geojson(inputs.vector_path, context)
        source_type = "vector"
    else:
        geometry, vector_crs, provenance = build_geometry(
            context=context, raster_path=raster_path, bbox=inputs.bbox,
            polygon=inputs.polygon, mask_path=None,
            coordinate_space=inputs.coordinate_space, crs=inputs.crs,
        )
        geometries = [geometry]
        source_type = provenance["source_type"]

    try:
        with rasterio.open(raster_path) as src:
            if src.crs is None:
                raise CrsRequiredError("Raster CRS is required for zonal statistics")
            if vector_crs != src.crs:
                try:
                    geometries = [
                        transform_geom(vector_crs, src.crs, geometry, precision=15)
                        for geometry in geometries
                    ]
                except Exception as exc:
                    raise CrsTransformError("Vector geometry could not be transformed to raster CRS") from exc
            bands = _validated_bands(src.count, inputs.bands)
            statistics = []
            total_blocks = 0
            for band in bands:
                count = 0
                total = 0.0
                total_squares = 0.0
                minimum = math.inf
                maximum = -math.inf
                block_count = 0
                for _, window in src.block_windows(band):
                    values = src.read(band, window=window, masked=True)
                    zone = geometry_mask(
                        geometries,
                        out_shape=(int(window.height), int(window.width)),
                        transform=src.window_transform(window),
                        invert=True,
                        all_touched=inputs.all_touched,
                    )
                    raw = np.asarray(values.data)
                    valid_mask = zone & ~np.ma.getmaskarray(values) & np.isfinite(raw)
                    valid = np.asarray(raw[valid_mask], dtype=np.float64)
                    if valid.size:
                        count += int(valid.size)
                        total += float(valid.sum(dtype=np.float64))
                        total_squares += float(np.square(valid).sum(dtype=np.float64))
                        minimum = min(minimum, float(valid.min()))
                        maximum = max(maximum, float(valid.max()))
                    block_count += 1
                total_blocks += block_count
                if count:
                    mean = total / count
                    variance = max(0.0, total_squares / count - mean * mean)
                    values_out = {
                        "min": minimum,
                        "max": maximum,
                        "mean": mean,
                        "std": math.sqrt(variance),
                    }
                else:
                    values_out = {"min": None, "max": None, "mean": None, "std": None}
                statistics.append({
                    "band": band,
                    "name": src.descriptions[band - 1] or f"Band {band}",
                    "dtype": src.dtypes[band - 1],
                    "count": count,
                    **values_out,
                    "block_count": block_count,
                })
            data = {
                "raster_crs": src.crs.to_string(),
                "vector_crs": vector_crs.to_string(),
                "source_type": source_type,
                "bands": statistics,
                "all_touched": inputs.all_touched,
                "read_strategy": "block_windows_geometry_mask",
                "total_blocks_read": total_blocks,
                "has_overlap": any(item["count"] > 0 for item in statistics),
            }
    except (BandNotFoundError, GeoError):
        raise
    except rasterio.errors.RasterioIOError as exc:
        raise ZonalStatisticsError("Raster could not be opened for zonal statistics") from exc
    except (OSError, ValueError) as exc:
        raise ZonalStatisticsError("Zonal statistics calculation failed") from exc
    _write_json(target, data)
    return data


class ZonalStatisticsTool(BaseTool):
    name = "zonal_statistics"
    description = (
        "Calculate count, mean, min, max, and standard deviation for raster cells "
        "inside a GeoJSON, bbox, or polygon zone, with CRS transformation and block reads."
    )
    category = "spatial_analysis"
    input_schema = ZonalStatisticsInput

    async def execute(self, inputs, context, execution_id):
        target = context.tool_output_dir(
            execution_id, datetime.now(timezone.utc).strftime("%Y%m%d")
        ) / "zonal-statistics.json"
        data = await anyio.to_thread.run_sync(_zonal_statistics, inputs, context, target)
        return ToolResult(
            success=True, tool=self.name, data=data,
            artifacts=[Artifact(kind="json", path=str(target), mime_type="application/json")],
        )
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Literal

import anyio
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
import rasterio
from rasterio.crs import CRS
from rasterio.features import shapes
from rasterio.warp import transform, transform_geom

from backend.app.geo_errors import (
    AreaCalculationError,
    CrsRequiredError,
    CrsTransformError,
    InvalidGeometryError,
    InvalidPixelError,
    InvalidVectorError,
)
from backend.app.raster import validate_raster_path
from backend.app.schemas.tool_result import Artifact, ToolResult
from backend.app.tools.base import BaseTool

CoordinateSpace = Literal["pixel", "projected", "geographic"]


class GetRasterCoordinateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    raster_path: Path = Field(description="Raster artifact ID or GeoTIFF path.")
    row: int = Field(ge=0, description="Zero-based raster row.")
    col: int = Field(ge=0, description="Zero-based raster column.")


class ExportGeoJSONInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    raster_path: Path | None = Field(
        default=None,
        description="Raster artifact used to georeference pixel bbox, polygon, or mask.",
    )
    bbox: list[float] | None = Field(
        default=None,
        description="[min_x, min_y, max_x, max_y]; pixel values mean [col_min,row_min,col_max,row_max].",
    )
    polygon: list[list[float]] | None = Field(
        default=None,
        description="Polygon outer-ring vertices as [x,y], or [col,row] in pixel space.",
    )
    mask_path: Path | None = Field(
        default=None,
        description="Binary mask artifact whose dimensions exactly match raster dimensions.",
    )
    coordinate_space: CoordinateSpace = "pixel"
    crs: str | None = Field(
        default=None,
        description="CRS for projected coordinates; geographic coordinates use EPSG:4326.",
    )
    properties: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def one_geometry_source(self):
        supplied = sum(value is not None for value in (self.bbox, self.polygon, self.mask_path))
        if supplied != 1:
            raise ValueError("Provide exactly one of bbox, polygon, or mask_path")
        if self.mask_path is not None and self.raster_path is None:
            raise ValueError("mask_path requires raster_path for georeferencing")
        if self.coordinate_space == "pixel" and self.raster_path is None:
            raise ValueError("Pixel geometry requires raster_path")
        return self


class CalculateAreaInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vector_path: Path | None = Field(default=None, description="GeoJSON vector artifact ID or path.")
    raster_path: Path | None = Field(
        default=None, description="Raster artifact used to georeference pixel geometry."
    )
    bbox: list[float] | None = None
    polygon: list[list[float]] | None = None
    mask_path: Path | None = None
    coordinate_space: CoordinateSpace = "pixel"
    crs: str | None = None

    @model_validator(mode="after")
    def one_area_source(self):
        supplied = sum(
            value is not None for value in (
                self.vector_path, self.bbox, self.polygon, self.mask_path,
            )
        )
        if supplied != 1:
            raise ValueError("Provide exactly one of vector_path, bbox, polygon, or mask_path")
        if self.mask_path is not None and self.raster_path is None:
            raise ValueError("mask_path requires raster_path")
        if self.vector_path is None and self.coordinate_space == "pixel" and self.raster_path is None:
            raise ValueError("Pixel geometry requires raster_path")
        return self


def _output_dir(context, execution_id: str) -> Path:
    return context.tool_output_dir(
        execution_id, datetime.now(timezone.utc).strftime("%Y%m%d")
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _crs(value) -> CRS:
    try:
        result = CRS.from_user_input(value)
    except Exception as exc:
        raise CrsRequiredError(f"Invalid or unsupported CRS: {value}") from exc
    if not result:
        raise CrsRequiredError("A valid CRS is required for spatial calculation")
    return result


def _close_ring(points: list[list[float]]) -> list[list[float]]:
    if len(points) < 3:
        raise InvalidGeometryError("A polygon requires at least three vertices")
    ring: list[list[float]] = []
    for point in points:
        if len(point) != 2:
            raise InvalidGeometryError("Every polygon vertex must contain exactly two values")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise InvalidGeometryError("Polygon coordinates must be finite")
        ring.append([x, y])
    if ring[0] != ring[-1]:
        ring.append(list(ring[0]))
    if len(ring) < 4:
        raise InvalidGeometryError("A polygon ring requires at least four closed coordinates")
    if _ring_area(ring) == 0:
        raise InvalidGeometryError("Polygon area must be greater than zero")
    return ring


def _bbox_ring(bbox: list[float]) -> list[list[float]]:
    if len(bbox) != 4:
        raise InvalidGeometryError("bbox must contain exactly four values")
    left, top_or_bottom, right, bottom_or_top = (float(value) for value in bbox)
    if not all(math.isfinite(value) for value in (left, top_or_bottom, right, bottom_or_top)):
        raise InvalidGeometryError("bbox coordinates must be finite")
    low_y, high_y = sorted((top_or_bottom, bottom_or_top))
    if right <= left or high_y <= low_y:
        raise InvalidGeometryError("bbox must have positive width and height")
    return [[left, low_y], [right, low_y], [right, high_y], [left, high_y], [left, low_y]]


def _ring_area(ring: list[list[float]]) -> float:
    return abs(sum(
        ring[index][0] * ring[index + 1][1]
        - ring[index + 1][0] * ring[index][1]
        for index in range(len(ring) - 1)
    )) / 2.0


def _project_pixel_ring(ring: list[list[float]], affine) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in (affine * (col, row) for col, row in ring)]


def _validate_geometry(geometry: dict) -> dict:
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if kind == "Polygon" and isinstance(coordinates, list) and coordinates:
        rings = [_close_ring(ring) for ring in coordinates]
        return {"type": "Polygon", "coordinates": rings}
    if kind == "MultiPolygon" and isinstance(coordinates, list) and coordinates:
        polygons = []
        for polygon in coordinates:
            if not isinstance(polygon, list) or not polygon:
                raise InvalidGeometryError("MultiPolygon contains an empty polygon")
            polygons.append([_close_ring(ring) for ring in polygon])
        return {"type": "MultiPolygon", "coordinates": polygons}
    raise InvalidGeometryError("Only Polygon and MultiPolygon geometry is supported")


def _mask_geometry(mask_path: Path, raster_path: Path, context) -> tuple[dict, CRS, dict]:
    mask_file = context.validate_read_path(mask_path)
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise CrsRequiredError("Raster CRS is required to georeference a mask")
        with Image.open(mask_file) as image:
            mask = np.asarray(image.convert("L")) > 0
        if mask.shape != (src.height, src.width):
            raise InvalidGeometryError(
                f"Mask dimensions {mask.shape[1]}x{mask.shape[0]} do not match "
                f"raster dimensions {src.width}x{src.height}"
            )
        polygons = [
            geometry["coordinates"]
            for geometry, value in shapes(
                mask.astype(np.uint8), mask=mask, transform=src.transform
            )
            if int(value) == 1
        ]
        if not polygons:
            raise InvalidGeometryError("Mask contains no selected pixels")
        geometry = (
            {"type": "Polygon", "coordinates": polygons[0]}
            if len(polygons) == 1
            else {"type": "MultiPolygon", "coordinates": polygons}
        )
        return _validate_geometry(geometry), src.crs, {
            "source_type": "mask",
            "mask_width": int(mask.shape[1]),
            "mask_height": int(mask.shape[0]),
            "selected_pixel_count": int(mask.sum()),
        }


def build_geometry(
    *, context, raster_path: Path | None, bbox: list[float] | None,
    polygon: list[list[float]] | None, mask_path: Path | None,
    coordinate_space: CoordinateSpace, crs: str | None,
) -> tuple[dict, CRS, dict]:
    raster_file = validate_raster_path(context, raster_path) if raster_path is not None else None
    if mask_path is not None:
        return _mask_geometry(mask_path, raster_file, context)

    ring = _bbox_ring(bbox) if bbox is not None else _close_ring(polygon or [])
    metadata = {"source_type": "bbox" if bbox is not None else "polygon"}
    if coordinate_space == "pixel":
        if raster_file is None:
            raise CrsRequiredError("Pixel geometry requires a georeferenced raster")
        with rasterio.open(raster_file) as src:
            if src.crs is None:
                raise CrsRequiredError("Raster CRS is required for pixel geometry")
            for col, row in ring:
                if col < 0 or row < 0 or col > src.width or row > src.height:
                    raise InvalidGeometryError("Pixel geometry extends outside raster bounds")
            ring = _project_pixel_ring(ring, src.transform)
            source_crs = src.crs
            metadata.update({"coordinate_space": "pixel", "raster_width": src.width, "raster_height": src.height})
    elif coordinate_space == "geographic":
        source_crs = _crs("EPSG:4326")
        metadata["coordinate_space"] = "geographic"
    else:
        if crs is not None:
            source_crs = _crs(crs)
        elif raster_file is not None:
            with rasterio.open(raster_file) as src:
                if src.crs is None:
                    raise CrsRequiredError("Raster CRS is required for projected geometry")
                source_crs = src.crs
        else:
            raise CrsRequiredError("Projected geometry requires crs or raster_path")
        metadata["coordinate_space"] = "projected"
    return _validate_geometry({"type": "Polygon", "coordinates": [ring]}), source_crs, metadata


def load_geojson(path: Path, context) -> tuple[list[dict], CRS, dict]:
    vector_path = context.validate_read_path(path)
    try:
        payload = json.loads(vector_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidVectorError("GeoJSON could not be read") from exc
    crs_value = (
        payload.get("crs", {}).get("properties", {}).get("name")
        if isinstance(payload, dict) else None
    ) or "EPSG:4326"
    source_crs = _crs(crs_value)
    if payload.get("type") == "FeatureCollection":
        features = payload.get("features")
    elif payload.get("type") == "Feature":
        features = [payload]
    elif payload.get("type") in {"Polygon", "MultiPolygon"}:
        features = [{"type": "Feature", "properties": {}, "geometry": payload}]
    else:
        raise InvalidVectorError("GeoJSON must be a FeatureCollection, Feature, Polygon, or MultiPolygon")
    if not isinstance(features, list) or not features:
        raise InvalidVectorError("GeoJSON contains no features")
    geometries = []
    for feature in features:
        geometry = feature.get("geometry") if isinstance(feature, dict) else None
        geometries.append(_validate_geometry(geometry or {}))
    return geometries, source_crs, payload


def _coordinate(path: Path, row: int, col: int) -> dict:
    try:
        with rasterio.open(path) as src:
            if row >= src.height or col >= src.width:
                raise InvalidPixelError(
                    f"Pixel row={row}, col={col} is outside raster {src.height}x{src.width}"
                )
            if src.crs is None:
                raise CrsRequiredError("Raster CRS is required for coordinate conversion")
            x, y = rasterio.transform.xy(src.transform, row, col, offset="center")
            try:
                longitude, latitude = transform(src.crs, "EPSG:4326", [x], [y])
            except Exception as exc:
                raise CrsTransformError("Raster coordinate could not be transformed to EPSG:4326") from exc
            return {
                "pixel": {"row": row, "col": col, "offset": "center"},
                "source_crs": src.crs.to_string(),
                "source_epsg": src.crs.to_epsg(),
                "projected_coordinate": {"x": float(x), "y": float(y)},
                "longitude": float(longitude[0]),
                "latitude": float(latitude[0]),
                "geographic_crs": "EPSG:4326",
            }
    except (CrsRequiredError, CrsTransformError, InvalidPixelError):
        raise
    except rasterio.errors.RasterioIOError as exc:
        raise InvalidVectorError("Raster could not be opened for coordinate conversion") from exc


def _export_geojson(inputs: ExportGeoJSONInput, context, target: Path) -> dict:
    geometry, source_crs, provenance = build_geometry(
        context=context, raster_path=inputs.raster_path, bbox=inputs.bbox,
        polygon=inputs.polygon, mask_path=inputs.mask_path,
        coordinate_space=inputs.coordinate_space, crs=inputs.crs,
    )
    properties = dict(inputs.properties)
    properties.setdefault("source_type", provenance["source_type"])
    payload = {
        "type": "FeatureCollection",
        "name": target.stem,
        "crs": {"type": "name", "properties": {"name": source_crs.to_string()}},
        "features": [{"type": "Feature", "properties": properties, "geometry": geometry}],
    }
    _write_json(target, payload)
    return {
        "geometry_type": geometry["type"],
        "feature_count": 1,
        "crs": source_crs.to_string(),
        "epsg": source_crs.to_epsg(),
        "properties": properties,
        "provenance": provenance,
    }


def _geometry_area_units(geometry: dict) -> float:
    polygons = (
        [geometry["coordinates"]]
        if geometry["type"] == "Polygon" else geometry["coordinates"]
    )
    area = 0.0
    for polygon in polygons:
        outer = _ring_area(polygon[0])
        holes = sum(_ring_area(ring) for ring in polygon[1:])
        area += max(0.0, outer - holes)
    return area


def _geometry_centroid_hint(geometries: list[dict]) -> tuple[float, float]:
    points = []
    for geometry in geometries:
        polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
        for polygon in polygons:
            points.extend(polygon[0][:-1])
    if not points:
        raise AreaCalculationError("Geometry contains no area coordinates")
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def calculate_geometries_area(geometries: list[dict], source_crs: CRS) -> tuple[float, str, str]:
    try:
        if source_crs.is_projected:
            unit = source_crs.linear_units_factor
            factor = float(unit[1] if isinstance(unit, tuple) else unit)
            area = sum(_geometry_area_units(item) for item in geometries) * factor * factor
            method = "projected_planar"
            area_crs = source_crs.to_string()
        elif source_crs.is_geographic:
            lon, lat = _geometry_centroid_hint(geometries)
            zone = max(1, min(60, int((lon + 180.0) // 6.0) + 1))
            target_crs = CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)
            projected = [transform_geom(source_crs, target_crs, item, precision=15) for item in geometries]
            area = sum(_geometry_area_units(_validate_geometry(item)) for item in projected)
            method = "geographic_to_local_utm"
            area_crs = target_crs.to_string()
        else:
            raise CrsRequiredError("CRS must be projected or geographic")
    except (CrsRequiredError, InvalidGeometryError):
        raise
    except Exception as exc:
        raise AreaCalculationError("Geometry area calculation failed") from exc
    if not math.isfinite(area) or area <= 0:
        raise AreaCalculationError("Calculated area must be finite and greater than zero")
    return float(area), method, area_crs


def _calculate_area(inputs: CalculateAreaInput, context, target: Path) -> dict:
    if inputs.vector_path is not None:
        geometries, source_crs, _ = load_geojson(inputs.vector_path, context)
        source_type = "vector"
    else:
        geometry, source_crs, provenance = build_geometry(
            context=context, raster_path=inputs.raster_path, bbox=inputs.bbox,
            polygon=inputs.polygon, mask_path=inputs.mask_path,
            coordinate_space=inputs.coordinate_space, crs=inputs.crs,
        )
        geometries = [geometry]
        source_type = provenance["source_type"]
    area_m2, method, area_crs = calculate_geometries_area(geometries, source_crs)
    data = {
        "area_m2": area_m2,
        "area_ha": area_m2 / 10000.0,
        "area_km2": area_m2 / 1_000_000.0,
        "source_crs": source_crs.to_string(),
        "area_crs": area_crs,
        "calculation_method": method,
        "geometry_count": len(geometries),
        "source_type": source_type,
    }
    _write_json(target, data)
    return data


class GetRasterCoordinateTool(BaseTool):
    name = "get_raster_coordinate"
    description = (
        "Convert a zero-based raster pixel center (row, col) through the raster affine "
        "transform and CRS, returning source coordinates plus EPSG:4326 longitude/latitude."
    )
    category = "geospatial"
    input_schema = GetRasterCoordinateInput

    async def execute(self, inputs, context, execution_id):
        path = validate_raster_path(context, inputs.raster_path)
        data = await anyio.to_thread.run_sync(_coordinate, path, inputs.row, inputs.col)
        target = _output_dir(context, execution_id) / "coordinate-analysis.json"
        await anyio.to_thread.run_sync(_write_json, target, data)
        return ToolResult(
            success=True, tool=self.name, data=data,
            artifacts=[Artifact(kind="json", path=str(target), mime_type="application/json")],
        )


class ExportGeoJSONTool(BaseTool):
    name = "export_geojson"
    description = (
        "Export exactly one bbox, polygon, or raster-aligned binary mask as a GeoJSON "
        "FeatureCollection with CRS, properties, and spatial provenance."
    )
    category = "geospatial"
    input_schema = ExportGeoJSONInput

    async def execute(self, inputs, context, execution_id):
        target = _output_dir(context, execution_id) / "spatial-result.geojson"
        data = await anyio.to_thread.run_sync(_export_geojson, inputs, context, target)
        return ToolResult(
            success=True, tool=self.name, data=data,
            artifacts=[Artifact(kind="geojson", path=str(target), mime_type="application/geo+json")],
        )


class CalculateAreaTool(BaseTool):
    name = "calculate_area"
    description = (
        "Calculate real polygon or mask area from its CRS. Projected CRS linear units are "
        "honored; geographic geometry is transformed to its local UTM CRS before area calculation."
    )
    category = "geospatial"
    input_schema = CalculateAreaInput

    async def execute(self, inputs, context, execution_id):
        target = _output_dir(context, execution_id) / "area-analysis.json"
        data = await anyio.to_thread.run_sync(_calculate_area, inputs, context, target)
        return ToolResult(
            success=True, tool=self.name, data=data,
            artifacts=[Artifact(kind="json", path=str(target), mime_type="application/json")],
        )
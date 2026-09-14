from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import rasterio
from pydantic import BaseModel, ConfigDict, Field

from backend.app.raster_errors import InvalidRasterError, RasterOpenFailedError

if TYPE_CHECKING:
    from backend.app.tools.context import ToolContext

RASTER_SUFFIXES = {".tif", ".tiff"}
TIFF_HEADERS = {b"II*\x00", b"MM\x00*"}


class RasterBounds(BaseModel):
    model_config = ConfigDict(extra="forbid")
    left: float
    bottom: float
    right: float
    top: float


class RasterMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    band_count: int = Field(gt=0)
    dtypes: list[str]
    driver: str
    crs: str | None
    epsg: int | None
    transform: list[float] = Field(min_length=6, max_length=6)
    resolution_x: float = Field(gt=0)
    resolution_y: float = Field(gt=0)
    bounds: RasterBounds
    nodata: float | str | None
    band_descriptions: list[str]


def is_raster_candidate(path: Path) -> bool:
    if path.suffix.lower() in RASTER_SUFFIXES:
        return True
    try:
        with path.open("rb") as stream:
            return stream.read(4) in TIFF_HEADERS
    except OSError:
        return False


def validate_raster_path(context: ToolContext, value: Path) -> Path:
    try:
        return context.validate_read_path(value)
    except (FileNotFoundError, ValueError) as exc:
        raise RasterOpenFailedError(str(exc)) from exc


def _nodata_value(value) -> float | str | None:
    if value is None:
        return None
    number = float(value)
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "Infinity" if number > 0 else "-Infinity"
    return number


def metadata_from_dataset(src) -> RasterMetadata:
    descriptions = [
        description.strip() if description and description.strip() else f"Band {index}"
        for index, description in enumerate(src.descriptions, start=1)
    ]
    crs = src.crs.to_string() if src.crs else None
    epsg = src.crs.to_epsg() if src.crs else None
    transform = src.transform
    return RasterMetadata(
        width=src.width,
        height=src.height,
        band_count=src.count,
        dtypes=list(src.dtypes),
        driver=src.driver,
        crs=crs,
        epsg=epsg,
        transform=[
            float(transform.a), float(transform.b), float(transform.c),
            float(transform.d), float(transform.e), float(transform.f),
        ],
        resolution_x=abs(float(src.res[0])),
        resolution_y=abs(float(src.res[1])),
        bounds=RasterBounds(
            left=float(src.bounds.left),
            bottom=float(src.bounds.bottom),
            right=float(src.bounds.right),
            top=float(src.bounds.top),
        ),
        nodata=_nodata_value(src.nodata),
        band_descriptions=descriptions,
    )


def read_raster_metadata(path: Path) -> RasterMetadata:
    try:
        with rasterio.open(path) as src:
            if src.driver != "GTiff":
                raise InvalidRasterError(
                    f"Expected a GeoTIFF dataset, received raster driver {src.driver}"
                )
            if src.width <= 0 or src.height <= 0 or src.count <= 0:
                raise InvalidRasterError("Raster dimensions and band count must be positive")
            return metadata_from_dataset(src)
    except (InvalidRasterError, RasterOpenFailedError):
        raise
    except rasterio.errors.RasterioIOError as exc:
        raise RasterOpenFailedError("GeoTIFF could not be opened") from exc
    except (OSError, ValueError) as exc:
        raise InvalidRasterError("GeoTIFF metadata is invalid") from exc

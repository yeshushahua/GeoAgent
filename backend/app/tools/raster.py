from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Literal

import anyio
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

from backend.app.raster import read_raster_metadata, validate_raster_path
from backend.app.schemas.tool_result import Artifact, ToolResult
from backend.app.tools.base import BaseTool
from backend.app.raster_errors import (
    BandNotFoundError,
    EmptyWindowError,
    InvalidRasterError,
    InvalidWindowError,
    PreviewFailedError,
    RasterOpenFailedError,
    UnsupportedRasterDtypeError,
)


class InspectRasterInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    raster_path: Path = Field(
        description="Workflow raster artifact ID (preferred) or a GeoTIFF reference."
    )


class RasterPreviewInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    raster_path: Path = Field(
        description="Workflow raster artifact ID (preferred) or a GeoTIFF reference."
    )
    bands: list[int] | None = Field(
        default=None,
        description=(
            "One-based band indexes. Defaults to band 1 for one-band rasters and "
            "bands 1,2,3 for rasters with at least three bands."
        ),
    )
    stretch: Literal["percentile", "minmax"] = "percentile"
    lower_percentile: float = Field(default=2.0, ge=0, lt=100)
    upper_percentile: float = Field(default=98.0, gt=0, le=100)
    max_size: int = Field(default=2048, ge=64, le=2048)
    resampling: Literal["nearest", "bilinear"] = "bilinear"


class CropRasterInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    raster_path: Path = Field(
        description="Workflow raster artifact ID (preferred) or a GeoTIFF reference."
    )
    region: Literal[
        "left_half", "right_half", "top_half", "bottom_half",
        "top_left_quarter", "top_right_quarter",
        "bottom_left_quarter", "bottom_right_quarter",
    ] | None = Field(
        default=None,
        description=(
            "Named half or quadrant. Prefer this for a user-named region so the Tool "
            "derives exact row/column boundaries from current raster dimensions."
        ),
    )
    row_start: int | None = Field(
        default=None, ge=0, description="Inclusive top row in raster pixel space."
    )
    row_end: int | None = Field(
        default=None, ge=0, description="Exclusive bottom row in raster pixel space."
    )
    col_start: int | None = Field(
        default=None, ge=0, description="Inclusive left column in raster pixel space."
    )
    col_end: int | None = Field(
        default=None, ge=0, description="Exclusive right column in raster pixel space."
    )

    @model_validator(mode="after")
    def one_window_mode(self):
        coordinates = (self.row_start, self.row_end, self.col_start, self.col_end)
        if self.region is not None and any(value is not None for value in coordinates):
            raise ValueError("Use either region or explicit pixel coordinates, not both")
        if self.region is None and any(value is None for value in coordinates):
            raise ValueError("Provide region or all four pixel window coordinates")
        return self


class RasterStatisticsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    raster_path: Path = Field(
        description="Workflow raster artifact ID (preferred) or a GeoTIFF reference."
    )
    bands: list[int] | None = Field(
        default=None,
        description="Optional one-based band indexes; all bands are used when omitted.",
    )


def _open_error() -> RasterOpenFailedError:
    return RasterOpenFailedError("GeoTIFF could not be opened")


def _validated_bands(count: int, bands: list[int] | None, *, preview: bool = False) -> list[int]:
    if bands is None:
        if preview:
            if count == 1:
                return [1]
            if count >= 3:
                return [1, 2, 3]
            raise BandNotFoundError(
                "A two-band raster requires an explicit one-band or three-band preview selection"
            )
        return list(range(1, count + 1))
    if preview:
        if len(bands) not in {1, 3}:
            raise BandNotFoundError("Raster preview requires exactly one or three bands")
    elif not bands:
        raise BandNotFoundError("At least one statistics band is required")
    invalid = sorted({band for band in bands if band < 1 or band > count})
    if invalid:
        raise BandNotFoundError(
            f"Requested bands {invalid} are outside the available range 1..{count}"
        )
    return list(bands)


def _stretch_channel(channel, stretch: str, lower: float, upper: float) -> np.ndarray:
    array = np.asarray(channel.data, dtype=np.float64)
    valid = ~np.ma.getmaskarray(channel) & np.isfinite(array)
    values = array[valid]
    if values.size == 0:
        raise PreviewFailedError("Selected band contains no finite valid pixels")
    if stretch == "percentile":
        low, high = np.percentile(values, [lower, upper])
    else:
        low, high = float(values.min()), float(values.max())
    output = np.zeros(array.shape, dtype=np.uint8)
    if not math.isfinite(float(low)) or not math.isfinite(float(high)):
        raise PreviewFailedError("Stretch range is not finite")
    if high <= low:
        output[valid] = 127
        return output
    scaled = np.clip((array[valid] - low) / (high - low), 0, 1)
    output[valid] = np.rint(scaled * 255).astype(np.uint8)
    return output


def _inspect(path: Path) -> dict:
    return read_raster_metadata(path).model_dump(mode="json")


def _preview(path: Path, inputs: RasterPreviewInput, target: Path) -> dict:
    if inputs.upper_percentile <= inputs.lower_percentile:
        raise PreviewFailedError("upper_percentile must be greater than lower_percentile")
    try:
        with rasterio.open(path) as src:
            if src.driver != "GTiff":
                raise InvalidRasterError("Raster preview currently supports GeoTIFF only")
            bands = _validated_bands(src.count, inputs.bands, preview=True)
            for band in bands:
                dtype = np.dtype(src.dtypes[band - 1])
                if not np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.complexfloating):
                    raise UnsupportedRasterDtypeError(
                        f"Band {band} dtype {dtype} cannot be rendered safely"
                    )
            scale = min(1.0, inputs.max_size / max(src.width, src.height))
            width = max(1, int(round(src.width * scale)))
            height = max(1, int(round(src.height * scale)))
            method = Resampling.nearest if inputs.resampling == "nearest" else Resampling.bilinear
            data = src.read(
                bands,
                out_shape=(len(bands), height, width),
                resampling=method,
                masked=True,
            )
            channels = [
                _stretch_channel(item, inputs.stretch, inputs.lower_percentile, inputs.upper_percentile)
                for item in data
            ]
            if len(channels) == 1:
                image = Image.fromarray(channels[0], mode="L")
            else:
                image = Image.fromarray(np.stack(channels, axis=-1), mode="RGB")
            try:
                image.save(target, "PNG")
            finally:
                image.close()
            return {
                "source_width": src.width,
                "source_height": src.height,
                "preview_width": width,
                "preview_height": height,
                "bands": bands,
                "band_descriptions": [
                    src.descriptions[band - 1] or f"Band {band}" for band in bands
                ],
                "stretch": inputs.stretch,
                "percentile_range": (
                    [inputs.lower_percentile, inputs.upper_percentile]
                    if inputs.stretch == "percentile" else None
                ),
                "resampling": inputs.resampling,
                "read_strategy": "downsampled_out_shape",
            }
    except (BandNotFoundError, InvalidRasterError, PreviewFailedError, UnsupportedRasterDtypeError):
        raise
    except rasterio.errors.RasterioIOError as exc:
        raise _open_error() from exc
    except (OSError, ValueError) as exc:
        raise PreviewFailedError("Raster preview generation failed") from exc


def _named_window(region: str, width: int, height: int) -> tuple[int, int, int, int]:
    mid_row, mid_col = height // 2, width // 2
    windows = {
        "left_half": (0, height, 0, mid_col),
        "right_half": (0, height, mid_col, width),
        "top_half": (0, mid_row, 0, width),
        "bottom_half": (mid_row, height, 0, width),
        "top_left_quarter": (0, mid_row, 0, mid_col),
        "top_right_quarter": (0, mid_row, mid_col, width),
        "bottom_left_quarter": (mid_row, height, 0, mid_col),
        "bottom_right_quarter": (mid_row, height, mid_col, width),
    }
    return windows[region]


def _crop(path: Path, inputs: CropRasterInput, target: Path) -> dict:
    try:
        with rasterio.open(path) as src:
            if inputs.region is not None:
                row_start, row_end, col_start, col_end = _named_window(
                    inputs.region, src.width, src.height
                )
            else:
                row_start = int(inputs.row_start)
                row_end = int(inputs.row_end)
                col_start = int(inputs.col_start)
                col_end = int(inputs.col_end)
            if row_end <= row_start or col_end <= col_start:
                raise EmptyWindowError("Raster crop window must have positive width and height")
            if row_end > src.height or col_end > src.width:
                raise InvalidWindowError(
                    f"Window exceeds raster bounds width={src.width}, height={src.height}"
                )
            window = Window(
                col_off=col_start,
                row_off=row_start,
                width=col_end - col_start,
                height=row_end - row_start,
            )
            data = src.read(window=window)
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                width=int(window.width),
                height=int(window.height),
                transform=src.window_transform(window),
            )
            descriptions = src.descriptions
            dataset_tags = src.tags()
            band_tags = [src.tags(index) for index in src.indexes]
            with rasterio.open(target, "w", **profile) as dst:
                dst.write(data)
                if dataset_tags:
                    dst.update_tags(**dataset_tags)
                for index, description in enumerate(descriptions, start=1):
                    if description:
                        dst.set_band_description(index, description)
                    if band_tags[index - 1]:
                        dst.update_tags(index, **band_tags[index - 1])
        metadata = read_raster_metadata(target)
        return {
            "window": {
                "row_start": row_start,
                "row_end": row_end,
                "col_start": col_start,
                "col_end": col_end,
                "region": inputs.region,
            },
            "metadata": metadata.model_dump(mode="json"),
            "read_strategy": "pixel_window",
        }
    except (EmptyWindowError, InvalidWindowError, InvalidRasterError, RasterOpenFailedError):
        raise
    except rasterio.errors.RasterioIOError as exc:
        raise _open_error() from exc
    except (OSError, ValueError) as exc:
        raise InvalidRasterError("Raster crop could not be written as GeoTIFF") from exc


def _statistics(path: Path, inputs: RasterStatisticsInput) -> dict:
    try:
        with rasterio.open(path) as src:
            bands = _validated_bands(src.count, inputs.bands)
            output = []
            total_blocks = 0
            for band in bands:
                count = 0
                nodata_count = 0
                total = 0.0
                total_squares = 0.0
                minimum = math.inf
                maximum = -math.inf
                block_count = 0
                for _, window in src.block_windows(band):
                    values = src.read(band, window=window, masked=True)
                    raw = np.asarray(values.data)
                    valid_mask = ~np.ma.getmaskarray(values) & np.isfinite(raw)
                    valid = np.asarray(raw[valid_mask], dtype=np.float64)
                    nodata_count += int(raw.size - valid.size)
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
                    stats = {
                        "min": minimum,
                        "max": maximum,
                        "mean": mean,
                        "std": math.sqrt(variance),
                    }
                else:
                    stats = {"min": None, "max": None, "mean": None, "std": None}
                output.append({
                    "band": band,
                    "name": src.descriptions[band - 1] or f"Band {band}",
                    "dtype": src.dtypes[band - 1],
                    **stats,
                    "valid_pixel_count": count,
                    "nodata_count": nodata_count,
                    "block_count": block_count,
                })
            return {
                "width": src.width,
                "height": src.height,
                "bands": output,
                "read_strategy": "block_windows",
                "total_blocks_read": total_blocks,
            }
    except BandNotFoundError:
        raise
    except rasterio.errors.RasterioIOError as exc:
        raise _open_error() from exc
    except (OSError, ValueError) as exc:
        raise InvalidRasterError("Raster statistics failed") from exc


class InspectRasterTool(BaseTool):
    name = "inspect_raster"
    description = (
        "Read GeoTIFF metadata without loading pixel arrays: dimensions, bands, dtypes, "
        "driver, CRS, affine transform, resolution, bounds, NoData, and band descriptions."
    )
    category = "remote_sensing"
    input_schema = InspectRasterInput

    async def execute(self, inputs, context, execution_id):
        path = validate_raster_path(context, inputs.raster_path)
        data = await anyio.to_thread.run_sync(_inspect, path)
        return ToolResult(success=True, tool=self.name, data=data)


class RasterPreviewTool(BaseTool):
    name = "raster_preview"
    description = (
        "Create a downsampled PNG visualization of one or three one-based GeoTIFF bands. "
        "The default is a 2-98 percentile stretch with a 2048-pixel maximum edge."
    )
    category = "remote_sensing"
    input_schema = RasterPreviewInput

    async def execute(self, inputs, context, execution_id):
        path = validate_raster_path(context, inputs.raster_path)
        output_dir = context.tool_output_dir(
            execution_id, datetime.now(timezone.utc).strftime("%Y%m%d")
        )
        target = output_dir / "raster-preview.png"
        data = await anyio.to_thread.run_sync(_preview, path, inputs, target)
        return ToolResult(
            success=True,
            tool=self.name,
            data=data,
            artifacts=[Artifact(kind="image", path=str(target), mime_type="image/png")],
        )


class CropRasterTool(BaseTool):
    name = "crop_raster"
    description = (
        "Strictly crop a GeoTIFF by a named half/quadrant or an inclusive/exclusive "
        "pixel window while preserving CRS, bands, dtype, NoData, resolution, and "
        "an updated affine transform."
    )
    category = "remote_sensing"
    input_schema = CropRasterInput

    async def execute(self, inputs, context, execution_id):
        path = validate_raster_path(context, inputs.raster_path)
        output_dir = context.tool_output_dir(
            execution_id, datetime.now(timezone.utc).strftime("%Y%m%d")
        )
        target = output_dir / "raster-crop.tif"
        data = await anyio.to_thread.run_sync(_crop, path, inputs, target)
        return ToolResult(
            success=True,
            tool=self.name,
            data=data,
            artifacts=[Artifact(kind="geotiff", path=str(target), mime_type="image/tiff")],
        )


class RasterStatisticsTool(BaseTool):
    name = "raster_statistics"
    description = (
        "Compute exact per-band min, max, mean, standard deviation, valid pixels, and "
        "NoData pixels through raster block windows using original un-stretched values."
    )
    category = "remote_sensing"
    input_schema = RasterStatisticsInput

    async def execute(self, inputs, context, execution_id):
        path = validate_raster_path(context, inputs.raster_path)
        data = await anyio.to_thread.run_sync(_statistics, path, inputs)
        return ToolResult(success=True, tool=self.name, data=data)

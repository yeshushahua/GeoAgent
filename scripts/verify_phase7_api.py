"""Live FastAPI acceptance for Phase 7 GeoTIFF upload and raster workflow."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import rasterio
from rasterio.transform import from_origin

from backend.app.core.config import get_settings


def make_phase7_raster(path: Path) -> Path:
    width, height = 256, 192
    rows, cols = np.indices((height, width))
    data = np.stack([
        (rows + 20).astype(np.uint16),
        (cols + 100).astype(np.uint16),
        (rows + cols + 200).astype(np.uint16),
    ])
    data[:, :4, :] = 0
    data[:, :, :4] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=3,
        dtype="uint16", crs="EPSG:32647",
        transform=from_origin(500000, 2000000, 10, 10), nodata=0,
        tiled=True, blockxsize=64, blockysize=64, compress="deflate",
    ) as dst:
        dst.write(data)
        for index, name in enumerate(("Red", "Green", "Blue"), start=1):
            dst.set_band_description(index, name)
    return path


def main() -> None:
    settings = get_settings()
    source = make_phase7_raster(settings.temp_dir / "phase7-live" / "live-source.tif")
    upload = (source.name, source.read_bytes(), "image/tiff")
    task = "裁剪影像左上四分之一区域，并统计各波段。"
    with httpx.Client(timeout=1800, trust_env=False) as client:
        health = client.get(f"{settings.api_base_url}/health")
        health.raise_for_status()
        tools = client.get(f"{settings.api_base_url}/tools")
        tools.raise_for_status()
        inspected = client.post(
            f"{settings.api_base_url}/tools/inspect_raster/execute",
            files={"image": upload},
        )
        inspected.raise_for_status()
        previewed = client.post(
            f"{settings.api_base_url}/tools/raster_preview/execute",
            files={"image": upload},
            data={"bands": "[1,2,3]", "max_size": "128", "stretch": "percentile"},
        )
        previewed.raise_for_status()
        response = client.post(
            f"{settings.api_base_url}/agent/run",
            files={"image": upload},
            data={"message": task, "max_steps": "6", "max_new_tokens": "128"},
        )
        response.raise_for_status()

    body = response.json()
    sequence = [step["tool_name"] for step in body["steps"] if step["tool_name"]]
    metadata = inspected.json()["data"]
    preview = previewed.json()
    assert health.json()["version"] == settings.app_version
    assert {"inspect_raster", "raster_preview", "crop_raster", "raster_statistics"}.issubset(
        item["name"] for item in tools.json()
    )
    assert metadata["width"] == 256 and metadata["height"] == 192
    assert metadata["crs"] == "EPSG:32647" and metadata["resolution_x"] == 10
    assert preview["success"] and preview["data"]["preview_width"] == 128
    assert Path(preview["artifacts"][0]["path"]).is_file()
    assert body["success"] and sequence == [
        "inspect_raster", "crop_raster", "raster_statistics"
    ]
    workflow = body["workflow"]
    assert workflow["original_raster_artifact_id"] == "raster-001"
    assert workflow["active_raster_artifact_id"] == "raster-crop-001"
    assert not workflow["active_image_artifact_id"]
    raster = workflow["raster"]
    assert raster["metadata"]["width"] == 128 and raster["metadata"]["height"] == 96
    assert raster["metadata"]["crs"] == "EPSG:32647"
    assert raster["metadata"]["resolution_x"] == raster["metadata"]["resolution_y"] == 10
    assert raster["metadata"]["transform"] == [10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0]
    assert raster["metadata"]["bounds"] == {
        "left": 500000.0, "bottom": 1999040.0,
        "right": 501280.0, "top": 2000000.0,
    }
    assert raster["statistics"]["read_strategy"] == "block_windows"
    assert all("path" not in artifact for artifact in workflow["artifacts"])

    target = settings.output_dir / "benchmarks" / "phase7" / "live_api_acceptance.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "health": health.json(), "task": task, "metadata": metadata,
        "preview": preview, "sequence": sequence, "response": body,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "sequence": sequence,
        "active_raster": workflow["active_raster_artifact_id"],
        "crop_size": [raster["metadata"]["width"], raster["metadata"]["height"]],
        "peak_vram_gib": body["metadata"]["peak_vram_gib"],
        "output": str(target),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

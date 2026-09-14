"""Short real RTX cross-phase smoke for ordinary vision and GeoTIFF analysis."""
from __future__ import annotations

import json
from pathlib import Path
import time

from fastapi.testclient import TestClient
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from backend.app.core.config import get_settings
from backend.app.main import create_app

pytestmark = pytest.mark.integration


def _raster(path: Path) -> Path:
    height, width = 96, 128
    data = np.zeros((3, height, width), dtype=np.uint8)
    data[0], data[1], data[2] = 75, 140, 65
    data[:, 28:35, :] = 180
    data[:, :, 60:66] = 35
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=3,
        dtype="uint8", crs="EPSG:32647",
        transform=from_origin(500000, 2000000, 10, 10), nodata=0,
    ) as dst:
        dst.write(data)
    return path


def _sequence(body: dict) -> list[str]:
    return [
        step["tool_name"] for step in body["steps"]
        if step["tool_name"] and step["success"]
    ]


def test_cross_phase_gpu_smoke():
    settings = get_settings()
    image = settings.project_root / "sample_data" / "images" / "foundation-demo.png"
    raster = _raster(settings.temp_dir / "phase7-smoke" / "cross-phase.tif")
    started = time.perf_counter()
    with TestClient(create_app(settings)) as client:
        client.post("/api/v1/models/vlm/unload").raise_for_status()

        print("CROSS_PHASE_SMOKE_START vision", flush=True)
        vision_response = client.post(
            "/api/v1/agent/run",
            files={"image": (image.name, image.read_bytes(), "image/png")},
            data={"message": "用一句话描述这张图片的主要内容。", "max_steps": "4",
                  "max_new_tokens": "64"},
        )
        vision_response.raise_for_status()
        vision = vision_response.json()
        assert vision["success"] and _sequence(vision) == ["analyze_image"], vision
        print("CROSS_PHASE_SMOKE_PASS vision", flush=True)

        print("CROSS_PHASE_SMOKE_START raster_visual", flush=True)
        raster_response = client.post(
            "/api/v1/agent/run",
            files={"image": (raster.name, raster.read_bytes(), "image/tiff")},
            data={"message": "检查这幅遥感影像并描述主要地物。", "max_steps": "6",
                  "max_new_tokens": "64"},
        )
        raster_response.raise_for_status()
        raster_body = raster_response.json()
        assert raster_body["success"] and _sequence(raster_body) == [
            "inspect_raster", "raster_preview", "analyze_image"
        ], raster_body
        assert raster_body["workflow"]["active_raster_artifact_id"] == "raster-001"
        assert raster_body["workflow"]["active_image_artifact_id"] == "raster-preview-001"
        print("CROSS_PHASE_SMOKE_PASS raster_visual", flush=True)

        model = client.get("/api/v1/models/vlm/status").json()
        assert model["state"] == "READY" and model["load_count"] == 1
        detector = client.get("/api/v1/models/detector/status").json()
        open_vocab = client.get("/api/v1/models/open-vocabulary/status").json()
        segmentation = client.get("/api/v1/models/segmentation/status").json()
        assert detector["load_count"] == open_vocab["load_count"] == segmentation["load_count"] == 0
        client.post("/api/v1/models/vlm/unload").raise_for_status()

    report = {
        "elapsed_s": round(time.perf_counter() - started, 2),
        "vision_sequence": _sequence(vision),
        "raster_sequence": _sequence(raster_body),
        "qwen_load_count": model["load_count"],
        "unused_model_load_counts": {
            "yolo11": detector["load_count"],
            "yoloe": open_vocab["load_count"],
            "sam": segmentation["load_count"],
        },
        "vision": vision,
        "raster": raster_body,
    }
    target = settings.output_dir / "benchmarks" / "phase7" / "cross_phase_smoke.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"CROSS_PHASE_SMOKE_ELAPSED_S {report['elapsed_s']}", flush=True)

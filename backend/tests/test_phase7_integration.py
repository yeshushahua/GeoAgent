"""Real RTX 4090 acceptance for Phase 7 remote-sensing workflows."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import anyio
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from backend.app.agent.schemas import AgentRequest
from backend.app.agent.service import VisionAgent
from backend.app.agent.trace import AgentTraceStore
from backend.app.core.config import get_settings
from backend.app.detection import DetectorManager
from backend.app.models.manager import ModelManager
from backend.app.open_vocabulary import OpenVocabularyDetectorManager
from backend.app.segmentation import SegmentationManager
from backend.app.services.storage import prepare_storage
from backend.app.tools import build_tool_system

pytestmark = pytest.mark.integration


def _synthetic_remote_sensing_raster(path: Path) -> Path:
    width, height = 256, 192
    rgb = np.zeros((3, height, width), dtype=np.uint8)
    rgb[0], rgb[1], rgb[2] = 70, 145, 65
    rgb[:, 0:4, :] = 0
    rgb[:, :, 0:4] = 0
    rgb[0, 70:110, :] = 35
    rgb[1, 70:110, :] = 95
    rgb[2, 70:110, :] = 190
    rgb[:, :, 122:130] = 185
    rgb[:, 36:43, :] = 180
    for row, col in ((20, 30), (22, 70), (125, 155), (135, 195)):
        rgb[0, row:row + 25, col:col + 35] = 175
        rgb[1, row:row + 25, col:col + 35] = 120
        rgb[2, row:row + 25, col:col + 35] = 80
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=3,
        dtype="uint8",
        crs="EPSG:32647",
        transform=from_origin(500000, 2000000, 10, 10),
        nodata=0,
        tiled=True,
        blockxsize=64,
        blockysize=64,
        compress="deflate",
    ) as dst:
        dst.write(rgb)
        for index, name in enumerate(("Red", "Green", "Blue"), start=1):
            dst.set_band_description(index, name)
    return path


def _sequence(response) -> list[str]:
    return [step.tool_name for step in response.steps if step.tool_name]


def test_real_phase7_remote_sensing_workflows_reuse_qwen():
    settings = get_settings()
    prepare_storage(settings)
    source = _synthetic_remote_sensing_raster(
        settings.temp_dir / "phase7-integration" / "synthetic-rgb.tif"
    )
    logger = logging.getLogger("phase7-integration")
    qwen = ModelManager(settings)
    closed = DetectorManager(settings, logger)
    open_vocab = OpenVocabularyDetectorManager(settings, logger)
    segmentation = SegmentationManager(settings, logger)
    registry, executor, _ = build_tool_system(
        settings, qwen, logger, closed, open_vocab, segmentation
    )
    agent = VisionAgent(
        registry, executor, qwen, AgentTraceStore(100), logger,
        repair_attempts=settings.agent_repair_attempts,
        planner_max_new_tokens=settings.agent_planner_max_new_tokens,
    )
    cases = {
        "metadata": "检查这幅遥感影像的基本信息。",
        "preview": "生成这幅遥感影像的 RGB 预览。",
        "visual": "检查这幅影像，并描述主要地物。",
        "crop_statistics": "裁剪影像左上四分之一区域，并统计各波段。",
    }
    results = {}
    try:
        suite_started = time.perf_counter()
        for name, message in cases.items():
            case_started = time.perf_counter()
            print(f"PHASE7_CASE_START {name}", flush=True)
            response = anyio.run(agent.run, AgentRequest(
                message=message, image_path=str(source), max_steps=6, max_new_tokens=128
            ))
            assert response.success, response.model_dump(mode="json")
            sequence = _sequence(response)
            assert sequence[0] == "inspect_raster", response.model_dump(mode="json")
            if name == "metadata":
                assert sequence == ["inspect_raster"]
            elif name == "preview":
                assert sequence == ["inspect_raster", "raster_preview"]
                assert response.workflow.active_image_artifact_id == "raster-preview-001"
            elif name == "visual":
                assert sequence == ["inspect_raster", "raster_preview", "analyze_image"]
                analyze = next(step for step in response.steps if step.tool_name == "analyze_image")
                assert analyze.arguments_summary["image_path"] == "raster-preview-001"
                assert response.workflow.active_raster_artifact_id == "raster-001"
            else:
                assert sequence == ["inspect_raster", "crop_raster", "raster_statistics"]
                crop_step = next(step for step in response.steps if step.tool_name == "crop_raster")
                assert crop_step.arguments_summary["region"] == "top_left_quarter"
                assert response.workflow.active_raster_artifact_id == "raster-crop-001"
                raster = response.workflow.raster
                assert raster.artifact_id == "raster-crop-001"
                assert raster.metadata["width"] == 128 and raster.metadata["height"] == 96
                assert raster.metadata["crs"] == "EPSG:32647"
                assert raster.metadata["resolution_x"] == raster.metadata["resolution_y"] == 10
                assert raster.metadata["transform"] == [
                    10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0
                ]
                assert raster.metadata["bounds"] == {
                    "left": 500000.0,
                    "bottom": 1999040.0,
                    "right": 501280.0,
                    "top": 2000000.0,
                }
                assert raster.statistics["read_strategy"] == "block_windows"
                assert len(raster.statistics["bands"]) == 3
            assert response.workflow.original_raster_artifact_id == "raster-001"
            assert response.metadata.workflow_total_ms == response.metadata.total_duration_ms
            results[name] = response.model_dump(mode="json")
            print(
                f"PHASE7_CASE_PASS {name} elapsed_s={time.perf_counter() - case_started:.2f} "
                f"sequence={_sequence(response)}",
                flush=True,
            )
        print(f"PHASE7_SUITE_ELAPSED_S {time.perf_counter() - suite_started:.2f}", flush=True)
        assert qwen.state.value == "READY"
        assert closed.load_count == 0
        assert open_vocab.load_count == 0
        assert segmentation.load_count == 0
        target = settings.output_dir / "benchmarks" / "phase7" / "integration.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "source": str(source),
            "qwen_state": qwen.state.value,
            "unused_model_load_counts": {
                "yolo11": closed.load_count,
                "yoloe": open_vocab.load_count,
                "sam": segmentation.load_count,
            },
            "cases": results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        segmentation.unload_model()
        open_vocab.unload_model()
        closed.unload_model()
        if qwen.state.value not in {"UNLOADED", "LOADING"}:
            qwen.unload_model()

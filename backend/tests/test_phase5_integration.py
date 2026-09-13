"""Real RTX 4090 acceptance for YOLOE + SAM and Qwen-driven Phase 5 workflows."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import anyio
import pytest
import torch

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


def _source(settings) -> Path:
    path = settings.dataset_dir / "phase5" / "construction-ppe" / "images" / "test" / "image40.jpg"
    assert path.is_file(), f"Phase 5 acceptance image is missing: {path}"
    return path


def _boxes(result: dict) -> list[dict]:
    return [item["bbox"] for item in result["data"]["detections"]]


def _write_report(settings, name: str, payload: dict) -> Path:
    target = settings.output_dir / "benchmarks" / "phase5" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def _deny_network(*args, **kwargs):
    raise AssertionError("Network access attempted during Phase 5 offline acceptance")


def test_real_models_offline_five_workflows_and_combined_vram(monkeypatch):
    settings = get_settings()
    prepare_storage(settings)
    source = _source(settings)
    monkeypatch.setattr("socket.create_connection", _deny_network)
    monkeypatch.setattr("requests.sessions.Session.request", _deny_network)

    logger = logging.getLogger("phase5-integration")
    qwen = ModelManager(settings)
    closed = DetectorManager(settings, logger)
    open_vocab = OpenVocabularyDetectorManager(settings, logger)
    segmentation = SegmentationManager(settings, logger)
    registry, executor, _ = build_tool_system(
        settings, qwen, logger, closed, open_vocab, segmentation
    )

    torch.cuda.empty_cache()
    try:
        qwen_status = qwen.load_model()
        closed_status = closed.load_model()
        open_status = open_vocab.load_model()
        sam_status = segmentation.load_model()
        torch.cuda.reset_peak_memory_stats(0)
        records = []
        last_detection = None
        last_segmentation = None
        for index in range(5):
            detected = anyio.run(
                executor.execute,
                "detect_open_vocab",
                {
                    "image_path": str(source),
                    "classes": ["yellow safety helmet"],
                    "confidence": 0.25,
                },
            )
            assert detected.success and detected.data["detection_count"] == 1
            segmented = anyio.run(
                executor.execute,
                "segment_objects",
                {"image_path": str(source), "boxes": _boxes(detected.model_dump())},
            )
            assert segmented.success and segmented.data["segment_count"] == 1
            assert segmented.data["segments"][0]["mask_area_pixels"] > 0
            records.append({
                "iteration": index + 1,
                "detection": detected.model_dump(mode="json"),
                "segmentation": segmented.model_dump(mode="json"),
                "allocated_vram_gib": round(torch.cuda.memory_allocated(0) / 2**30, 3),
                "peak_vram_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 3),
            })
            last_detection, last_segmentation = detected, segmented

        cropped = anyio.run(
            executor.execute,
            "crop_image",
            {"image_path": str(source), "x1": 0, "y1": 0, "x2": 320, "y2": 320},
        )
        assert cropped.success
        crop_path = cropped.artifacts[0].path
        crop_detection = anyio.run(
            executor.execute,
            "detect_open_vocab",
            {
                "image_path": crop_path,
                "classes": ["yellow safety helmet"],
                "confidence": 0.25,
            },
        )
        assert crop_detection.success and crop_detection.data["detection_count"] == 1
        crop_segmentation = anyio.run(
            executor.execute,
            "segment_objects",
            {"image_path": crop_path, "boxes": _boxes(crop_detection.model_dump())},
        )
        assert crop_segmentation.success and crop_segmentation.data["image_width"] == 320
        missing = anyio.run(
            executor.execute,
            "detect_open_vocab",
            {
                "image_path": str(source),
                "classes": ["bright purple submarine floating above the workers"],
            },
        )
        assert missing.success and missing.data["detection_count"] == 0
        assert qwen.state.value == "READY"
        assert closed.state.value == "READY"
        assert open_vocab.load_count == 1
        assert segmentation.load_count == 1
        allocations = [row["allocated_vram_gib"] for row in records]
        assert max(allocations) - min(allocations) < 0.25
        assert max(row["peak_vram_gib"] for row in records) < 23.5
        report = {
            "offline_network_blocked": True,
            "source": str(source),
            "models": {
                "qwen": qwen_status,
                "closed_detector": closed_status,
                "open_vocabulary": open_status,
                "segmentation": sam_status,
            },
            "five_workflows": records,
            "crop": {
                "artifact": crop_path,
                "detection": crop_detection.model_dump(mode="json"),
                "segmentation": crop_segmentation.model_dump(mode="json"),
            },
            "zero_target": missing.model_dump(mode="json"),
            "final_artifacts": [
                item.model_dump(mode="json")
                for item in (last_detection.artifacts + last_segmentation.artifacts)
            ],
        }
        _write_report(settings, "model_validation.json", report)
    finally:
        segmentation.unload_model()
        open_vocab.unload_model()
        closed.unload_model()
        if qwen.state.value != "LOADING":
            qwen.unload_model()


@pytest.mark.parametrize(
    ("message", "expected_tools"),
    [
        ("检测图中的黄色安全帽。", ["detect_open_vocab"]),
        (
            "找到图中的黄色安全帽，并把它们精确分割出来。",
            ["detect_open_vocab", "segment_objects"],
        ),
        (
            "裁剪左上四分之一区域，在里面寻找黄色安全帽，并把找到的目标分割出来。",
            ["inspect_image", "crop_image", "detect_open_vocab", "segment_objects"],
        ),
        (
            "找出图中明亮紫色潜水艇；如果找到就把它精确分割出来。",
            ["detect_open_vocab"],
        ),
    ],
)
def test_real_qwen_agent_phase5_scenarios(message, expected_tools):
    settings = get_settings()
    prepare_storage(settings)
    logger = logging.getLogger("phase5-agent-integration")
    qwen = ModelManager(settings)
    closed = DetectorManager(settings, logger)
    open_vocab = OpenVocabularyDetectorManager(settings, logger)
    segmentation = SegmentationManager(settings, logger)
    registry, executor, _ = build_tool_system(
        settings, qwen, logger, closed, open_vocab, segmentation
    )
    agent = VisionAgent(
        registry, executor, qwen, AgentTraceStore(50), logger,
        repair_attempts=settings.agent_repair_attempts,
        planner_max_new_tokens=settings.agent_planner_max_new_tokens,
    )
    try:
        response = anyio.run(
            agent.run,
            AgentRequest(
                message=message,
                image_path=str(_source(settings)),
                max_steps=6,
                max_new_tokens=128,
            ),
        )
        assert response.success, response.model_dump(mode="json")
        sequence = [step.tool_name for step in response.steps if step.tool_name]
        assert sequence == expected_tools
        if "segment_objects" in sequence:
            detect_step = next(
                step for step in response.steps if step.tool_name == "detect_open_vocab"
            )
            segment_step = next(
                step for step in response.steps if step.tool_name == "segment_objects"
            )
            boxes = [item["bbox"] for item in detect_step.observation_summary["detections"]]
            assert segment_step.arguments_summary["boxes"] == boxes
            assert segment_step.observation_summary["segment_count"] == len(boxes)
            if "crop_image" in sequence:
                assert detect_step.arguments_summary["image_path"] == "crop.png"
                assert segment_step.arguments_summary["image_path"] == "crop.png"
        if response.steps[0].tool_name == "detect_open_vocab" and "紫色潜水艇" in message:
            assert response.steps[0].observation_summary["detection_count"] == 0
        target = settings.output_dir / "benchmarks" / "phase5" / "agent_scenarios.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "message": message,
                "expected_tools": expected_tools,
                "response": response.model_dump(mode="json"),
            }, ensure_ascii=False) + "\n")
    finally:
        segmentation.unload_model()
        open_vocab.unload_model()
        closed.unload_model()
        if qwen.state.value != "LOADING":
            qwen.unload_model()

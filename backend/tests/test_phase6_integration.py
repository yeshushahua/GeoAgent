"""Real RTX 4090 acceptance for Phase 6 multi-step workflow state."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import anyio
import pytest

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
    user_test = settings.project_root / "test" / "img" / "1.jpg"
    if user_test.is_file():
        return user_test
    return settings.dataset_dir / "phase5" / "construction-ppe" / "images" / "test" / "image40.jpg"


def test_real_phase6_workflows_reuse_one_model_set():
    settings = get_settings()
    prepare_storage(settings)
    source = _source(settings)
    assert source.is_file()
    logger = logging.getLogger("phase6-integration")
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
        "case1": "先检查图片，然后找到图中的所有黄色安全帽和反光背心。",
        "case2": "找到图中的所有黄色安全帽和反光背心，并把它们分别分割出来。",
        "case3": (
            "先检查图片尺寸，然后裁剪左半部分，在裁剪结果中找到所有黄色安全帽和"
            "反光背心，并将它们分割出来，最后总结每类目标数量和面积占比。"
        ),
        "case4": "找到黄色安全帽、反光背心和挖掘机，并将找到的目标分割出来。",
    }
    results = {}
    try:
        for name, message in cases.items():
            response = anyio.run(agent.run, AgentRequest(
                message=message, image_path=str(source), max_steps=6, max_new_tokens=128
            ))
            assert response.success, response.model_dump(mode="json")
            sequence = [step.tool_name for step in response.steps if step.tool_name]
            assert "detect_open_vocab" in sequence
            if name == "case1":
                assert "inspect_image" in sequence and "segment_objects" not in sequence
            else:
                assert "segment_objects" in sequence
            if name == "case3":
                assert "crop_image" in sequence
                detect = next(step for step in response.steps if step.tool_name == "detect_open_vocab")
                segment = next(step for step in response.steps if step.tool_name == "segment_objects")
                assert detect.arguments_summary["image_path"] == "crop-001"
                assert segment.arguments_summary["image_path"] == "crop-001"
                assert response.workflow.active_image_artifact_id == "crop-001"
            if "segment_objects" in sequence:
                for segment in response.workflow.segmentations:
                    assert segment.source_detection_id is not None
                    detection = next(
                        item for item in response.workflow.detections
                        if item.detection_id == segment.source_detection_id
                    )
                    assert detection.source_artifact_id == segment.source_artifact_id
            if name == "case4":
                assert any(item.detected == 0 for item in response.workflow.categories.values())
                assert any(item.detected > 0 for item in response.workflow.categories.values())
            assert response.metadata.workflow_total_ms == response.metadata.total_duration_ms
            results[name] = response.model_dump(mode="json")
        assert qwen.state.value == "READY"
        assert open_vocab.load_count == 1
        assert segmentation.load_count == 1
        assert open_vocab.status()["device"] == segmentation.status()["device"] == "cuda:0"
        target = settings.output_dir / "benchmarks" / "phase6" / "integration.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "source": str(source),
            "model_reuse": {
                "open_vocab_load_count": open_vocab.load_count,
                "segmentation_load_count": segmentation.load_count,
            },
            "cases": results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        segmentation.unload_model()
        open_vocab.unload_model()
        closed.unload_model()
        if qwen.state.value not in {"UNLOADED", "LOADING"}:
            qwen.unload_model()

import json
import logging
from pathlib import Path

from PIL import Image
import pytest

from backend.app.agent.planner import AgentPlannerTurn
from backend.app.agent.prompts import AGENT_SYSTEM_PROMPT
from backend.app.agent.schemas import AgentRequest
from backend.app.agent.service import VisionAgent
from backend.app.agent.trace import AgentTraceStore
from backend.app.models.manager import ModelState
from backend.app.detection.schemas import BoundingBox, Detection
from backend.app.schemas.inference import GenerationInfo, GpuMemory, ImageInfo, InferenceResult
from backend.app.tools import build_tool_system
from backend.tests.test_detect_objects_tool import FakeDetector
from backend.app.open_vocabulary.schemas import OpenVocabularyDetection
from backend.tests.test_phase5_tools import (
    FakeOpenVocabularyManager,
    FakeSegmentationManager,
)


class FakeAgentManager:
    def __init__(self, settings):
        self.settings = settings
        self.state = ModelState.UNLOADED
        self.load_calls = 0

    def load_model(self):
        self.load_calls += 1
        self.state = ModelState.READY
        return self.status()

    def status(self):
        return {"state": self.state.value, "last_error": None}

    def infer(self, image, prompt, max_new_tokens):
        return InferenceResult(
            success=True, model="Qwen3-VL-4B-Instruct", text="裁剪区域包含蓝色图形。",
            latency_ms=10, device="cuda:0", dtype="bfloat16",
            image=ImageInfo(
                width=image.width, height=image.height, mode=image.mode,
                format=image.format or "PNG", preprocessing_width=image.width,
                preprocessing_height=image.height, resized=False,
            ),
            generation=GenerationInfo(max_new_tokens=max_new_tokens),
            gpu=GpuMemory(allocated_gb=8.2, reserved_gb=8.3, peak_allocated_gb=8.4),
        )


class ScriptedPlanner:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    async def decide(self, state, repair_output=None, repair_error=None):
        self.calls.append({
            "step": state.step_count,
            "observations": len(state.observations),
            "repair": repair_output is not None,
            "definitions": [item.name for item in state.tool_definitions],
        })
        output = self.outputs.pop(0)
        if callable(output):
            output = output(state)
        return AgentPlannerTurn(output, 2.0, 8.2, 8.4)


def source_image(settings, size=(40, 20)):
    path = settings.project_root / "agent.png"
    Image.new("RGB", size, "blue").save(path, "PNG")
    return path


def make_agent(
    settings, planner, detector_manager=None,
    open_vocab_manager=None, segmentation_manager=None,
):
    manager = FakeAgentManager(settings)
    registry, executor, _ = build_tool_system(
        settings, manager, logging.getLogger("test"), detector_manager,
        open_vocab_manager, segmentation_manager,
    )
    traces = AgentTraceStore(50)
    return VisionAgent(
        registry, executor, manager, traces, logging.getLogger("test"),
        repair_attempts=1, planner=planner,
    ), manager, traces


def tool_call(name, arguments):
    return json.dumps({"type": "tool_call", "tool_name": name, "arguments": arguments})


@pytest.mark.anyio
async def test_dynamic_discovery_single_tool_and_final(settings):
    path = source_image(settings)
    planner = ScriptedPlanner([
        tool_call("inspect_image", {"image_path": str(path)}),
        '{"type":"final","answer":"图像尺寸为 40 × 20 像素，格式为 PNG。"}',
    ])
    agent, manager, traces = make_agent(settings, planner)
    response = await agent.run(AgentRequest(message="告诉我尺寸", image_path=str(path)))
    assert response.success and response.answer.startswith("图像尺寸")
    assert [step.tool_name for step in response.steps if step.tool_name] == ["inspect_image"]
    assert manager.load_calls == 1
    assert planner.calls[0]["definitions"] == [
        "analyze_image", "crop_image", "crop_raster", "detect_objects",
        "detect_open_vocab", "inspect_image", "inspect_raster", "raster_preview",
        "raster_statistics", "segment_objects",
    ]
    trace = traces.list()[0].model_dump()
    assert trace["prompt_length"] == len("告诉我尺寸")
    assert "告诉我尺寸" not in json.dumps(trace, ensure_ascii=False)


@pytest.mark.anyio
async def test_multistep_artifact_propagation(settings):
    path = source_image(settings, (480, 300))

    def crop_from_observation(state):
        inspected = state.observations[-1].result.data
        assert inspected["width"] == 480
        assert inspected["height"] == 300
        return tool_call("crop_image", {
            "image_path": str(path),
            "x1": 0,
            "y1": 0,
            "x2": inspected["width"] // 2,
            "y2": inspected["height"] // 2,
        })

    def analyze_crop(state):
        artifact = state.observations[-1].result.artifacts[0].path
        return tool_call("analyze_image", {
            "image_path": artifact,
            "prompt": "请分析这个裁剪区域。",
            "max_new_tokens": 64,
        })

    planner = ScriptedPlanner([
        tool_call("inspect_image", {"image_path": str(path)}),
        crop_from_observation,
        analyze_crop,
        '{"type":"final","answer":"左上区域包含蓝色图形。"}',
    ])
    agent, _, _ = make_agent(settings, planner)
    response = await agent.run(AgentRequest(
        message="裁剪左上四分之一并分析", image_path=str(path), max_steps=6,
        max_new_tokens=64,
    ))
    assert response.success
    assert [step.tool_name for step in response.steps if step.tool_name] == [
        "inspect_image", "crop_image", "analyze_image"
    ]
    crop_path = response.artifacts[0].path
    assert Path(crop_path).is_file()
    with Image.open(crop_path) as crop:
        assert crop.size == (240, 150)
    crop_arguments = response.steps[1].arguments_summary
    assert crop_arguments == {
        "image_path": "original-image-001", "x1": 0, "y1": 0, "x2": 240, "y2": 150,
    }
    assert (crop_arguments["x2"], crop_arguments["y2"]) != (120, 75)
    assert planner.calls[2]["observations"] == 2
    assert response.steps[2].arguments_summary["image_path"] == "crop-001"


def test_planner_requires_all_explicit_subgoals_before_final():
    required = (
        "Completing one Tool call never completes unrelated explicit goals",
        "original request's distinct requested capabilities",
        "Planner-written prose cannot",
        "replace a missing Tool observation",
        "does not ground broader visual content",
        "at most 80 Chinese characters",
        "ResultAggregator will append",
    )
    assert all(fragment in AGENT_SYSTEM_PROMPT for fragment in required)
    assert "先检测图中的目标" not in AGENT_SYSTEM_PROMPT


def test_spatial_quadrant_policy_and_integer_rule_are_explicit(settings):
    required = (
        '"top-left quarter"',
        "x2=floor(width / 2)",
        "y2=floor(height / 2)",
        "NEVER use width / 4",
        "never copy dimensions from examples",
    )
    assert all(fragment in AGENT_SYSTEM_PROMPT for fragment in required)

    manager = FakeAgentManager(settings)
    registry, _, _ = build_tool_system(settings, manager, logging.getLogger("test"))
    description = registry.get("crop_image").definition()["description"]
    assert "2-by-2 split" in description
    assert "floor(width / 2)" in description
    assert "never width / 4" in description


@pytest.mark.anyio
async def test_agent_detects_objects_and_exposes_structured_observation(settings):
    path = source_image(settings, (100, 60))
    detector = FakeDetector([
        Detection(
            class_id=0, class_name="person", confidence=0.94,
            bbox=BoundingBox(x1=5, y1=4, x2=45, y2=55),
        )
    ])
    planner = ScriptedPlanner([
        tool_call("detect_objects", {"image_path": str(path)}),
        '{"type":"final","answer":"检测到 1 个 person。"}',
    ])
    agent, _, _ = make_agent(settings, planner, detector)
    response = await agent.run(AgentRequest(message="检测目标并计数", image_path=str(path)))
    assert response.success
    assert [step.tool_name for step in response.steps if step.tool_name] == ["detect_objects"]
    observation = response.steps[0].observation_summary
    assert observation["detection_count"] == 1
    assert observation["class_counts"] == {"person": 1}
    assert Path(response.artifacts[0].path).name == "annotated.jpg"


@pytest.mark.anyio
async def test_detection_observation_grounds_analysis(settings):
    path = source_image(settings, (100, 60))
    detector = FakeDetector([
        Detection(
            class_id=5, class_name="bus", confidence=0.88,
            bbox=BoundingBox(x1=10, y1=8, x2=90, y2=55),
        )
    ])

    def analyze_after_detection(state):
        detection = state.observations[-1].result.data
        assert detection["class_counts"] == {"bus": 1}
        return tool_call("analyze_image", {
            "image_path": str(path),
            "prompt": "结合检测结果 bus=1 分析场景。",
            "max_new_tokens": 64,
        })

    planner = ScriptedPlanner([
        tool_call("detect_objects", {"image_path": str(path)}),
        analyze_after_detection,
        '{"type":"final","answer":"检测结果显示一辆公交车。"}',
    ])
    agent, _, _ = make_agent(settings, planner, detector)
    response = await agent.run(AgentRequest(
        message="先检测再结合结果分析", image_path=str(path), max_new_tokens=64
    ))
    assert response.success
    assert [step.tool_name for step in response.steps if step.tool_name] == [
        "detect_objects", "analyze_image"
    ]
    assert response.steps[1].arguments_summary["detection_observation_count"] == 1


@pytest.mark.anyio
async def test_crop_artifact_is_passed_to_detect_objects(settings):
    path = source_image(settings, (480, 300))
    detector = FakeDetector()

    def detect_crop(state):
        artifact = state.observations[-1].result.artifacts[0].path
        return tool_call("detect_objects", {"image_path": artifact})

    planner = ScriptedPlanner([
        tool_call("inspect_image", {"image_path": str(path)}),
        tool_call("crop_image", {
            "image_path": str(path), "x1": 0, "y1": 0, "x2": 240, "y2": 150,
        }),
        detect_crop,
        '{"type":"final","answer":"裁剪区域内未检出 COCO 目标。"}',
    ])
    agent, _, _ = make_agent(settings, planner, detector)
    response = await agent.run(AgentRequest(
        message="裁剪左上四分之一后检测", image_path=str(path), max_steps=6
    ))
    assert response.success
    assert [step.tool_name for step in response.steps if step.tool_name] == [
        "inspect_image", "crop_image", "detect_objects"
    ]
    crop_artifact = response.steps[1].artifacts[0].path
    assert detector.calls[0][0] == Path(crop_artifact)
    assert response.steps[2].arguments_summary["image_path"] == "crop-001"
    assert Path(response.artifacts[-1].path).name == "annotated.jpg"


@pytest.mark.anyio
async def test_open_vocab_bbox_is_passed_to_sam(settings):
    path = source_image(settings, (100, 60))
    bbox = BoundingBox(x1=5, y1=4, x2=45, y2=40)
    open_vocab = FakeOpenVocabularyManager([
        OpenVocabularyDetection(
            detection_id="detection-001", class_name="yellow safety helmet",
            confidence=0.91, bbox=bbox,
        )
    ])
    segmentation = FakeSegmentationManager()

    def segment_detection(state):
        observation = state.observations[-1].result.data
        assert observation["detection_count"] == 1
        return tool_call("segment_objects", {
            "image_path": str(path),
            "boxes": [observation["detections"][0]["bbox"]],
        })

    planner = ScriptedPlanner([
        tool_call("detect_open_vocab", {
            "image_path": str(path), "classes": ["yellow safety helmet"],
        }),
        segment_detection,
        '{"type":"final","answer":"找到并分割了 1 顶黄色安全帽。"}',
    ])
    agent, _, _ = make_agent(
        settings, planner, open_vocab_manager=open_vocab,
        segmentation_manager=segmentation,
    )
    response = await agent.run(AgentRequest(
        message="找到黄色安全帽并精确分割", image_path=str(path)
    ))
    assert response.success
    assert [step.tool_name for step in response.steps if step.tool_name] == [
        "detect_open_vocab", "segment_objects"
    ]
    assert segmentation.calls[0][0] == path
    assert segmentation.calls[0][1] == [bbox]
    assert response.steps[1].arguments_summary["boxes"] == [bbox.model_dump()]
    assert response.steps[1].observation_summary["segment_count"] == 1
    assert response.steps[0].arguments_summary["classes"] == ["yellow safety helmet"]
    assert response.steps[0].arguments_summary["confidence"] == 0.25
    assert response.steps[0].arguments_summary["iou_threshold"] == 0.45
    for step in response.steps:
        assert step.duration_ms == pytest.approx(
            step.planner_duration_ms + step.tool_duration_ms, abs=0.02
        )
    assert response.steps[0].model_inference_duration_ms == 8.0
    assert response.steps[0].prompt_encoding_duration_ms == 12.0
    metrics = response.metadata
    assert metrics.total_duration_ms == pytest.approx(
        metrics.planner_duration_ms + metrics.tool_duration_ms
        + metrics.framework_overhead_ms,
        abs=0.02,
    )
    assert metrics.model_inference_duration_ms == 28.0
    assert metrics.prompt_encoding_duration_ms == 12.0
    assert metrics.model_load_duration_ms == pytest.approx(
        metrics.agent_model_load_duration_ms + metrics.tool_model_load_duration_ms,
        abs=0.02,
    )
    assert metrics.framework_overhead_ms == pytest.approx(
        metrics.agent_model_load_duration_ms + metrics.framework_runtime_overhead_ms,
        abs=0.02,
    )


@pytest.mark.anyio
async def test_agent_and_manual_open_vocab_use_identical_effective_arguments(settings):
    path = source_image(settings, (100, 60))
    detection = OpenVocabularyDetection(
        detection_id="detection-001", class_name="yellow helmet",
        confidence=0.87, bbox=BoundingBox(x1=5, y1=4, x2=45, y2=40),
    )
    open_vocab = FakeOpenVocabularyManager([detection])
    planner = ScriptedPlanner([
        tool_call("detect_open_vocab", {
            "image_path": str(path), "classes": ["yellow helmet"],
        }),
        '{"type":"final","answer":"检测到 1 顶黄色安全帽。"}',
    ])
    agent, _, _ = make_agent(settings, planner, open_vocab_manager=open_vocab)
    response = await agent.run(AgentRequest(
        message="检测图中的 yellow helmet。", image_path=str(path)
    ))
    manual = await agent.executor.execute("detect_open_vocab", {
        "image_path": str(path), "classes": ["yellow helmet"],
        "confidence": 0.25, "iou_threshold": 0.45,
    })
    assert open_vocab.calls[0] == open_vocab.calls[1]
    assert response.steps[0].arguments_summary == {
        "image_path": "original-image-001", "classes": ["yellow helmet"],
        "confidence": 0.25, "iou_threshold": 0.45,
    }
    assert response.steps[0].observation_summary["detections"] == manual.data["detections"]


@pytest.mark.anyio
async def test_crop_propagates_to_open_vocab_and_sam(settings):
    path = source_image(settings, (480, 300))
    bbox = BoundingBox(x1=10, y1=8, x2=80, y2=70)
    open_vocab = FakeOpenVocabularyManager([
        OpenVocabularyDetection(
            detection_id="detection-001", class_name="yellow safety helmet",
            confidence=0.83, bbox=bbox,
        )
    ])
    segmentation = FakeSegmentationManager()

    def detect_crop(state):
        crop_path = state.observations[-1].result.artifacts[0].path
        return tool_call("detect_open_vocab", {
            "image_path": crop_path, "classes": ["yellow safety helmet"],
        })

    def segment_crop(state):
        detection = state.observations[-1].result.data
        crop_path = state.observations[-2].result.artifacts[0].path
        return tool_call("segment_objects", {
            "image_path": crop_path,
            "boxes": [item["bbox"] for item in detection["detections"]],
        })

    planner = ScriptedPlanner([
        tool_call("inspect_image", {"image_path": str(path)}),
        tool_call("crop_image", {
            "image_path": str(path), "x1": 0, "y1": 0, "x2": 240, "y2": 150,
        }),
        detect_crop,
        segment_crop,
        '{"type":"final","answer":"裁剪区域中的目标已分割。"}',
    ])
    agent, _, _ = make_agent(
        settings, planner, open_vocab_manager=open_vocab,
        segmentation_manager=segmentation,
    )
    response = await agent.run(AgentRequest(
        message="裁剪左上四分之一，寻找黄色安全帽并分割", image_path=str(path)
    ))
    assert response.success
    assert [step.tool_name for step in response.steps if step.tool_name] == [
        "inspect_image", "crop_image", "detect_open_vocab", "segment_objects"
    ]
    crop_path = Path(response.steps[1].artifacts[0].path)
    assert open_vocab.calls[0][0] == crop_path
    assert segmentation.calls[0][0] == crop_path
    assert response.steps[2].arguments_summary["image_path"] == "crop-001"
    assert response.steps[3].arguments_summary["image_path"] == "crop-001"


@pytest.mark.anyio
async def test_zero_open_vocab_detection_does_not_call_sam(settings):
    path = source_image(settings, (100, 60))
    open_vocab = FakeOpenVocabularyManager()
    segmentation = FakeSegmentationManager()
    planner = ScriptedPlanner([
        tool_call("detect_open_vocab", {
            "image_path": str(path), "classes": ["invisible violet turbine"],
        }),
        '{"type":"final","answer":"未检测到指定目标。"}',
    ])
    agent, _, _ = make_agent(
        settings, planner, open_vocab_manager=open_vocab,
        segmentation_manager=segmentation,
    )
    response = await agent.run(AgentRequest(
        message="找出不存在的紫色涡轮并分割", image_path=str(path)
    ))
    assert response.success
    assert [step.tool_name for step in response.steps if step.tool_name] == [
        "detect_open_vocab"
    ]
    assert response.steps[0].observation_summary["detection_count"] == 0
    assert segmentation.calls == []


@pytest.mark.anyio
async def test_malformed_unknown_and_invalid_outputs_are_repaired(settings):
    path = source_image(settings)
    for invalid in (
        "not json",
        tool_call("unknown_tool", {}),
        tool_call("crop_image", {"image_path": str(path)}),
    ):
        planner = ScriptedPlanner([
            invalid,
            '{"type":"final","answer":"已安全恢复。"}',
        ])
        agent, _, _ = make_agent(settings, planner)
        response = await agent.run(AgentRequest(message="测试恢复", image_path=str(path)))
        assert response.success and response.answer == "已安全恢复。"
        assert planner.calls[1]["repair"] is True


@pytest.mark.anyio
async def test_duplicate_call_is_blocked_and_observed(settings):
    path = source_image(settings)
    call = tool_call("inspect_image", {"image_path": str(path)})
    planner = ScriptedPlanner([
        call, call, '{"type":"final","answer":"重复调用已停止。"}',
    ])
    agent, _, _ = make_agent(settings, planner)
    response = await agent.run(AgentRequest(
        message="测试重复", image_path=str(path), max_steps=3
    ))
    assert response.success
    assert response.metadata.tool_call_count == 1
    assert response.steps[1].decision_type == "blocked"
    assert response.steps[1].error_type == "DUPLICATE_TOOL_CALL"
    assert planner.calls[2]["observations"] == 2


@pytest.mark.anyio
async def test_max_steps_and_tool_error_are_safe(settings):
    path = source_image(settings)
    planner = ScriptedPlanner([tool_call("inspect_image", {"image_path": str(path)})])
    agent, _, _ = make_agent(settings, planner)
    stopped = await agent.run(AgentRequest(
        message="不能结束", image_path=str(path), max_steps=1
    ))
    assert not stopped.success and stopped.error.type == "MAX_STEPS_EXCEEDED"

    failed_crop = ScriptedPlanner([
        tool_call("inspect_image", {"image_path": str(path)}),
        tool_call("crop_image", {
            "image_path": str(path), "x1": 0, "y1": 0, "x2": 999, "y2": 999,
        }),
        '{"type":"final","answer":"裁剪坐标越界，任务未完成。"}',
    ])
    agent, _, _ = make_agent(settings, failed_crop)
    handled = await agent.run(AgentRequest(message="错误裁剪", image_path=str(path)))
    assert handled.success
    assert handled.steps[0].success is True
    assert handled.steps[1].success is False
    assert handled.steps[1].error_type == "INVALID_CROP"
    assert failed_crop.calls[2]["observations"] == 2


@pytest.mark.anyio
async def test_invalid_final_json_recovers_completed_analysis(settings):
    path = source_image(settings)
    planner = ScriptedPlanner([
        tool_call("analyze_image", {
            "image_path": str(path), "prompt": "describe", "max_new_tokens": 64,
        }),
        "not-json",
        "still-not-json",
    ])
    agent, _, _ = make_agent(settings, planner)
    response = await agent.run(AgentRequest(
        message="分析图片", image_path=str(path), max_steps=3
    ))
    assert response.success and response.answer and "not-json" not in response.answer
    assert response.steps[-1].decision_type == "final"
    assert any("final JSON was invalid" in item for item in response.workflow.progress.warnings)

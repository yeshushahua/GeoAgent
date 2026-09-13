import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest
from PIL import Image

from backend.app.core.config import get_settings
from backend.app.main import create_app


def _has_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


@pytest.mark.integration
def test_real_qwen_vision_agent_scenarios_and_repeated_vram():
    settings = get_settings()
    source = settings.project_root / "sample_data" / "images" / "foundation-demo.png"
    report = {"source": str(source), "scenarios": {}, "repeated_runs": []}

    with TestClient(create_app(settings)) as client:
        assert (
            client.app.state.vision_agent.model_manager
            is client.app.state.tool_executor.context.model_manager
        )
        unloaded = client.post("/api/v1/models/vlm/unload")
        assert unloaded.status_code == 200 and unloaded.json()["state"] == "UNLOADED"

        def run(message: str):
            response = client.post(
                "/api/v1/agent/run",
                files={"image": ("scene.png", source.read_bytes(), "image/png")},
                data={"message": message, "max_steps": "6", "max_new_tokens": "64"},
            )
            assert response.status_code == 200, response.text
            return response.json()

        scenario_a = run("告诉我这张图片的宽度、高度和格式。")
        sequence_a = [step["tool_name"] for step in scenario_a["steps"] if step["tool_name"]]
        assert scenario_a["success"], scenario_a
        assert sequence_a == ["inspect_image"], scenario_a
        assert _has_chinese(scenario_a["answer"])
        report["scenarios"]["A"] = scenario_a

        scenario_b = run("分析一下这张图片主要有什么内容。")
        sequence_b = [step["tool_name"] for step in scenario_b["steps"] if step["tool_name"]]
        assert scenario_b["success"], scenario_b
        assert sequence_b == ["analyze_image"], scenario_b
        assert _has_chinese(scenario_b["answer"])
        report["scenarios"]["B"] = scenario_b

        scenario_c = run("先检查图片尺寸，然后裁剪左上四分之一区域。")
        sequence_c = [step["tool_name"] for step in scenario_c["steps"] if step["tool_name"]]
        assert scenario_c["success"], scenario_c
        assert sequence_c == ["inspect_image", "crop_image"], scenario_c
        crop_step_c = next(step for step in scenario_c["steps"] if step["tool_name"] == "crop_image")
        assert crop_step_c["arguments_summary"] == {
            "image_path": "original_image", "x1": 0, "y1": 0, "x2": 240, "y2": 150,
        }, scenario_c
        crop_path_c = Path(scenario_c["artifacts"][0]["path"])
        with Image.open(crop_path_c) as crop_c:
            assert crop_c.size == (240, 150)
        report["scenarios"]["C"] = scenario_c

        scenario_d = run("先检查图片尺寸，裁剪左上四分之一区域，然后分析裁剪后的内容。")
        sequence_d = [step["tool_name"] for step in scenario_d["steps"] if step["tool_name"]]
        assert scenario_d["success"], scenario_d
        assert sequence_d == ["inspect_image", "crop_image", "analyze_image"], scenario_d
        assert scenario_d["artifacts"], scenario_d
        crop_path = Path(scenario_d["artifacts"][0]["path"])
        assert crop_path.is_file() and crop_path.name == "crop.png"
        crop_step_d = next(step for step in scenario_d["steps"] if step["tool_name"] == "crop_image")
        assert crop_step_d["arguments_summary"] == {
            "image_path": "original_image", "x1": 0, "y1": 0, "x2": 240, "y2": 150,
        }, scenario_d
        with Image.open(crop_path) as crop_d:
            assert crop_d.size == (240, 150)
        analyze_step = next(step for step in scenario_d["steps"] if step["tool_name"] == "analyze_image")
        assert analyze_step["arguments_summary"]["image_path"] == "crop.png"
        assert _has_chinese(scenario_d["answer"])
        report["scenarios"]["D"] = scenario_d

        repeated = [scenario_b]
        for index in range(4):
            repeated.append(run(f"分析这张图片的主要内容。这是稳定性运行 {index + 2}。"))
        for index, result in enumerate(repeated, 1):
            assert result["success"] and [
                step["tool_name"] for step in result["steps"] if step["tool_name"]
            ] == ["analyze_image"], result
            report["repeated_runs"].append({
                "run": index,
                "run_id": result["run_id"],
                "allocated_vram_gib": result["metadata"]["allocated_vram_gib"],
                "peak_vram_gib": result["metadata"]["peak_vram_gib"],
                "total_duration_ms": result["metadata"]["total_duration_ms"],
                "planner_duration_ms": result["metadata"]["planner_duration_ms"],
                "tool_duration_ms": result["metadata"]["tool_duration_ms"],
            })
        allocations = [item["allocated_vram_gib"] for item in report["repeated_runs"]]
        report["vram_growth_gib"] = round(allocations[-1] - allocations[0], 3)
        assert report["vram_growth_gib"] <= 0.5

        traces = client.get("/api/v1/agent/executions?limit=20").json()
        assert len(traces) >= 7
        assert all("prompt" not in item or "prompt_length" in item for item in traces)
        report["agent_traces"] = traces
        report["model_status"] = client.get("/api/v1/models/vlm/status").json()
        client.post("/api/v1/models/vlm/unload")

    target = settings.output_dir / "benchmarks" / "phase3" / "phase3_agent_validation.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

"""Verify the running Phase 3 FastAPI service with a real Qwen Agent run."""
import json
from pathlib import Path

import httpx
from PIL import Image

from backend.app.core.config import get_settings


def main():
    settings = get_settings()
    source = settings.project_root / "sample_data" / "images" / "foundation-demo.png"
    base = settings.api_base_url
    task = "先检查图片尺寸，裁剪左上四分之一区域，然后分析裁剪后的内容。"
    with httpx.Client(timeout=1200, trust_env=False) as client:
        health = client.get(f"{base}/health")
        health.raise_for_status()
        tools = client.get(f"{base}/tools")
        tools.raise_for_status()
        client.post(f"{base}/models/vlm/unload").raise_for_status()
        agent = client.post(
            f"{base}/agent/run",
            files={"image": ("scene.png", source.read_bytes(), "image/png")},
            data={"message": task, "max_steps": "6", "max_new_tokens": "64"},
        )
        agent.raise_for_status()
        executions = client.get(f"{base}/agent/executions", params={"limit": 20})
        executions.raise_for_status()
        tool_check = client.post(
            f"{base}/tools/inspect_image/execute",
            files={"image": ("scene.png", source.read_bytes(), "image/png")},
        )
        tool_check.raise_for_status()
        status = client.get(f"{base}/models/vlm/status")
        status.raise_for_status()
    body = agent.json()
    sequence = [step["tool_name"] for step in body["steps"] if step["tool_name"]]
    assert health.json()["version"] == "0.4.0"
    assert body["success"] and body["answer"]
    assert sequence == ["inspect_image", "crop_image", "analyze_image"]
    assert body["artifacts"]
    crop_path = Path(body["artifacts"][0]["path"])
    assert crop_path.is_file() and crop_path.name == "crop.png"
    crop_step = next(step for step in body["steps"] if step["tool_name"] == "crop_image")
    assert crop_step["arguments_summary"] == {
        "image_path": "original_image", "x1": 0, "y1": 0, "x2": 240, "y2": 150,
    }
    with Image.open(crop_path) as crop:
        assert crop.size == (240, 150)
    analyze_step = next(step for step in body["steps"] if step["tool_name"] == "analyze_image")
    assert analyze_step["arguments_summary"]["image_path"] == "crop.png"
    assert executions.json() and executions.json()[0]["run_id"] == body["run_id"]
    assert tool_check.json()["success"]
    payload = {
        "health": health.json(),
        "registered_tools": [item["name"] for item in tools.json()],
        "task": task,
        "agent": body,
        "agent_executions": executions.json(),
        "tool_api_check": tool_check.json(),
        "model_status": status.json(),
    }
    target = settings.output_dir / "benchmarks" / "phase3" / "live_api_acceptance.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "health": payload["health"],
        "run_id": body["run_id"],
        "sequence": sequence,
        "answer": body["answer"],
        "artifact": str(crop_path),
        "crop_arguments": crop_step["arguments_summary"],
        "crop_size": [240, 150],
        "analyze_image_path": analyze_step["arguments_summary"]["image_path"],
        "metadata": body["metadata"],
        "agent_trace_count": len(payload["agent_executions"]),
        "tool_api_success": payload["tool_api_check"]["success"],
        "output": str(target),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

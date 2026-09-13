"""Live FastAPI acceptance for Phase 4 object detection and crop-to-detect."""
import json
from pathlib import Path

import httpx
from PIL import Image

from backend.app.core.config import get_settings


def main():
    settings = get_settings()
    source = settings.dataset_dir / "phase4" / "images" / "bus.jpg"
    base = settings.api_base_url
    task = "裁剪左上四分之一区域，然后检测里面的目标。"
    with httpx.Client(timeout=1200, trust_env=False) as client:
        health = client.get(f"{base}/health")
        health.raise_for_status()
        tools = client.get(f"{base}/tools")
        tools.raise_for_status()
        vlm_status = client.get(f"{base}/models/vlm/status")
        vlm_status.raise_for_status()
        detector_before = client.post(f"{base}/models/detector/unload")
        detector_before.raise_for_status()
        direct = client.post(
            f"{base}/tools/detect_objects/execute",
            files={"image": ("bus.jpg", source.read_bytes(), "image/jpeg")},
        )
        direct.raise_for_status()
        inspect = client.post(
            f"{base}/tools/inspect_image/execute",
            files={"image": ("bus.jpg", source.read_bytes(), "image/jpeg")},
        )
        inspect.raise_for_status()
        agent = client.post(
            f"{base}/agent/run",
            files={"image": ("bus.jpg", source.read_bytes(), "image/jpeg")},
            data={"message": task, "max_steps": "6", "max_new_tokens": "128"},
        )
        agent.raise_for_status()
        agent_traces = client.get(f"{base}/agent/executions", params={"limit": 20})
        agent_traces.raise_for_status()
        detector_after = client.get(f"{base}/models/detector/status")
        detector_after.raise_for_status()

    direct_body = direct.json()
    agent_body = agent.json()
    sequence = [step["tool_name"] for step in agent_body["steps"] if step["tool_name"]]
    assert health.json()["version"] == "0.5.0"
    assert "detect_objects" in [item["name"] for item in tools.json()]
    assert direct_body["success"] and direct_body["data"]["detection_count"] == 5
    assert direct_body["data"]["class_counts"] == {"bus": 1, "person": 4}
    direct_artifact = Path(direct_body["artifacts"][0]["path"])
    assert direct_artifact.is_file() and direct_artifact.name == "annotated.jpg"
    assert inspect.json()["success"] and inspect.json()["data"]["width"] == 810
    assert agent_body["success"] and sequence == [
        "inspect_image", "crop_image", "detect_objects"
    ]
    crop_step = next(step for step in agent_body["steps"] if step["tool_name"] == "crop_image")
    detect_step = next(
        step for step in agent_body["steps"] if step["tool_name"] == "detect_objects"
    )
    assert crop_step["arguments_summary"]["x2"] == 405
    assert crop_step["arguments_summary"]["y2"] == 540
    assert detect_step["arguments_summary"]["image_path"] == "crop.png"
    assert detect_step["observation_summary"]["image_width"] == 405
    assert detect_step["observation_summary"]["image_height"] == 540
    crop_artifact = Path(crop_step["artifacts"][0]["path"])
    annotated_artifact = Path(detect_step["artifacts"][0]["path"])
    with Image.open(crop_artifact) as crop, Image.open(annotated_artifact) as annotated:
        assert crop.size == annotated.size == (405, 540)
    assert detector_after.json()["load_count"] == 1
    assert agent_traces.json()[0]["run_id"] == agent_body["run_id"]

    payload = {
        "health": health.json(),
        "tools": [item["name"] for item in tools.json()],
        "vlm_status_before": vlm_status.json(),
        "detector_before": detector_before.json(),
        "direct_detection": direct_body,
        "legacy_inspect": inspect.json(),
        "task": task,
        "agent": agent_body,
        "agent_traces": agent_traces.json(),
        "detector_after": detector_after.json(),
    }
    target = settings.output_dir / "benchmarks" / "phase4" / "live_api_acceptance.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "run_id": agent_body["run_id"],
        "sequence": sequence,
        "direct_counts": direct_body["data"]["class_counts"],
        "crop_counts": detect_step["observation_summary"]["class_counts"],
        "crop_artifact": str(crop_artifact),
        "annotated_artifact": str(annotated_artifact),
        "detector": detector_after.json(),
        "output": str(target),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

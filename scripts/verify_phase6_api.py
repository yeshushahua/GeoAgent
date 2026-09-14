"""Live FastAPI acceptance for Phase 6 artifact-linked multi-step workflows."""
from __future__ import annotations

import json
from pathlib import Path

import httpx

from backend.app.core.config import get_settings


def _source(settings) -> Path:
    user_image = settings.project_root / "test" / "img" / "1.jpg"
    if user_image.is_file():
        return user_image
    return (
        settings.dataset_dir / "phase5" / "construction-ppe" /
        "images" / "test" / "image40.jpg"
    )


def main() -> None:
    settings = get_settings()
    source = _source(settings)
    task = (
        "先检查图片尺寸，然后裁剪左半部分，在裁剪结果中找到所有黄色安全帽和"
        "反光背心，并将它们分割出来，最后总结每类目标数量和面积占比。"
    )
    with httpx.Client(timeout=1200, trust_env=False) as client:
        health = client.get(f"{settings.api_base_url}/health")
        health.raise_for_status()
        tools = client.get(f"{settings.api_base_url}/tools")
        tools.raise_for_status()
        response = client.post(
            f"{settings.api_base_url}/agent/run",
            files={"image": (source.name, source.read_bytes(), "image/jpeg")},
            data={"message": task, "max_steps": "6", "max_new_tokens": "128"},
        )
        response.raise_for_status()
        traces = client.get(
            f"{settings.api_base_url}/agent/executions", params={"limit": 10}
        )
        traces.raise_for_status()

    body = response.json()
    sequence = [step["tool_name"] for step in body["steps"] if step["tool_name"]]
    assert health.json()["version"] == settings.app_version
    assert body["success"] and sequence == [
        "inspect_image", "crop_image", "detect_open_vocab", "segment_objects"
    ]
    workflow = body["workflow"]
    assert workflow["active_image_artifact_id"] == "crop-001"
    assert all("path" not in artifact for artifact in workflow["artifacts"])
    assert all(item["source_artifact_id"] == "crop-001" for item in workflow["detections"])
    assert workflow["detections"] and len(workflow["detections"]) == len(workflow["segmentations"])
    detections = {item["detection_id"]: item for item in workflow["detections"]}
    for item in workflow["segmentations"]:
        linked = detections[item["source_detection_id"]]
        assert item["source_artifact_id"] == linked["source_artifact_id"] == "crop-001"
        assert item["bbox"] == linked["bbox"]
    detect_step = next(step for step in body["steps"] if step["tool_name"] == "detect_open_vocab")
    segment_step = next(step for step in body["steps"] if step["tool_name"] == "segment_objects")
    assert detect_step["arguments_summary"]["image_path"] == "crop-001"
    assert segment_step["arguments_summary"]["image_path"] == "crop-001"
    assert segment_step["arguments_summary"]["detection_ids"] == [
        item["detection_id"] for item in workflow["detections"]
    ]
    assert body["metadata"]["workflow_total_ms"] == body["metadata"]["total_duration_ms"]
    assert body["metadata"]["successful_steps"] == len(body["steps"])
    assert body["metadata"]["failed_steps"] == 0
    assert traces.json()[0]["run_id"] == body["run_id"]
    assert {"detect_open_vocab", "segment_objects"}.issubset(
        item["name"] for item in tools.json()
    )

    target = settings.output_dir / "benchmarks" / "phase6" / "live_api_acceptance.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "health": health.json(), "task": task, "sequence": sequence,
        "response": body, "trace": traces.json()[0],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "sequence": sequence,
        "active_image": workflow["active_image_artifact_id"],
        "categories": workflow["categories"],
        "peak_vram_gib": body["metadata"]["peak_vram_gib"],
        "output": str(target),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

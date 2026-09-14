"""Live FastAPI acceptance for Phase 5 open-vocabulary detection and SAM."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
from PIL import Image

from backend.app.core.config import get_settings


def _assert_artifact(path: str, size: tuple[int, int]) -> Path:
    target = Path(path)
    assert target.is_file() and target.stat().st_size > 0
    with Image.open(target) as image:
        assert image.size == size
    return target


def main() -> None:
    settings = get_settings()
    source = settings.dataset_dir / "phase5" / "construction-ppe" / "images" / "test" / "image40.jpg"
    image_bytes = source.read_bytes()
    base = settings.api_base_url
    with httpx.Client(timeout=1200, trust_env=False) as client:
        health = client.get(f"{base}/health"); health.raise_for_status()
        tools = client.get(f"{base}/tools"); tools.raise_for_status()
        client.post(f"{base}/models/open-vocabulary/unload").raise_for_status()
        client.post(f"{base}/models/segmentation/unload").raise_for_status()
        detected = client.post(
            f"{base}/tools/detect_open_vocab/execute",
            files={"image": (source.name, image_bytes, "image/jpeg")},
            data={"classes": json.dumps(["yellow safety helmet"])},
        ); detected.raise_for_status()
        detected_body = detected.json()
        boxes = [item["bbox"] for item in detected_body["data"]["detections"]]
        segmented = client.post(
            f"{base}/tools/segment_objects/execute",
            files={"image": (source.name, image_bytes, "image/jpeg")},
            data={"boxes": json.dumps(boxes)},
        ); segmented.raise_for_status()
        agent = client.post(
            f"{base}/agent/run",
            files={"image": (source.name, image_bytes, "image/jpeg")},
            data={
                "message": "裁剪左上四分之一区域，在里面寻找黄色安全帽，并把找到的目标分割出来。",
                "max_steps": "6", "max_new_tokens": "128",
            },
        ); agent.raise_for_status()
        open_status = client.get(f"{base}/models/open-vocabulary/status"); open_status.raise_for_status()
        sam_status = client.get(f"{base}/models/segmentation/status"); sam_status.raise_for_status()

    tool_names = [item["name"] for item in tools.json()]
    assert health.json()["version"] == settings.app_version
    assert {"detect_open_vocab", "segment_objects"}.issubset(tool_names)
    assert detected_body["success"] and detected_body["data"]["detection_count"] >= 1
    assert segmented.json()["success"] and segmented.json()["data"]["segment_count"] == len(boxes)
    for item in segmented.json()["data"]["segments"]:
        assert item["mask_area_pixels"] > 0 and 0 < item["mask_area_ratio"] < 1
        _assert_artifact(item["mask_artifact_path"], (640, 640))
    _assert_artifact(segmented.json()["data"]["overlay_artifact_path"], (640, 640))
    agent_body = agent.json()
    sequence = [step["tool_name"] for step in agent_body["steps"] if step["tool_name"]]
    assert agent_body["success"]
    assert sequence == ["inspect_image", "crop_image", "detect_open_vocab", "segment_objects"]
    detect_step = next(step for step in agent_body["steps"] if step["tool_name"] == "detect_open_vocab")
    segment_step = next(step for step in agent_body["steps"] if step["tool_name"] == "segment_objects")
    assert detect_step["arguments_summary"]["image_path"] == "crop-001"
    assert segment_step["arguments_summary"]["image_path"] == "crop-001"
    assert segment_step["arguments_summary"]["detection_ids"] == (
        detect_step["observation_summary"]["workflow_detection_ids"]
    )
    assert open_status.json()["state"] == sam_status.json()["state"] == "READY"
    assert open_status.json()["load_count"] == sam_status.json()["load_count"] == 1

    payload = {
        "health": health.json(), "tools": tool_names,
        "direct_detection": detected_body, "direct_segmentation": segmented.json(),
        "agent": agent_body, "open_vocabulary": open_status.json(),
        "segmentation": sam_status.json(),
    }
    target = settings.output_dir / "benchmarks" / "phase5" / "live_api_acceptance.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"sequence": sequence, "boxes": boxes, "output": str(target)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

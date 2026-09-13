"""Live Gradio acceptance for Phase 5 status, Agent flow, masks, and manual tools."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from backend.app.core.config import get_settings
from backend.app.services.storage import prepare_storage


def _path(value) -> Path:
    if not isinstance(value, dict):
        return Path(value)
    for key in ("path", "artifact_path"):
        if value.get(key):
            return Path(value[key])
    if "image" in value:
        return _path(value["image"])
    raise AssertionError(f"Gradio file value has no path: {value}")


def main() -> None:
    settings = get_settings()
    prepare_storage(settings)
    from gradio_client import Client, handle_file

    host = "127.0.0.1" if settings.gradio_host == "0.0.0.0" else settings.gradio_host
    client = Client(f"http://{host}:{settings.gradio_port}", verbose=False, httpx_kwargs={"trust_env": False})
    status, info = client.predict(api_name="/status")
    assert "YOLOE-26s" in status and "SAM 2.1 Base" in status
    assert {"detect_open_vocab", "segment_objects"}.issubset(info["tools"])

    source = settings.dataset_dir / "phase5" / "construction-ppe" / "images" / "test" / "image40.jpg"
    task = "找到图中的黄色安全帽，并把它们精确分割出来。"
    answer, result, status, info, agent_panel, preview, gallery = client.predict(
        handle_file(str(source)), task, 128, api_name="/analyze"
    )
    sequence = [step["tool_name"] for step in result["steps"] if step["tool_name"]]
    assert result["success"] and answer and sequence == ["detect_open_vocab", "segment_objects"]
    assert "分割实例" in agent_panel and "面积比例" in agent_panel and gallery
    overlay = _path(preview)
    with Image.open(overlay) as image:
        assert image.size == (640, 640)
    for item in gallery:
        assert _path(item).is_file()

    summary, manual, manual_preview, manual_panel, manual_gallery = client.predict(
        "detect_open_vocab", handle_file(str(source)), "", 128,
        "yellow safety helmet", 0.25, 0.45, 0, 0, 320, 320, "[]",
        api_name="/execute_tool"
    )
    assert manual["success"] and manual["data"]["detection_count"] >= 1
    assert "YOLOE-26s" in manual_panel and manual_preview and not manual_gallery
    box = manual["data"]["detections"][0]["bbox"]
    summary2, mask_result, mask_preview, mask_panel, mask_gallery = client.predict(
        "segment_objects", handle_file(str(source)), "", 128, "", 0.25, 0.45,
        0, 0, 0, 0, json.dumps([box]), api_name="/execute_tool"
    )
    assert mask_result["success"] and mask_result["data"]["segment_count"] == 1
    assert mask_preview and mask_gallery and "SAM 2.1 Base" in mask_panel

    payload = {
        "task": task, "answer": answer, "sequence": sequence,
        "agent": result, "agent_panel": agent_panel, "overlay": str(overlay),
        "gallery": [str(_path(item)) for item in gallery],
        "manual_detection": manual, "manual_segmentation": mask_result,
    }
    target = settings.output_dir / "benchmarks" / "phase5" / "gradio_acceptance.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"sequence": sequence, "mask_count": len(gallery), "output": str(target)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

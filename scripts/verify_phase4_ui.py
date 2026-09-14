"""Live Gradio queue/component acceptance for Phase 4 detection."""
import json
import logging
from pathlib import Path

import httpx
from PIL import Image

from backend.app.core.config import get_settings
from backend.app.core.logging import configure_logging
from backend.app.services.storage import prepare_storage


def main():
    settings = get_settings()
    configure_logging(settings.log_level)
    prepare_storage(settings)
    from gradio_client import Client, handle_file

    host = "127.0.0.1" if settings.gradio_host == "0.0.0.0" else settings.gradio_host
    url = f"http://{host}:{settings.gradio_port}"
    client = Client(url, verbose=False, httpx_kwargs={"trust_env": False})
    status, info = client.predict(api_name="/status")
    assert "YOLO11s" in status and "detect_objects" in info["tools"]

    source = settings.dataset_dir / "phase4" / "images" / "bus.jpg"
    task = "检测这张图片里的目标，并告诉我每类有多少个。"
    answer, result, status, info, agent_panel, result_preview = client.predict(
        handle_file(str(source)), task, 128, api_name="/analyze"
    )
    assert result["success"] and answer
    sequence = [step["tool_name"] for step in result["steps"] if step["tool_name"]]
    assert sequence == ["detect_objects"]
    detection = result["steps"][0]["observation_summary"]
    assert detection["detection_count"] == 5
    assert detection["class_counts"] == {"bus": 1, "person": 4}
    assert "检测总数：**5**" in agent_panel
    assert "bus × 1" in agent_panel and "person × 4" in agent_panel
    artifact = Path(result["artifacts"][0]["artifact_path"])
    preview = Path(result_preview["path"] if isinstance(result_preview, dict) else result_preview)
    with Image.open(artifact) as annotated, Image.open(preview) as shown:
        assert annotated.size == shown.size == (810, 1080)
        assert annotated.convert("RGB").tobytes() == shown.convert("RGB").tobytes()
    assert info["detector"]["state"] == "READY"
    assert info["detector"]["load_count"] == 1

    summary, manual, manual_preview, manual_panel, manual_gallery = client.predict(
        "detect_objects", handle_file(str(source)), "", 64, "", 0.25, 0.45,
        0, 0, 256, 256, "[]", "", api_name="/execute_tool"
    )
    assert manual["success"] and manual["data"]["detection_count"] == 5
    assert "检测完成" in summary and "YOLO11s" in manual_panel
    assert manual_preview
    assert not manual_gallery

    with httpx.Client(timeout=30, trust_env=False) as http:
        detector_unloaded = http.post(
            f"{settings.api_base_url}/models/detector/unload"
        ).json()
    output = {
        "task": task,
        "answer": answer,
        "sequence": sequence,
        "detection": detection,
        "artifact": str(artifact),
        "preview": str(preview),
        "agent_panel": agent_panel,
        "manual_detection": manual,
        "detector_unloaded": detector_unloaded,
    }
    target = settings.output_dir / "benchmarks" / "phase4" / "gradio_acceptance.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Gradio Phase 4 detection acceptance PASS: %s", target)


if __name__ == "__main__":
    main()

"""Live Gradio acceptance for Phase 6 Workflow trace and artifacts."""
from __future__ import annotations

import json
from pathlib import Path

from backend.app.core.config import get_settings
from backend.app.services.storage import prepare_storage


def _source(settings) -> Path:
    user_image = settings.project_root / "test" / "img" / "1.jpg"
    if user_image.is_file():
        return user_image
    return (
        settings.dataset_dir / "phase5" / "construction-ppe" /
        "images" / "test" / "image40.jpg"
    )


def _path(value) -> Path:
    if isinstance(value, dict):
        if value.get("path"):
            return Path(value["path"])
        if "image" in value:
            return _path(value["image"])
    return Path(value)


def main() -> None:
    settings = get_settings()
    prepare_storage(settings)
    from gradio_client import Client, handle_file

    host = "127.0.0.1" if settings.gradio_host == "0.0.0.0" else settings.gradio_host
    client = Client(
        f"http://{host}:{settings.gradio_port}", verbose=False,
        httpx_kwargs={"trust_env": False},
    )
    status, info = client.predict(api_name="/status")
    assert "Phase" not in status
    assert {"detect_open_vocab", "segment_objects"}.issubset(info["tools"])

    source = _source(settings)
    task = "找到黄色安全帽、反光背心和挖掘机，并将找到的目标分割出来。"
    answer, result, status, info, workflow_panel, preview, gallery = client.predict(
        handle_file(str(source)), task, 128, api_name="/analyze"
    )
    sequence = [step["tool_name"] for step in result["steps"] if step["tool_name"]]
    assert result["success"] and answer
    assert sequence == ["detect_open_vocab", "segment_objects"]
    categories = result["workflow"]["categories"]
    assert categories["excavator"]["detected"] == 0
    assert any(item["detected"] > 0 for name, item in categories.items() if name != "excavator")
    assert all(
        item["segmented"] == item["detected"]
        for name, item in categories.items() if name != "excavator"
    )
    assert all(text in workflow_panel for text in (
        "### Workflow", "Detection IDs", "Artifact 依赖", "聚合结果", "生成最终回答"
    ))
    assert "挖掘机未在图像中检测到" in answer or "excavator：未检测到" in answer
    serialized = json.dumps(result, ensure_ascii=False)
    assert "D:/" not in serialized and "E:/" not in serialized
    assert gallery and _path(preview).is_file()
    assert all(_path(item).is_file() for item in gallery)

    summary, manual, manual_preview, manual_panel, manual_gallery = client.predict(
        "inspect_image", handle_file(str(source)), "", 64, "", 0.25, 0.45,
        0, 0, 0, 0, "[]", "", api_name="/execute_tool"
    )
    assert manual["success"] and manual["data"]["width"] > 0
    assert summary and manual_preview and "成功" in manual_panel and not manual_gallery

    target = settings.output_dir / "benchmarks" / "phase6" / "gradio_acceptance.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "task": task, "answer": answer, "sequence": sequence,
        "workflow": result["workflow"], "workflow_panel": workflow_panel,
        "manual_inspect": manual,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "sequence": sequence, "categories": categories,
        "gallery_count": len(gallery), "output": str(target),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

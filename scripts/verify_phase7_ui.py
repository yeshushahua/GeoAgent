"""Live Gradio acceptance for Phase 7 raster panels and preview."""
from __future__ import annotations

import json
from pathlib import Path

from backend.app.core.config import get_settings
from backend.app.services.storage import prepare_storage
from scripts.verify_phase7_api import make_phase7_raster


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
    assert {"inspect_raster", "raster_preview", "crop_raster", "raster_statistics"}.issubset(
        info["tools"]
    )

    source = make_phase7_raster(settings.temp_dir / "phase7-live" / "gradio-source.tif")
    task = "生成这幅遥感影像的 RGB 预览。"
    answer, result, status, info, workflow_panel, raster_panel, preview, gallery = client.predict(
        None, handle_file(str(source)), task, 128, api_name="/analyze"
    )
    sequence = [step["tool_name"] for step in result["steps"] if step["tool_name"]]
    assert result["success"] and answer
    assert sequence == ["inspect_raster", "raster_preview"]
    workflow = result["workflow"]
    assert workflow["active_raster_artifact_id"] == "raster-001"
    assert workflow["active_image_artifact_id"] == "raster-preview-001"
    assert all(text in workflow_panel for text in (
        "### Workflow", "当前 Raster", "raster_preview", "Artifact 依赖", "生成最终回答"
    ))
    assert all(text in raster_panel for text in (
        "Raster Information", "256 × 192", "EPSG:32647", "10.0 × 10.0"
    ))
    assert preview and _path(preview).is_file() and not gallery
    serialized = json.dumps(result, ensure_ascii=False)
    assert "D:/" not in serialized and "E:/" not in serialized

    summary, manual, manual_preview, manual_panel, manual_gallery, manual_raster = client.predict(
        "raster_preview", None, handle_file(str(source)), "", 64, "", 0.25, 0.45,
        0, 0, 256, 192, "[]", "", "1,2,3", "percentile", 2, 98, 128,
        "bilinear", None, 0, 64, 0, 64, api_name="/execute_tool"
    )
    assert manual["success"] and manual["data"]["preview_width"] == 128
    assert summary and manual_preview and "成功" in manual_panel and not manual_gallery
    assert "Raster Information" in manual_raster and "Bands [1, 2, 3]" in manual_raster

    target = settings.output_dir / "benchmarks" / "phase7" / "gradio_acceptance.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "task": task, "answer": answer, "sequence": sequence,
        "workflow": workflow, "workflow_panel": workflow_panel,
        "raster_panel": raster_panel, "manual_preview": manual,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "sequence": sequence,
        "active_raster": workflow["active_raster_artifact_id"],
        "active_image": workflow["active_image_artifact_id"],
        "output": str(target),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

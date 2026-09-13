"""HTTP-level Gradio Phase 3 acceptance using real component event endpoints."""
import argparse
import json
import logging
from pathlib import Path

import httpx
from PIL import Image

from backend.app.core.config import get_settings
from backend.app.core.logging import configure_logging
from backend.app.services.storage import prepare_storage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-unavailable", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    configure_logging()
    prepare_storage(settings)
    from gradio_client import Client, handle_file

    host = "127.0.0.1" if settings.gradio_host == "0.0.0.0" else settings.gradio_host
    url = f"http://{host}:{settings.gradio_port}"
    with httpx.Client(trust_env=False, timeout=15) as http:
        response = http.get(url)
        response.raise_for_status()
        assert "GeoAgent" in response.text
    client = Client(url, verbose=False, httpx_kwargs={"trust_env": False})
    status, info = client.predict(api_name="/status")
    if args.expect_unavailable:
        assert "后端不可用" in status, status
        assert info["backend"] == "unavailable"
        logging.info("Gradio unavailable-backend HTTP test PASS")
        return
    assert "RTX 4090" in status
    source = settings.project_root / "sample_data" / "images" / "foundation-demo.png"
    preview = client.predict(handle_file(str(source)), api_name="/preview")
    result_path = Path(preview["path"] if isinstance(preview, dict) else preview)
    with Image.open(source) as original, Image.open(result_path) as result:
        assert original.size == result.size
        assert original.convert("RGB").tobytes() == result.convert("RGB").tobytes()
    status, info = client.predict(api_name="/unload_model")
    assert info["model"]["state"] == "UNLOADED"
    task = "先检查图片尺寸，裁剪左上四分之一区域，然后分析裁剪后的内容。"
    answer, result, status, info, agent_panel, result_preview = client.predict(
        handle_file(str(source)), task, 64, api_name="/analyze"
    )
    assert answer and result["success"]
    sequence = [step["tool_name"] for step in result["steps"] if step["tool_name"]]
    assert sequence == ["inspect_image", "crop_image", "analyze_image"]
    assert result["metadata"]["model"] == "Qwen3-VL-4B-Instruct"
    assert result["metadata"]["device"] == "cuda:0"
    assert result["metadata"]["planner_duration_ms"] > 0
    assert result["metadata"]["tool_duration_ms"] > 0
    assert result["metadata"]["peak_vram_gib"] > 0
    assert result["run_id"] in agent_panel
    assert all(name in agent_panel for name in sequence)
    assert "生成最终回答" in agent_panel
    assert result["artifacts"] and result_preview
    crop_path = Path(result["artifacts"][0]["artifact_path"])
    crop_step = next(step for step in result["steps"] if step["tool_name"] == "crop_image")
    assert crop_step["arguments_summary"] == {
        "image_path": "original_image", "x1": 0, "y1": 0, "x2": 240, "y2": 150,
    }
    analyze_step = next(step for step in result["steps"] if step["tool_name"] == "analyze_image")
    assert analyze_step["arguments_summary"]["image_path"] == "crop.png"
    preview_path = Path(result_preview["path"] if isinstance(result_preview, dict) else result_preview)
    with Image.open(crop_path) as crop, Image.open(preview_path) as preview_image:
        assert crop.size == (240, 150)
        assert crop.size == preview_image.size
        assert crop.convert("RGB").tobytes() == preview_image.convert("RGB").tobytes()
    assert info["model"]["state"] == "READY"
    summary, inspect_result, inspect_preview, inspect_panel, inspect_gallery = client.predict(
        "inspect_image", handle_file(str(source)), "", 64, "", 0.25, 0.45,
        0, 0, 256, 256, "[]", api_name="/execute_tool"
    )
    assert summary and inspect_result["data"]["width"] > 0
    assert inspect_preview and "成功" in inspect_panel
    assert not inspect_gallery
    status, info = client.predict(api_name="/unload_model")
    assert info["model"]["state"] == "UNLOADED"
    output = {
        "answer": answer,
        "task": task,
        "agent_result": result,
        "agent_panel": agent_panel,
        "result_preview": str(preview_path),
        "manual_inspect": inspect_result,
        "final_model_state": info["model"],
    }
    target = settings.output_dir / "benchmarks" / "phase3" / "gradio_acceptance.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info(
        "Gradio Phase 3 PASS: Chinese UI event, auto-load Agent API, autonomous Scenario D, execution panel, crop preview, manual Tool debug, unload"
    )
    logging.info("Acceptance written to %s", target)


if __name__ == "__main__":
    main()

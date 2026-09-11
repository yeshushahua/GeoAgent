"""Verify the live Phase 2 HTTP Tool API with the local Qwen model."""
import json
from pathlib import Path

import httpx

from backend.app.core.config import get_settings


def main():
    settings = get_settings()
    source = settings.project_root / "sample_data" / "images" / "foundation-demo.png"
    base = settings.api_base_url
    with httpx.Client(timeout=900, trust_env=False) as client:
        health = client.get(f"{base}/health")
        health.raise_for_status()
        tools = client.get(f"{base}/tools")
        tools.raise_for_status()
        detail = client.get(f"{base}/tools/analyze_image")
        detail.raise_for_status()
        unloaded = client.post(f"{base}/models/vlm/unload")
        unloaded.raise_for_status()
        inspect = client.post(
            f"{base}/tools/inspect_image/execute",
            files={"image": ("unsafe/../scene.png", source.read_bytes(), "image/png")},
        )
        inspect.raise_for_status()
        crop = client.post(
            f"{base}/tools/crop_image/execute",
            files={"image": ("scene.png", source.read_bytes(), "image/png")},
            data={"x1": "10", "y1": "10", "x2": "310", "y2": "210"},
        )
        crop.raise_for_status()
        crop_path = Path(crop.json()["artifacts"][0]["path"])
        analyze = client.post(
            f"{base}/tools/analyze_image/execute",
            data={
                "image_path": str(crop_path),
                "prompt": "Describe this cropped image briefly.",
                "max_new_tokens": "64",
            },
        )
        analyze.raise_for_status()
        status = client.get(f"{base}/models/vlm/status")
        status.raise_for_status()
        traces = client.get(f"{base}/tools/executions", params={"limit": 20})
        traces.raise_for_status()
    payload = {
        "health": health.json(),
        "tools": tools.json(),
        "analyze_definition": detail.json(),
        "unloaded": unloaded.json(),
        "inspect": inspect.json(),
        "crop": crop.json(),
        "analyze": analyze.json(),
        "status_after_analyze": status.json(),
        "traces": traces.json(),
    }
    assert payload["health"]["version"] == "0.3.0"
    assert [item["name"] for item in payload["tools"]] == [
        "analyze_image", "crop_image", "inspect_image"
    ]
    assert payload["unloaded"]["state"] == "UNLOADED"
    assert payload["inspect"]["success"] and payload["crop"]["success"]
    assert crop_path.is_file() and crop_path.is_relative_to(settings.output_dir / "tools")
    assert payload["analyze"]["success"] and payload["analyze"]["data"]["answer"]
    assert payload["status_after_analyze"]["state"] == "READY"
    assert len(payload["traces"]) >= 3
    target = settings.output_dir / "benchmarks" / "phase2" / "live_api_acceptance.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "health": payload["health"],
        "tools": [item["name"] for item in payload["tools"]],
        "crop_artifact": str(crop_path),
        "answer": payload["analyze"]["data"]["answer"],
        "execution_id": payload["analyze"]["metadata"]["execution_id"],
        "duration_ms": payload["analyze"]["metadata"]["duration_ms"],
        "latency_ms": payload["analyze"]["metadata"]["latency_ms"],
        "state": payload["status_after_analyze"]["state"],
        "trace_count": len(payload["traces"]),
        "output": str(target),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

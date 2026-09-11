import json
from pathlib import Path
import time

from fastapi.testclient import TestClient
import pytest

from backend.app.core.config import get_settings
from backend.app.main import create_app


@pytest.mark.integration
def test_real_phase2_tool_chain_auto_load_and_memory_regression():
    settings = get_settings()
    source = settings.project_root / "sample_data" / "images" / "foundation-demo.png"
    assert source.is_file() and settings.hf_hub_offline is True
    report = {"source": str(source), "runs": []}
    with TestClient(create_app(settings)) as client:
        unloaded = client.post("/api/v1/models/vlm/unload")
        assert unloaded.status_code == 200 and unloaded.json()["state"] == "UNLOADED"

        files = {"image": ("scene.png", source.read_bytes(), "image/png")}
        auto = client.post(
            "/api/v1/tools/analyze_image/execute",
            files=files,
            data={"prompt": "Describe this image briefly.", "max_new_tokens": "64"},
        )
        auto_body = auto.json()
        assert auto.status_code == 200 and auto_body["success"]
        assert auto_body["data"]["answer"]
        assert auto_body["metadata"]["model"] == "Qwen3-VL-4B-Instruct"
        assert auto_body["metadata"]["execution_id"]
        assert auto_body["metadata"]["duration_ms"] > 0
        assert client.get("/api/v1/models/vlm/status").json()["state"] == "READY"
        report["auto_load"] = auto_body

        inspected = client.post(
            "/api/v1/tools/inspect_image/execute",
            files={"image": ("scene.png", source.read_bytes(), "image/png")},
        ).json()
        assert inspected["success"]
        cropped = client.post(
            "/api/v1/tools/crop_image/execute",
            files={"image": ("scene.png", source.read_bytes(), "image/png")},
            data={"x1": "10", "y1": "10", "x2": "310", "y2": "210"},
        ).json()
        assert cropped["success"]
        crop_path = Path(cropped["artifacts"][0]["path"])
        assert crop_path.is_file() and crop_path.is_relative_to(settings.output_dir / "tools")
        chained = client.post(
            "/api/v1/tools/analyze_image/execute",
            data={
                "image_path": str(crop_path),
                "prompt": "Describe this cropped image briefly.",
                "max_new_tokens": "64",
            },
        ).json()
        assert chained["success"] and chained["data"]["answer"]
        report["manual_sequence"] = {
            "inspect_image": inspected,
            "crop_image": cropped,
            "analyze_image": chained,
        }

        runs = [auto_body, chained]
        for index in range(3):
            response = client.post(
                "/api/v1/tools/analyze_image/execute",
                files={"image": ("scene.png", source.read_bytes(), "image/png")},
                data={"prompt": f"Describe this image briefly, run {index + 3}.", "max_new_tokens": "64"},
            )
            assert response.status_code == 200 and response.json()["success"]
            runs.append(response.json())
        allocations = [item["metadata"]["gpu_allocated_gb"] for item in runs]
        assert allocations[-1] - allocations[0] <= 0.25
        report["runs"] = runs
        report["memory"] = {
            "allocated_gb": allocations,
            "growth_gb": round(allocations[-1] - allocations[0], 3),
        }
        tool_overheads = [
            round(item["metadata"]["duration_ms"] - item["metadata"]["latency_ms"], 2)
            for item in runs[1:]
        ]
        started = time.perf_counter()
        direct = client.post(
            "/api/v1/models/vlm/infer",
            files={"image": ("scene.png", source.read_bytes(), "image/png")},
            data={"prompt": "Describe this image briefly.", "max_new_tokens": "64"},
        )
        direct_http_ms = round((time.perf_counter() - started) * 1000, 2)
        assert direct.status_code == 200 and direct.json()["text"]
        report["performance"] = {
            "tool_executor_overhead_ms": tool_overheads,
            "tool_executor_overhead_avg_ms": round(sum(tool_overheads) / len(tool_overheads), 2),
            "direct_api_total_ms": direct_http_ms,
            "direct_model_latency_ms": direct.json()["latency_ms"],
        }
        traces = client.get("/api/v1/tools/executions?limit=20").json()
        assert any(item["tool"] == "analyze_image" for item in traces)
        report["traces"] = traces
        client.post("/api/v1/models/vlm/unload")

    target = settings.output_dir / "benchmarks" / "phase2" / "phase2_tool_validation.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

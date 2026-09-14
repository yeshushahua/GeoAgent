import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from backend.app.core.config import get_settings
from backend.app.main import create_app


def _sequence(result):
    return [step["tool_name"] for step in result["steps"] if step["tool_name"]]


@pytest.mark.integration
def test_real_yolo_detector_stability_and_artifacts():
    settings = get_settings()
    images = settings.dataset_dir / "phase4" / "images"
    sources = [images / "bus.jpg", images / "zidane.jpg"]
    report = {"model": {}, "runs": [], "class_filter": {}}

    with TestClient(create_app(settings)) as client:
        client.post("/api/v1/models/vlm/unload")
        unloaded = client.post("/api/v1/models/detector/unload").json()
        assert unloaded["state"] == "UNLOADED" and unloaded["load_count"] == 0

        for index in range(5):
            source = sources[index % len(sources)]
            response = client.post(
                "/api/v1/tools/detect_objects/execute",
                files={"image": (source.name, source.read_bytes(), "image/jpeg")},
                data={"confidence": "0.25", "iou_threshold": "0.45"},
            )
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["success"] and body["data"]["detection_count"] > 0
            assert all(0 <= item["confidence"] <= 1 for item in body["data"]["detections"])
            width, height = body["data"]["image_width"], body["data"]["image_height"]
            assert all(
                0 <= item["bbox"]["x1"] < item["bbox"]["x2"] <= width
                and 0 <= item["bbox"]["y1"] < item["bbox"]["y2"] <= height
                for item in body["data"]["detections"]
            )
            artifact = Path(body["artifacts"][0]["path"])
            with Image.open(artifact) as annotated:
                assert annotated.size == (width, height)
            report["runs"].append({
                "run": index + 1,
                "image": source.name,
                "detection_count": body["data"]["detection_count"],
                "class_counts": body["data"]["class_counts"],
                "artifact": str(artifact),
                "load_count": body["metadata"]["detector_load_count"],
                "load_time_s": body["metadata"]["detector_load_time_s"],
                "inference_ms": body["metadata"]["detector_inference_ms"],
                "allocated_vram_gib": body["metadata"]["gpu_allocated_gb"],
                "peak_vram_gib": body["metadata"]["gpu_peak_gb"],
            })

        allocations = [item["allocated_vram_gib"] for item in report["runs"]]
        assert all(item["load_count"] == 1 for item in report["runs"])
        assert max(allocations) - min(allocations) <= 0.5

        source = sources[0]
        filtered = client.post(
            "/api/v1/tools/detect_objects/execute",
            files={"image": (source.name, source.read_bytes(), "image/jpeg")},
            data={"classes": '["bus"]'},
        ).json()
        assert filtered["success"] and filtered["data"]["detection_count"] >= 1
        assert set(filtered["data"]["class_counts"]) == {"bus"}
        report["class_filter"] = filtered["data"]
        report["model"] = client.get("/api/v1/models/detector/status").json()
        assert report["model"]["load_count"] == 1
        client.post("/api/v1/models/detector/unload")

    target = settings.output_dir / "benchmarks" / "phase4" / "detector_validation.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


@pytest.mark.integration
def test_real_qwen_agent_detection_scenarios():
    settings = get_settings()
    source = settings.dataset_dir / "phase4" / "images" / "bus.jpg"
    report = {"source": str(source), "scenarios": {}, "models": {}}

    with TestClient(create_app(settings)) as client:
        client.post("/api/v1/models/vlm/unload")
        client.post("/api/v1/models/detector/unload")

        def run(message: str):
            response = client.post(
                "/api/v1/agent/run",
                files={"image": ("bus.jpg", source.read_bytes(), "image/jpeg")},
                data={"message": message, "max_steps": "6", "max_new_tokens": "128"},
            )
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["success"], body
            return body

        scenario_a = run("检测这张图片里的目标，并告诉我每类有多少个。")
        assert _sequence(scenario_a) == ["detect_objects"], scenario_a
        detected_a = scenario_a["steps"][0]["observation_summary"]
        assert detected_a["detection_count"] > 0 and detected_a["class_counts"]
        assert Path(scenario_a["artifacts"][-1]["path"]).name == "annotated.jpg"
        report["scenarios"]["A"] = scenario_a

        scenario_b = run("先检测图中的目标，再结合检测结果分析这张图片。")
        assert _sequence(scenario_b) == ["detect_objects", "analyze_image"], scenario_b
        analyze_step = next(
            step for step in scenario_b["steps"] if step["tool_name"] == "analyze_image"
        )
        assert analyze_step["arguments_summary"]["detection_observation_count"] == 1
        assert scenario_b["steps"][0]["observation_summary"]["detection_count"] > 0
        report["scenarios"]["B"] = scenario_b

        scenario_c = run("裁剪左上四分之一区域，然后检测里面的目标。")
        assert _sequence(scenario_c) == [
            "inspect_image", "crop_image", "detect_objects"
        ], scenario_c
        crop_step = next(step for step in scenario_c["steps"] if step["tool_name"] == "crop_image")
        assert crop_step["arguments_summary"] == {
            "image_path": "original-image-001", "x1": 0, "y1": 0, "x2": 405, "y2": 540,
        }, scenario_c
        detect_step = next(
            step for step in scenario_c["steps"] if step["tool_name"] == "detect_objects"
        )
        assert detect_step["arguments_summary"]["image_path"] == "crop-001"
        assert detect_step["observation_summary"]["image_width"] == 405
        assert detect_step["observation_summary"]["image_height"] == 540
        assert detect_step["observation_summary"]["detection_count"] > 0
        crop_artifact = Path(crop_step["artifacts"][0]["path"])
        annotated_artifact = Path(detect_step["artifacts"][0]["path"])
        assert crop_artifact.name == "crop.png" and annotated_artifact.name == "annotated.jpg"
        report["scenarios"]["C"] = scenario_c

        report["models"] = {
            "qwen": client.get("/api/v1/models/vlm/status").json(),
            "detector": client.get("/api/v1/models/detector/status").json(),
        }
        assert report["models"]["qwen"]["state"] == "READY"
        assert report["models"]["detector"]["state"] == "READY"
        assert report["models"]["detector"]["load_count"] == 1
        assert scenario_c["metadata"]["allocated_vram_gib"] < 20
        client.post("/api/v1/models/detector/unload")
        client.post("/api/v1/models/vlm/unload")

    target = settings.output_dir / "benchmarks" / "phase4" / "agent_validation.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

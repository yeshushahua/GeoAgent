"""Run the Phase 1 cold-load, stability, high-resolution and reload benchmark."""
from datetime import datetime
import json
from pathlib import Path
import statistics

from PIL import Image
import torch

from backend.app.core.config import get_settings
from backend.app.models.manager import ModelManager
from backend.app.models.qwen_vl import get_gpu_memory
from backend.app.services.storage import prepare_storage


CASES = [
    ("foundation-demo.png", "Describe this image in detail."),
    ("phase1-street.png", "请详细描述这张图片中的主要内容。"),
    ("phase1-objects.png", "What objects are visible in this image?"),
    ("phase1-spatial.png", "What is located in the upper-left area of the image? Describe the spatial relationships between the major objects."),
    ("phase1-street.png", "请判断图像中最主要的场景类型，并说明判断依据。"),
    ("phase1-objects.png", "Approximately how many colored balls are present? This is approximate VLM counting."),
    ("phase1-aerial-rgb.png", "请描述这幅航拍或遥感 RGB 图像中的主要地物类型及其空间分布特征。"),
    ("phase1-high-resolution.png", "Describe the main spatial pattern and confirm the image was safely resized."),
    ("phase1-aerial-rgb.png", "What is located near the center and right side of this aerial-style image?"),
    ("phase1-spatial.png", "请用中文说明红色、蓝色和绿色图形之间的空间关系。"),
]


def run_case(manager: ModelManager, image_path: Path, prompt: str) -> dict:
    with Image.open(image_path) as opened:
        image = opened.copy()
        image.format = opened.format
    result = manager.infer(image, prompt, 96)
    return {
        "model": result.model,
        "prompt": prompt,
        "image": image_path.name,
        "image_resolution": [result.image.width, result.image.height],
        "preprocessing_resolution": [
            result.image.preprocessing_width, result.image.preprocessing_height
        ],
        "inference_latency_ms": result.latency_ms,
        "output_length": len(result.text),
        "output_text": result.text,
        "gpu_allocated_gb": result.gpu.allocated_gb,
        "gpu_reserved_gb": result.gpu.reserved_gb,
        "peak_gpu_allocated_gb": result.gpu.peak_allocated_gb,
        "success": result.success,
    }


def main() -> None:
    settings = get_settings()
    prepare_storage(settings)
    if not settings.hf_hub_offline:
        raise RuntimeError("Benchmark requires HF_HUB_OFFLINE=true")
    output_dir = settings.output_dir / "benchmarks" / "phase1"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = settings.project_root / "sample_data" / "images"
    manager = ModelManager(settings)
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "model_id": settings.vlm_model_id,
        "model_path": str(settings.vlm_model_path),
        "device": settings.vlm_device,
        "dtype": settings.vlm_dtype,
        "attention": settings.vlm_attention,
        "offline": settings.hf_hub_offline,
        "max_image_edge": settings.vlm_max_image_edge,
        "max_visual_tokens": settings.vlm_max_visual_tokens,
        "vram_before_load": get_gpu_memory().model_dump(),
        "runs": [],
    }
    try:
        first = manager.load_model()
        report["cold_load_time_s"] = first["load_time_s"]
        report["vram_after_first_load"] = first["gpu_memory"]
        for image_name, prompt in CASES:
            report["runs"].append(run_case(manager, sample_dir / image_name, prompt))
        report["state_after_10_runs"] = manager.status()
        inference_allocated = [run["gpu_allocated_gb"] for run in report["runs"]]
        latencies = [run["inference_latency_ms"] for run in report["runs"]]
        report["performance"] = {
            "continuous_inferences": len(report["runs"]),
            "first_latency_ms": latencies[0],
            "warm_average_latency_ms": round(statistics.mean(latencies[1:]), 2),
            "fastest_latency_ms": min(latencies),
            "slowest_latency_ms": max(latencies),
            "first_allocated_gb": inference_allocated[0],
            "last_allocated_gb": inference_allocated[-1],
            "allocated_growth_gb": round(inference_allocated[-1] - inference_allocated[0], 3),
            "peak_allocated_gb": max(run["peak_gpu_allocated_gb"] for run in report["runs"]),
        }
        report["first_unload"] = manager.unload_model()
        second = manager.load_model()
        report["second_load"] = second
        report["second_load_inference"] = run_case(
            manager, sample_dir / "foundation-demo.png", "Briefly describe this image."
        )
        report["second_unload"] = manager.unload_model()
    finally:
        if manager.state.value != "UNLOADED":
            manager.unload_model()
    report["torch_version"] = torch.__version__
    report["cuda_version"] = torch.version.cuda
    report["gpu_name"] = torch.cuda.get_device_name(0)
    target = output_dir / "phase1_vlm_benchmark.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["performance"], indent=2))
    print(f"Benchmark written to {target}")


if __name__ == "__main__":
    main()

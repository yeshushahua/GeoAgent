"""HTTP-level Gradio Phase 1 acceptance using real component event endpoints."""
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
        assert "Backend unavailable" in status, status
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
    status, info = client.predict(api_name="/load_model")
    assert info["model"]["state"] == "READY"
    answer, metrics, status, info = client.predict(
        handle_file(str(source)), "Describe this image briefly.", 64, api_name="/analyze"
    )
    assert answer and metrics["model"] == "Qwen3-VL-4B-Instruct"
    assert metrics["device"] == "cuda:0" and metrics["dtype"] == "bfloat16"
    assert metrics["latency_ms"] > 0 and metrics["gpu"]["peak_allocated_gb"] > 0
    assert info["model"]["state"] == "READY"
    status, info = client.predict(api_name="/unload_model")
    assert info["model"]["state"] == "UNLOADED"
    output = {
        "answer": answer,
        "metrics": metrics,
        "final_model_state": info["model"],
    }
    target = settings.output_dir / "benchmarks" / "phase1" / "gradio_acceptance.json"
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info(
        "Gradio Phase 1 PASS: page, upload/preview, API load, real VLM analyze, metrics, unload"
    )
    logging.info("Acceptance written to %s", target)


if __name__ == "__main__":
    main()

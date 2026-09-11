"""HTTP-level Gradio acceptance using its real upload and event endpoints."""
import argparse
import json
import logging
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

from backend.app.core.config import get_settings
from backend.app.core.logging import configure_logging
from backend.app.services.storage import prepare_storage
from frontend.app import PLACEHOLDER


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
    assert "Ready" in status, status
    assert info["cuda_available"] and "RTX 4090" in info["gpu_name"]
    source = settings.project_root / "sample_data" / "images" / "foundation-demo.png"
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (480, 300), "#edf3f1")
        draw = ImageDraw.Draw(image)
        draw.rectangle((40, 60, 160, 200), fill="#1c6660")
        draw.ellipse((220, 70, 370, 220), fill="#efb45d")
        draw.text((40, 255), "GeoAgent / Phase 0 / synthetic demo", fill="#183a38")
        image.save(source)
    preview = client.predict(handle_file(str(source)), api_name="/preview")
    result_path = Path(preview["path"] if isinstance(preview, dict) else preview)
    with Image.open(source) as original, Image.open(result_path) as result:
        assert original.size == result.size
        assert original.convert("RGB").tobytes() == result.convert("RGB").tobytes()
    _, history = client.predict("hello", [], api_name="/chat")
    assert "hello" in json.dumps(history)
    assert PLACEHOLDER in json.dumps(history)
    logging.info("Gradio HTTP test PASS: page 200, real backend RTX 4090 status, image upload/preview pixel equality, chat placeholder")
    logging.info("System: %s", json.dumps(info))


if __name__ == "__main__":
    main()

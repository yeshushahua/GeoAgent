"""Gradio UI; model operations go through FastAPI only."""
import io
import logging
import os

import httpx

from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import configure_logging
from backend.app.services.storage import prepare_storage

logger = logging.getLogger("geoagent")
PLACEHOLDER = "GeoAgent is ready. Multimodal model integration begins in Phase 1."


def _model_markdown(system: dict, model: dict) -> str:
    memory = model.get("gpu_memory", {})
    state = model.get("state", "UNKNOWN")
    marker = "●" if state == "READY" else "○"
    return (
        f"### Qwen3-VL-4B-Instruct {marker} {state}\n"
        f"GPU: **{system.get('gpu_name') or 'Unavailable'}** · "
        f"VRAM allocated: **{memory.get('allocated_gb', 0):.3f} / "
        f"{system.get('gpu_vram_gb') or 0:.2f} GiB**\n\n"
        f"Device: {model.get('device', '-')} · dtype: {model.get('dtype', '-')} · "
        f"attention: {model.get('attention', '-')}"
    )


def fetch_status(settings: Settings) -> tuple[str, dict]:
    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            health = client.get(f"{settings.api_base_url}/health")
            health.raise_for_status()
            system = client.get(f"{settings.api_base_url}/system")
            system.raise_for_status()
            model = client.get(f"{settings.api_base_url}/models/vlm/status")
            model.raise_for_status()
        health_data, info, model_info = health.json(), system.json(), model.json()
        if health_data.get("status") != "ok":
            raise ValueError("Backend health is not ok")
        ready = info["cuda_available"] and "RTX 4090" in (info["gpu_name"] or "")
        title = "System ● Ready" if ready else "System ● GPU attention required"
        return (
            f"### {title}\nBackend: **Online** · CUDA: **{info['cuda_available']}**\n\n"
            f"Storage: {info['storage_root']}\n\n{_model_markdown(info, model_info)}",
            {"system": info, "model": model_info, "tools": "No tools used in Phase 1."},
        )
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Backend unavailable: %s", exc)
        return "### Backend unavailable\nStart the FastAPI backend and refresh status.", {
            "backend": "unavailable"
        }


def model_action(settings: Settings, action: str) -> tuple[str, dict]:
    try:
        with httpx.Client(timeout=600, trust_env=False) as client:
            response = client.post(f"{settings.api_base_url}/models/vlm/{action}")
            response.raise_for_status()
        return fetch_status(settings)
    except httpx.HTTPStatusError as exc:
        detail = exc.response.json().get("error", {})
        return f"### Model {action} failed\n{detail.get('message', str(exc))}", detail
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Model action failed: %s", exc)
        return f"### Backend unavailable\n{exc}", {"backend": "unavailable"}


def analyze(settings: Settings, image, prompt: str, max_new_tokens: int):
    if image is None:
        return "Upload an image first.", {}, *fetch_status(settings)
    if not prompt or not prompt.strip():
        return "Enter a prompt first.", {}, *fetch_status(settings)
    payload = io.BytesIO()
    image.convert("RGB").save(payload, format="PNG")
    try:
        with httpx.Client(timeout=600, trust_env=False) as client:
            response = client.post(
                f"{settings.api_base_url}/models/vlm/infer",
                files={"image": ("upload.png", payload.getvalue(), "image/png")},
                data={"prompt": prompt.strip(), "max_new_tokens": int(max_new_tokens)},
            )
            response.raise_for_status()
        result = response.json()
        metadata = {
            "model": result["model"],
            "latency_ms": result["latency_ms"],
            "device": result["device"],
            "dtype": result["dtype"],
            "image": result["image"],
            "generation": result["generation"],
            "gpu": result["gpu"],
            "tools": "No tools used in Phase 1.",
        }
        status, details = fetch_status(settings)
        return result["text"], metadata, status, details
    except httpx.HTTPStatusError as exc:
        detail = exc.response.json().get("error", {})
        status, details = fetch_status(settings)
        return f"Request failed: {detail.get('message', str(exc))}", detail, status, details
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        status, details = fetch_status(settings)
        return f"Backend unavailable: {exc}", {}, status, details


def chat(message: str, history: list | None):
    history = list(history or [])
    if message.strip():
        history.extend(
            [{"role": "user", "content": message}, {"role": "assistant", "content": PLACEHOLDER}]
        )
    return "", history


def preview_image(image):
    return image


def build_ui(settings: Settings | None = None):
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    prepare_storage(settings)
    for key in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(key, "")
        os.environ[key] = ",".join(filter(None, [existing, "127.0.0.1", "localhost", "::1"]))
    import gradio as gr

    with gr.Blocks(title="GeoAgent", analytics_enabled=False) as demo:
        gr.Markdown("# GeoAgent\nQwen3-VL-4B Multimodal Baseline · **Phase 1**")
        status = gr.Markdown("### Connecting to backend…")
        with gr.Row():
            load = gr.Button("Load Model", variant="primary")
            unload = gr.Button("Unload Model")
            refresh = gr.Button("Refresh Status")
        with gr.Row():
            input_image = gr.Image(
                label="Input Image · Upload", type="pil", format="png", sources=["upload"]
            )
            result_image = gr.Image(
                label="Result / Preview", type="pil", format="png", interactive=False
            )
        prompt = gr.Textbox(
            label="Prompt", value="Describe this image in detail.", lines=3
        )
        max_tokens = gr.Slider(
            minimum=64,
            maximum=512,
            value=settings.vlm_default_max_new_tokens,
            step=32,
            label="Max new tokens",
        )
        analyze_button = gr.Button("Analyze", variant="primary")
        response = gr.Markdown(label="Qwen3-VL Response")
        with gr.Accordion("System / Model", open=True):
            details = gr.JSON(label="Live backend / model information")
            inference_details = gr.JSON(label="Inference metrics")
            gr.Markdown("No tools used in Phase 1.")
        input_image.change(preview_image, input_image, result_image, api_name="preview")
        refresh.click(lambda: fetch_status(settings), outputs=[status, details], api_name="status")
        load.click(
            lambda: model_action(settings, "load"),
            outputs=[status, details],
            api_name="load_model",
        )
        unload.click(
            lambda: model_action(settings, "unload"),
            outputs=[status, details],
            api_name="unload_model",
        )
        analyze_button.click(
            lambda image, prompt, tokens: analyze(settings, image, prompt, tokens),
            [input_image, prompt, max_tokens],
            [response, inference_details, status, details],
            api_name="analyze",
        )
        demo.load(lambda: fetch_status(settings), outputs=[status, details])
    return demo


if __name__ == "__main__":
    config = get_settings()
    build_ui(config).launch(
        server_name=config.gradio_host,
        server_port=config.gradio_port,
        share=False,
        theme="soft",
    )

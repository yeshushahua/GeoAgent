"""A lightweight UI; all system status comes from the backend."""
import logging
import os

import httpx

from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import configure_logging
from backend.app.services.storage import prepare_storage

logger = logging.getLogger("geoagent")
PLACEHOLDER = "GeoAgent is ready. Multimodal model integration begins in Phase 1."


def fetch_status(settings: Settings) -> tuple[str, dict]:
    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            health = client.get(f"{settings.api_base_url}/health")
            health.raise_for_status()
            system = client.get(f"{settings.api_base_url}/system")
            system.raise_for_status()
        health_data, info = health.json(), system.json()
        if health_data.get("status") != "ok":
            raise ValueError("Backend health is not ok")
        ready = info["cuda_available"] and "RTX 4090" in (info["gpu_name"] or "")
        title = "System ● Ready" if ready else "System ● GPU attention required"
        return (f"### {title}\nBackend: **Online** · CUDA: **{info['cuda_available']}**\n\n"
                f"GPU: **{info['gpu_name'] or 'Unavailable'}**\n\nStorage: `{info['storage_root']}`", info)
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Backend unavailable: %s", exc)
        return "### Backend unavailable\nStart the FastAPI backend and refresh status.", {"backend": "unavailable"}


def chat(message: str, history: list | None):
    history = list(history or [])
    if message.strip():
        history.extend([{"role": "user", "content": message}, {"role": "assistant", "content": PLACEHOLDER}])
    return "", history


def preview_image(image):
    return image


def build_ui(settings: Settings | None = None):
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    prepare_storage(settings)
    # Gradio checks localhost at startup using inherited proxy settings.
    for key in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(key, "")
        os.environ[key] = ",".join(filter(None, [existing, "127.0.0.1", "localhost", "::1"]))
    import gradio as gr

    with gr.Blocks(title="GeoAgent", analytics_enabled=False) as demo:
        gr.Markdown("# GeoAgent\nMultimodal AI Agent for Visual & Geospatial Analysis · **Phase 0**")
        status = gr.Markdown("### Connecting to backend…")
        with gr.Row():
            input_image = gr.Image(label="Input Image · Upload", type="pil", format="png", sources=["upload"])
            result_image = gr.Image(label="Result / Preview", type="pil", format="png", interactive=False)
        gr.Markdown("## Chat\nFoundation preview: model integration begins in Phase 1.")
        chatbot = gr.Chatbot(label="User / Agent", height=260)
        message = gr.Textbox(label="Message", placeholder="Say hello…")
        send = gr.Button("Send", variant="primary")
        with gr.Accordion("System / Tool Execution", open=True):
            details = gr.JSON(label="Live backend system information")
            refresh = gr.Button("Refresh status")
        input_image.change(preview_image, input_image, result_image, api_name="preview")
        message.submit(chat, [message, chatbot], [message, chatbot], api_name="chat")
        send.click(chat, [message, chatbot], [message, chatbot], api_name=False)
        refresh.click(lambda: fetch_status(settings), outputs=[status, details], api_name="status")
        demo.load(lambda: fetch_status(settings), outputs=[status, details])
    return demo


if __name__ == "__main__":
    config = get_settings()
    build_ui(config).launch(server_name=config.gradio_host, server_port=config.gradio_port,
                            share=False, theme="soft")

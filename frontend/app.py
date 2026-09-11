"""Gradio UI; user analysis goes through the Phase 2 Tool API."""
import io
import logging
import os

import httpx
from PIL import Image

from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import configure_logging
from backend.app.services.storage import prepare_storage

logger = logging.getLogger("geoagent")
PLACEHOLDER = "GeoAgent is ready. Autonomous agent behavior begins in a later phase."


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
            tools = client.get(f"{settings.api_base_url}/tools")
            tools.raise_for_status()
        health_data, info, model_info, tool_info = health.json(), system.json(), model.json(), tools.json()
        if health_data.get("status") != "ok":
            raise ValueError("Backend health is not ok")
        ready = info["cuda_available"] and "RTX 4090" in (info["gpu_name"] or "")
        title = "System ● Ready" if ready else "System ● GPU attention required"
        return (
            f"### {title}\nBackend: **Online** · CUDA: **{info['cuda_available']}**\n\n"
            f"Storage: {info['storage_root']}\n\n{_model_markdown(info, model_info)}",
            {"system": info, "model": model_info, "tools": [item["name"] for item in tool_info]},
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


def _tool_markdown(result: dict) -> str:
    metadata = result.get("metadata", {})
    success = result.get("success", False)
    tool = result.get("tool", "unknown")
    model_step = " → Qwen3-VL" if tool == "analyze_image" else ""
    return (
        "### Tool Execution\n"
        f"Tool: **{tool}**  \n"
        f"Status: **{'SUCCESS' if success else 'FAILED'}**  \n"
        f"Execution ID: `{metadata.get('execution_id', '-')}`  \n"
        f"Duration: **{metadata.get('duration_ms', 0):.2f} ms**  \n\n"
        f"Sequence: `{tool}{model_step} → {'completed' if success else 'failed'}`"
    )


def _post_tool(settings: Settings, tool_name: str, image, fields: dict) -> dict:
    payload = io.BytesIO()
    image.convert("RGB").save(payload, format="PNG")
    with httpx.Client(timeout=900, trust_env=False) as client:
        response = client.post(
            f"{settings.api_base_url}/tools/{tool_name}/execute",
            files={"image": ("upload.png", payload.getvalue(), "image/png")},
            data={key: str(value) for key, value in fields.items() if value is not None},
        )
    return response.json()


def analyze(settings: Settings, image, prompt: str, max_new_tokens: int):
    if image is None:
        yield "Upload an image first.", {}, *fetch_status(settings), "### Tool Execution\nNo execution."
        return
    if not prompt or not prompt.strip():
        yield "Enter a prompt first.", {}, *fetch_status(settings), "### Tool Execution\nNo execution."
        return
    try:
        status_text, details = fetch_status(settings)
        state = details.get("model", {}).get("state")
        if state == "ERROR":
            message = details["model"].get("last_error") or "Unknown model error"
            yield f"Model error: {message}", {}, status_text, details, "### Tool Execution\nNo execution."
            return
        if state == "UNLOADED":
            yield "**Loading Qwen3-VL...**", {}, "### Loading Qwen3-VL...", details, "### Tool Execution\nPreparing `analyze_image`..."
        yield "**Analyzing...**", {}, "### Analyzing...", details, "### Tool Execution\nRunning `analyze_image → Qwen3-VL`..."
        result = _post_tool(
            settings,
            "analyze_image",
            image,
            {"prompt": prompt.strip(), "max_new_tokens": int(max_new_tokens)},
        )
        status, details = fetch_status(settings)
        if not result.get("success"):
            error = result.get("error", {})
            yield f"Request failed: {error.get('message', 'Unknown tool error')}", result, status, details, _tool_markdown(result)
            return
        yield result["data"]["answer"], result, status, details, _tool_markdown(result)
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        status, details = fetch_status(settings)
        yield f"Backend unavailable: {exc}", {}, status, details, "### Tool Execution\nRequest failed before execution."


def execute_selected_tool(settings, tool_name, image, prompt, tokens, x1, y1, x2, y2):
    if image is None:
        return "Upload an image first.", {}, None, "### Tool Execution\nNo execution."
    fields = {
        "prompt": prompt.strip() if prompt else None,
        "max_new_tokens": int(tokens),
        "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
    }
    try:
        result = _post_tool(settings, tool_name, image, fields)
        preview = image
        if result.get("success") and result.get("artifacts"):
            artifact_path = result["artifacts"][0]["path"]
            with Image.open(artifact_path) as opened:
                preview = opened.copy()
        if result.get("success"):
            answer = result.get("data", {}).get("answer") or "Tool completed successfully."
        else:
            answer = f"Tool failed: {result.get('error', {}).get('message', 'Unknown error')}"
        return answer, result, preview, _tool_markdown(result)
    except (httpx.HTTPError, ValueError, KeyError, OSError) as exc:
        return f"Tool request failed: {exc}", {}, image, "### Tool Execution\nRequest failed before execution."


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

    def analyze_event(image, prompt, tokens):
        yield from analyze(settings, image, prompt, tokens)

    with gr.Blocks(title="GeoAgent", analytics_enabled=False) as demo:
        gr.Markdown("# GeoAgent\nTool-enabled multimodal system · **Phase 2**")
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
        tool_execution = gr.Markdown("### Tool Execution\nNo execution yet.")
        with gr.Accordion("Manual Tool Execution", open=False):
            tool_choice = gr.Dropdown(
                ["inspect_image", "crop_image", "analyze_image"],
                value="inspect_image",
                label="Tool",
            )
            with gr.Row():
                x1 = gr.Number(value=0, precision=0, label="x1")
                y1 = gr.Number(value=0, precision=0, label="y1")
                x2 = gr.Number(value=256, precision=0, label="x2")
                y2 = gr.Number(value=256, precision=0, label="y2")
            run_tool = gr.Button("Execute Tool")
        with gr.Accordion("System / Model", open=True):
            details = gr.JSON(label="Live backend / model information")
            inference_details = gr.JSON(label="Inference metrics")
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
            analyze_event,
            [input_image, prompt, max_tokens],
            [response, inference_details, status, details, tool_execution],
            api_name="analyze",
        )
        run_tool.click(
            lambda tool, image, prompt, tokens, left, top, right, bottom: execute_selected_tool(
                settings, tool, image, prompt, tokens, left, top, right, bottom
            ),
            [tool_choice, input_image, prompt, max_tokens, x1, y1, x2, y2],
            [response, inference_details, result_image, tool_execution],
            api_name="execute_tool",
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

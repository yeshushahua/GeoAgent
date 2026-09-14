"""Chinese Gradio UI; the primary analysis path goes through VisionAgent."""
import io
import json
import logging
import os
from pathlib import Path

import httpx
from PIL import Image

from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import configure_logging
from backend.app.services.storage import prepare_storage
from backend.app.tools import build_tool_registry

logger = logging.getLogger("geoagent")
PLACEHOLDER = "GeoAgent 已就绪。"
MANUAL_PARAMETER_ORDER = (
    "prompt", "max_new_tokens", "classes", "confidence", "iou_threshold",
    "x1", "y1", "x2", "y2", "boxes", "detection_ids", "bands", "stretch",
    "lower_percentile", "upper_percentile", "max_size", "resampling",
    "region", "row_start", "row_end", "col_start", "col_end",
)


def manual_tool_definitions() -> dict[str, dict]:
    return {item["name"]: item for item in build_tool_registry().list_tools()}


def manual_parameter_visibility(definition: dict) -> dict[str, bool]:
    properties = definition["input_schema"].get("properties", {})
    return {name: name in properties for name in MANUAL_PARAMETER_ORDER}


def _array_schema(schema: dict) -> bool:
    return schema.get("type") == "array" or any(
        item.get("type") == "array" for item in schema.get("anyOf", [])
    )


def build_manual_form_fields(definition: dict, values: dict) -> dict:
    """Serialize only fields present in the selected tool's Pydantic JSON Schema."""
    properties = definition["input_schema"].get("properties", {})
    fields = {}
    for name, schema in properties.items():
        if name == "image_path" or name not in values:
            continue
        value = values[name]
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if _array_schema(schema):
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    value = [item.strip() for item in value.split(",") if item.strip()]
            if not isinstance(value, list):
                raise ValueError(f"{name} 必须是 JSON 数组或逗号分隔列表")
            fields[name] = json.dumps(value, ensure_ascii=False)
        else:
            fields[name] = value.strip() if isinstance(value, str) else value
    return fields


def _model_markdown(system: dict, model: dict) -> str:
    memory = model.get("gpu_memory", {})
    state = model.get("state", "UNKNOWN")
    marker = "●" if state == "READY" else "○"
    return (
        f"### Qwen3-VL-4B-Instruct {marker} {state}\n"
        f"GPU：**{system.get('gpu_name') or '不可用'}** · "
        f"已分配显存：**{memory.get('allocated_gb', 0):.3f} / "
        f"{system.get('gpu_vram_gb') or 0:.2f} GiB**\n\n"
        f"设备：{model.get('device', '-')} · dtype：{model.get('dtype', '-')} · "
        f"attention：{model.get('attention', '-')}"
    )


def _detector_markdown(detector: dict) -> str:
    state = detector.get("state", "UNKNOWN")
    marker = "●" if state == "READY" else "○"
    return (
        f"### YOLO11s COCO Detector {marker} {state}\n"
        f"设备：{detector.get('device', '-')} · "
        f"加载次数：{detector.get('load_count', 0)} · "
        f"权重：{detector.get('model_path', '-')}"
    )


def _advanced_model_markdown(title: str, model: dict) -> str:
    state = model.get("state", "UNKNOWN")
    marker = "●" if state == "READY" else "○"
    return (
        f"### {title} {marker} {state}\n"
        f"设备：{model.get('device', '-')} · 加载次数：{model.get('load_count', 0)} · "
        f"离线：{model.get('offline', False)}\n\n权重：{model.get('model_path', '-')}"
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
            detector = client.get(f"{settings.api_base_url}/models/detector/status")
            detector.raise_for_status()
            open_vocab = client.get(f"{settings.api_base_url}/models/open-vocabulary/status")
            open_vocab.raise_for_status()
            segmentation = client.get(f"{settings.api_base_url}/models/segmentation/status")
            segmentation.raise_for_status()
            tools = client.get(f"{settings.api_base_url}/tools")
            tools.raise_for_status()
        health_data, info, model_info, detector_info, open_vocab_info, segmentation_info, tool_info = (
            health.json(), system.json(), model.json(), detector.json(),
            open_vocab.json(), segmentation.json(), tools.json()
        )
        if health_data.get("status") != "ok":
            raise ValueError("Backend health is not ok")
        ready = info["cuda_available"] and "RTX 4090" in (info["gpu_name"] or "")
        title = "系统 ● 就绪" if ready else "系统 ● GPU 需要检查"
        return (
            f"### {title}\n后端：**在线** · CUDA：**{info['cuda_available']}**\n\n"
            f"存储目录：{info['storage_root']}\n\n{_model_markdown(info, model_info)}\n\n"
            f"{_detector_markdown(detector_info)}\n\n"
            f"{_advanced_model_markdown('YOLOE-26s Open Vocabulary', open_vocab_info)}\n\n"
            f"{_advanced_model_markdown('SAM 2.1 Base', segmentation_info)}",
            {
                "system": info,
                "model": model_info,
                "detector": detector_info,
                "open_vocabulary": open_vocab_info,
                "segmentation": segmentation_info,
                "tools": [item["name"] for item in tool_info],
            },
        )
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Backend unavailable: %s", exc)
        return "### 后端不可用\n请启动 FastAPI 后端并刷新状态。", {
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
        return f"### 模型操作失败\n{detail.get('message', str(exc))}", detail
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Model action failed: %s", exc)
        return f"### 后端不可用\n{exc}", {"backend": "unavailable"}


def _tool_markdown(result: dict) -> str:
    metadata = result.get("metadata", {})
    success = result.get("success", False)
    tool = result.get("tool", "unknown")
    model_step = {
        "analyze_image": " → Qwen3-VL",
        "detect_objects": " → YOLO11s",
        "detect_open_vocab": " → YOLOE-26s",
        "segment_objects": " → SAM 2.1 Base",
    }.get(tool, "")
    arguments = metadata.get("arguments_summary", {})
    timing = []
    timing.append(f"本次模型加载：**{metadata.get('model_load_ms', 0):.2f} ms**")
    if metadata.get("prompt_encoding_ms", 0):
        timing.append(f"提示编码：**{metadata['prompt_encoding_ms']:.2f} ms**")
    timing.append(f"模型推理：**{metadata.get('inference_ms', 0):.2f} ms**")
    timing.append(f"工具开销：**{metadata.get('tool_overhead_ms', 0):.2f} ms**")
    timing_text = ("  \n" + " · ".join(timing)) if timing else ""
    return (
        "### 工具调用\n"
        f"Tool：**{tool}**  \n"
        f"状态：**{'成功' if success else '失败'}**  \n"
        f"Execution ID：`{metadata.get('execution_id', '-')}`  \n"
        f"Tool 总耗时：**{metadata.get('duration_ms', 0):.2f} ms**{timing_text}  \n"
        f"安全参数：`{json.dumps(arguments, ensure_ascii=False)}`  \n\n"
        f"执行：`{tool}{model_step} → {'完成' if success else '失败'}`"
    )


def _upload_payload(image, raster_path=None):
    if raster_path:
        path = Path(raster_path)
        return path.name, path.read_bytes(), "image/tiff"
    payload = io.BytesIO()
    image.convert("RGB").save(payload, format="PNG")
    return "upload.png", payload.getvalue(), "image/png"


def _post_tool(
    settings: Settings, tool_name: str, image, fields: dict, raster_path=None
) -> dict:
    filename, content, mime_type = _upload_payload(image, raster_path)
    with httpx.Client(timeout=900, trust_env=False) as client:
        response = client.post(
            f"{settings.api_base_url}/tools/{tool_name}/execute",
            files={"image": (filename, content, mime_type)},
            data={key: str(value) for key, value in fields.items() if value is not None},
        )
    return response.json()


def _post_agent(
    settings: Settings, image, message: str, max_new_tokens: int, raster_path=None
) -> dict:
    filename, content, mime_type = _upload_payload(image, raster_path)
    with httpx.Client(timeout=1200, trust_env=False) as client:
        response = client.post(
            f"{settings.api_base_url}/agent/run",
            files={"image": (filename, content, mime_type)},
            data={
                "message": message,
                "max_steps": settings.agent_default_max_steps,
                "max_new_tokens": int(max_new_tokens),
            },
        )
    return response.json()


def _agent_markdown(result: dict) -> str:
    if not result:
        return "### Agent 执行过程\n尚未执行。"
    workflow = result.get("workflow", {})
    lines = [
        "### Workflow",
        f"Run ID：`{result.get('run_id', '-')}`",
        f"当前图像：`{workflow.get('active_image_artifact_id') or '-'}`",
        f"当前 Raster：`{workflow.get('active_raster_artifact_id') or '-'}`",
        "",
    ]
    for step in result.get("steps", []):
        if step.get("decision_type") == "final":
            lines.append(f"{step['index']}. **生成最终回答** · {step.get('duration_ms', 0):.2f} ms")
            continue
        tool = step.get("tool_name") or "调用已阻止"
        marker = "✓" if step.get("success") else "✗"
        lines.append(
            f"{step['index']}. `{tool}` {marker} · 总计 {step.get('duration_ms', 0):.2f} ms "
            f"（Planner {step.get('planner_duration_ms', 0):.2f} ms · "
            f"Tool {step.get('tool_duration_ms', 0):.2f} ms）"
        )
        if step.get("tool_duration_ms", 0):
            lines.append(
                "   Tool 明细："
                f"加载 {step.get('model_load_duration_ms', 0):.2f} ms · "
                f"提示编码 {step.get('prompt_encoding_duration_ms', 0):.2f} ms · "
                f"推理 {step.get('model_inference_duration_ms', 0):.2f} ms · "
                f"开销 {step.get('tool_overhead_ms', 0):.2f} ms"
            )
        arguments = step.get("arguments_summary", {})
        if arguments:
            lines.append(f"   参数：`{json.dumps(arguments, ensure_ascii=False)}`")
        source_id = step.get("observation_summary", {}).get("source_artifact_id")
        produced_ids = step.get("artifact_ids", [])
        if source_id:
            lines.append(f"   来源：`{source_id}`")
        if produced_ids:
            lines.append("   产物：" + "、".join(f"`{item}`" for item in produced_ids))
        if tool == "detect_objects" and step.get("success"):
            observation = step.get("observation_summary", {})
            counts = observation.get("class_counts", {})
            summary = "、".join(f"{name} × {count}" for name, count in counts.items()) or "未检出目标"
            lines.append(
                f"   检测总数：**{observation.get('detection_count', 0)}** · {summary}"
            )
        if tool == "detect_open_vocab" and step.get("success"):
            observation = step.get("observation_summary", {})
            counts = observation.get("class_counts", {})
            summary = "、".join(f"{name} × {count}" for name, count in counts.items()) or "未检出目标"
            lines.append(
                f"   开放检测总数：**{observation.get('detection_count', 0)}** · {summary}"
            )
            detection_ids = observation.get("workflow_detection_ids", [])
            if detection_ids:
                lines.append("   Detection IDs：" + "、".join(f"`{item}`" for item in detection_ids))
        if tool == "segment_objects" and step.get("success"):
            observation = step.get("observation_summary", {})
            ratios = [item.get("mask_area_ratio", 0) for item in observation.get("segments", [])]
            summary = "、".join(f"{value:.2%}" for value in ratios) or "无实例"
            failed = len(observation.get("failures", []))
            failure_text = f" · 失败：**{failed}**" if failed else ""
            lines.append(
                f"   分割实例：**{observation.get('segment_count', 0)}**"
                f"{failure_text} · 面积比例：{summary}"
            )
        if tool == "inspect_raster" and step.get("success"):
            observation = step.get("observation_summary", {})
            lines.append(
                f"   Raster：**{observation.get('width')} × {observation.get('height')}** · "
                f"波段 **{observation.get('band_count')}** · "
                f"CRS `{observation.get('crs') or 'unknown'}`"
            )
        if tool == "raster_preview" and step.get("success"):
            observation = step.get("observation_summary", {})
            lines.append(
                f"   Preview：**{observation.get('preview_width')} × "
                f"{observation.get('preview_height')}** · "
                f"Bands {observation.get('bands')} · {observation.get('stretch')}"
            )
        if tool == "crop_raster" and step.get("success"):
            observation = step.get("observation_summary", {})
            metadata = observation.get("metadata", {})
            lines.append(
                f"   Raster Crop：**{metadata.get('width')} × {metadata.get('height')}** · "
                f"CRS `{metadata.get('crs') or 'unknown'}`"
            )
        if tool == "raster_statistics" and step.get("success"):
            observation = step.get("observation_summary", {})
            lines.append(
                f"   波段统计：**{len(observation.get('bands', []))}** · "
                f"读取策略 `{observation.get('read_strategy')}`"
            )
    categories = workflow.get("categories", {})
    if categories:
        lines.extend(["", "**聚合结果**"])
        for name, item in categories.items():
            lines.append(
                f"- {name}：检测 {item.get('detected', 0)} · "
                f"分割 {item.get('segmented', 0)} · "
                f"联合面积占比 {item.get('union_mask_area_ratio', 0):.2%}"
            )
    artifacts = workflow.get("artifacts", [])
    if artifacts:
        lines.extend(["", "**Artifact 依赖**"])
        for artifact in artifacts:
            parent = artifact.get("parent_artifact_id") or "-"
            lines.append(
                f"- `{artifact.get('artifact_id')}` · {artifact.get('artifact_type')} · "
                f"parent `{parent}`"
            )
    metadata = result.get("metadata", {})
    lines.extend([
        "",
        f"总耗时：**{metadata.get('total_duration_ms', 0):.2f} ms** · "
        f"Planner：**{metadata.get('planner_duration_ms', 0):.2f} ms** · "
        f"Tools：**{metadata.get('tool_duration_ms', 0):.2f} ms** · "
        f"框架阶段：**{metadata.get('framework_overhead_ms', 0):.2f} ms**",
        f"加载明细：Agent 模型 **{metadata.get('agent_model_load_duration_ms', 0):.2f} ms** · "
        f"Tool 模型 **{metadata.get('tool_model_load_duration_ms', 0):.2f} ms** · "
        f"框架其余开销 **{metadata.get('framework_runtime_overhead_ms', 0):.2f} ms**",
        f"Tools 明细：提示编码 **{metadata.get('prompt_encoding_duration_ms', 0):.2f} ms** · "
        f"推理 **{metadata.get('model_inference_duration_ms', 0):.2f} ms** · "
        f"工具开销 **{metadata.get('tool_overhead_ms', 0):.2f} ms**",
    ])
    return "\n".join(lines)


def _ui_safe_result(value):
    """Keep debug JSON readable without exposing machine-local absolute paths."""
    if isinstance(value, list):
        return [_ui_safe_result(item) for item in value]
    if isinstance(value, dict):
        converted = {key: _ui_safe_result(item) for key, item in value.items()}
        if "path" in converted and "mime_type" in converted:
            converted["artifact_path"] = os.path.basename(converted.pop("path"))
        for key in ("source_image_path", "mask_artifact_path", "overlay_artifact_path"):
            if isinstance(converted.get(key), str):
                converted[key] = os.path.basename(converted[key])
        return converted
    return value


def _raster_markdown(result: dict) -> str:
    raster = result.get("workflow", {}).get("raster")
    if not raster:
        return "### Raster Information\n尚未载入遥感栅格。"
    metadata = raster.get("metadata", {})
    bounds = metadata.get("bounds") or {}
    crs = metadata.get("crs") or "unknown"
    lines = [
        "### Raster Information",
        f"Artifact：{raster.get('artifact_id', '-')} · 类型：{raster.get('artifact_type', '-')}",
        f"尺寸：**{metadata.get('width', '-')} × {metadata.get('height', '-')}** · "
        f"波段：**{metadata.get('band_count', '-')}** · dtype："
        f"{', '.join(metadata.get('dtypes', [])) or '-'}",
        f"CRS：{crs} · 分辨率：**{metadata.get('resolution_x', '-')} × "
        f"{metadata.get('resolution_y', '-')}** · NoData：{metadata.get('nodata')}",
        (
            "Bounds："
            f"left={bounds.get('left', '-')}, bottom={bounds.get('bottom', '-')}, "
            f"right={bounds.get('right', '-')}, top={bounds.get('top', '-')}"
        ),
    ]
    statistics = raster.get("statistics")
    if statistics:
        lines.append("")
        lines.append("**Band Statistics**")
        for item in statistics.get("bands", []):
            lines.append(
                f"- Band {item.get('band')}：min {item.get('min')} · max {item.get('max')} · "
                f"mean {item.get('mean')} · std {item.get('std')} · "
                f"valid {item.get('valid_pixel_count')} · NoData {item.get('nodata_count')}"
            )
    return "\n".join(lines)


def _display_preview(result: dict, fallback):
    for artifact in reversed(result.get("artifacts", [])):
        mime_type = artifact.get("mime_type") or ""
        if artifact.get("type") == "image" and mime_type.startswith("image/"):
            try:
                with Image.open(artifact["path"]) as opened:
                    return opened.copy()
            except (OSError, KeyError):
                continue
    return fallback


def _mask_gallery(result: dict) -> list[str]:
    return [
        artifact["path"] for artifact in result.get("artifacts", [])
        if artifact.get("type") == "mask" and artifact.get("path")
    ]


def analyze(
    settings: Settings, image, prompt: str, max_new_tokens: int, raster_path=None
):
    if image is None and not raster_path:
        yield (
            "请先上传普通图像或 GeoTIFF。",
            {}, *fetch_status(settings), "### Agent 执行过程\n尚未执行。",
            "### Raster Information\n尚未载入遥感栅格。", None, [],
        )
        return
    fallback = None if raster_path else image
    if not prompt or not prompt.strip():
        yield (
            "请先输入任务指令。",
            {}, *fetch_status(settings), "### Agent 执行过程\n尚未执行。",
            "### Raster Information\n尚未分析。", fallback, [],
        )
        return
    try:
        status_text, details = fetch_status(settings)
        state = details.get("model", {}).get("state")
        if state == "ERROR":
            message = details["model"].get("last_error") or "Unknown model error"
            yield (
                f"模型错误：{message}", {}, status_text, details,
                "### Agent 执行过程\n尚未执行。",
                "### Raster Information\n尚未分析。", fallback, [],
            )
            return
        if state == "UNLOADED":
            yield (
                "**正在加载 Qwen3-VL...**", {}, "### 正在加载 Qwen3-VL...", details,
                "### Agent 执行过程\n正在准备 Vision Agent...",
                "### Raster Information\n正在识别输入类型…", fallback, [],
            )
        yield (
            "**Agent 正在决策并调用工具...**", {}, "### Agent 正在分析...", details,
            "### Agent 执行过程\n正在执行可观察的决策与工具步骤...",
            "### Raster Information\n正在处理…", fallback, [],
        )
        result = _post_agent(
            settings, image, prompt.strip(), int(max_new_tokens), raster_path
        )
        status, details = fetch_status(settings)
        if not result.get("success"):
            error = result.get("error", {})
            yield (
                f"任务失败：{error.get('message', '未知 Agent 错误')}",
                _ui_safe_result(result), status, details, _agent_markdown(result),
                _raster_markdown(result), fallback, [],
            )
            return
        preview = _display_preview(result, fallback)
        yield (
            result["answer"], _ui_safe_result(result), status, details,
            _agent_markdown(result), _raster_markdown(result), preview,
            _mask_gallery(result),
        )
    except (httpx.HTTPError, ValueError, KeyError, OSError) as exc:
        status, details = fetch_status(settings)
        yield (
            f"后端不可用：{exc}", {}, status, details,
            "### Agent 执行过程\n请求未能执行。",
            "### Raster Information\n请求未能执行。", fallback, [],
        )


def _manual_raster_information(tool_name: str, result: dict) -> str:
    if not result.get("success") or tool_name not in {
        "inspect_raster", "raster_preview", "crop_raster", "raster_statistics"
    }:
        return "### Raster Information\n手动 Raster Tool 尚未产生结果。"
    data = result.get("data", {})
    if tool_name == "raster_statistics":
        lines = ["### Raster Information", "**Band Statistics**"]
        for item in data.get("bands", []):
            lines.append(
                f"- Band {item.get('band')}：min {item.get('min')} · max {item.get('max')} · "
                f"mean {item.get('mean')} · std {item.get('std')} · "
                f"valid {item.get('valid_pixel_count')} · NoData {item.get('nodata_count')}"
            )
        return "\n".join(lines)
    metadata = data.get("metadata", data)
    fake = {
        "workflow": {
            "raster": {
                "artifact_id": "manual-result",
                "artifact_type": "raster_crop" if tool_name == "crop_raster" else "raster",
                "metadata": metadata,
            }
        }
    }
    if tool_name == "raster_preview":
        return (
            "### Raster Information\n"
            f"Preview：{data.get('preview_width')} × {data.get('preview_height')} · "
            f"Bands {data.get('bands')} · {data.get('stretch')} stretch · "
            f"{data.get('resampling')}"
        )
    return _raster_markdown(fake)


def execute_selected_tool(
    settings, tool_name, image, raster_path, prompt, tokens, classes,
    confidence, iou_threshold, x1, y1, x2, y2, boxes, detection_ids,
    bands, stretch, lower_percentile, upper_percentile, max_size, resampling,
    region, row_start, row_end, col_start, col_end,
):
    if image is None and not raster_path:
        return (
            "请先上传普通图像或 GeoTIFF。", {}, None,
            "### 工具调用\n尚未执行。", [],
            "### Raster Information\n尚未载入遥感栅格。",
        )
    fallback = None if raster_path else image
    try:
        definition = manual_tool_definitions()[tool_name]
        fields = build_manual_form_fields(definition, {
            "prompt": prompt, "max_new_tokens": int(tokens), "classes": classes,
            "confidence": confidence, "iou_threshold": iou_threshold,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2, "boxes": boxes,
            "detection_ids": detection_ids, "bands": bands, "stretch": stretch,
            "lower_percentile": lower_percentile,
            "upper_percentile": upper_percentile, "max_size": max_size,
            "resampling": resampling, "region": region,
            "row_start": row_start, "row_end": row_end,
            "col_start": col_start, "col_end": col_end,
        })
        result = _post_tool(settings, tool_name, image, fields, raster_path)
        preview = _display_preview(result, fallback)
        if result.get("success"):
            data = result.get("data", {})
            if tool_name in {"detect_objects", "detect_open_vocab"}:
                counts = data.get("class_counts", {})
                summary = "、".join(f"{name} × {count}" for name, count in counts.items()) or "未检出目标"
                answer = f"检测完成，共 {data.get('detection_count', 0)} 个目标：{summary}。"
            elif tool_name == "segment_objects":
                ratios = "、".join(
                    f"{item.get('mask_area_ratio', 0):.2%}" for item in data.get("segments", [])
                )
                answer = f"分割完成，共 {data.get('segment_count', 0)} 个实例，面积比例：{ratios}。"
            elif tool_name == "inspect_raster":
                answer = (
                    f"Raster 尺寸 {data.get('width')} × {data.get('height')}，"
                    f"{data.get('band_count')} 个波段，CRS {data.get('crs') or 'unknown'}。"
                )
            elif tool_name == "raster_preview":
                answer = (
                    f"Raster Preview 已生成：{data.get('preview_width')} × "
                    f"{data.get('preview_height')}。"
                )
            elif tool_name == "crop_raster":
                metadata = data.get("metadata", {})
                answer = f"Raster 裁剪完成：{metadata.get('width')} × {metadata.get('height')}。"
            elif tool_name == "raster_statistics":
                answer = f"已完成 {len(data.get('bands', []))} 个波段的统计。"
            else:
                answer = data.get("answer") or "工具执行成功。"
        else:
            answer = f"工具执行失败：{result.get('error', {}).get('message', '未知错误')}"
        return (
            answer, _ui_safe_result(result), preview, _tool_markdown(result),
            _mask_gallery(result), _manual_raster_information(tool_name, result),
        )
    except (httpx.HTTPError, ValueError, KeyError, OSError) as exc:
        return (
            f"工具请求失败：{exc}", {}, fallback, "### 工具调用\n请求未能执行。", [],
            "### Raster Information\n请求未能执行。",
        )


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
    definitions = manual_tool_definitions()

    def manual_updates(tool_name):
        definition = definitions[tool_name]
        visibility = manual_parameter_visibility(definition)
        help_text = (
            f"**{tool_name}** — {definition['description']}\n\n"
            f"Schema 字段：`{', '.join(definition['input_schema'].get('properties', {}))}`"
        )
        return [help_text] + [
            gr.update(visible=visibility[name]) for name in MANUAL_PARAMETER_ORDER
        ]

    def analyze_event(image, raster_path, prompt, tokens):
        yield from analyze(settings, image, prompt, tokens, raster_path)

    with gr.Blocks(title="GeoAgent", analytics_enabled=False) as demo:
        gr.Markdown("# GeoAgent\n多模态遥感智能体 · **Phase 7 · Remote Sensing Extension**")
        status = gr.Markdown("### 正在连接后端…")
        with gr.Row():
            load = gr.Button("加载模型", variant="primary")
            unload = gr.Button("卸载模型")
            refresh = gr.Button("刷新状态")
        with gr.Row():
            input_image = gr.Image(
                label="普通图像 · 上传", type="pil", format="png", sources=["upload"]
            )
            raster_upload = gr.File(
                label="GeoTIFF · 上传（.tif / .tiff）",
                type="filepath", file_types=[".tif", ".tiff"],
            )
            result_image = gr.Image(
                label="智能分析结果 / 预览", type="pil", format="png", interactive=False
            )
        raster_information = gr.Markdown(
            "### Raster Information\n尚未载入遥感栅格。"
        )
        mask_gallery = gr.Gallery(
            label="实例 Mask", columns=4, rows=1, height="auto", object_fit="contain"
        )
        prompt = gr.Textbox(
            label="任务指令", value="分析一下这张图片主要有什么内容。", lines=3
        )
        max_tokens = gr.Slider(
            minimum=64,
            maximum=512,
            value=settings.vlm_default_max_new_tokens,
            step=32,
            label="最大生成长度",
        )
        analyze_button = gr.Button("开始分析", variant="primary")
        response = gr.Markdown(label="最终回答")
        tool_execution = gr.Markdown("### Agent 执行过程\n尚未执行。")
        with gr.Accordion("高级 / 手动工具调试", open=False):
            tool_choice = gr.Dropdown(
                list(definitions),
                value="inspect_image",
                label="Tool",
            )
            manual_help = gr.Markdown()
            manual_prompt = gr.Textbox(label="prompt", value="描述这张图片。", visible=False)
            manual_tokens = gr.Slider(
                64, 512, value=settings.vlm_default_max_new_tokens, step=32,
                label="max_new_tokens", visible=False,
            )
            manual_classes = gr.Textbox(
                label="classes", placeholder="yellow helmet, excavator", visible=False,
            )
            with gr.Row():
                confidence = gr.Slider(0.01, 1.0, value=0.25, step=0.01, label="confidence", visible=False)
                iou_threshold = gr.Slider(0.01, 1.0, value=0.45, step=0.01, label="iou_threshold", visible=False)
            with gr.Row():
                x1 = gr.Number(value=0, precision=0, label="x1", visible=False)
                y1 = gr.Number(value=0, precision=0, label="y1", visible=False)
                x2 = gr.Number(value=256, precision=0, label="x2", visible=False)
                y2 = gr.Number(value=256, precision=0, label="y2", visible=False)
            manual_boxes = gr.Textbox(
                label="boxes", value='[{"x1": 0, "y1": 0, "x2": 256, "y2": 256}]',
                lines=3, visible=False,
            )
            manual_detection_ids = gr.Textbox(
                label="detection_ids（仅 Agent Workflow）",
                placeholder="det-001, det-002", visible=False,
            )
            manual_bands = gr.Textbox(
                label="bands（1-based）", placeholder="1, 2, 3", visible=False,
            )
            with gr.Row():
                manual_stretch = gr.Dropdown(
                    ["percentile", "minmax"], value="percentile",
                    label="stretch", visible=False,
                )
                manual_resampling = gr.Dropdown(
                    ["nearest", "bilinear"], value="bilinear",
                    label="resampling", visible=False,
                )
                manual_max_size = gr.Number(
                    value=2048, precision=0, label="max_size", visible=False,
                )
            with gr.Row():
                manual_lower_percentile = gr.Number(
                    value=2, label="lower_percentile", visible=False,
                )
                manual_upper_percentile = gr.Number(
                    value=98, label="upper_percentile", visible=False,
                )
            manual_region = gr.Dropdown(
                [
                    "left_half", "right_half", "top_half", "bottom_half",
                    "top_left_quarter", "top_right_quarter",
                    "bottom_left_quarter", "bottom_right_quarter",
                ],
                value=None, label="region（与像素窗口二选一）", visible=False,
            )
            with gr.Row():
                manual_row_start = gr.Number(value=0, precision=0, label="row_start", visible=False)
                manual_row_end = gr.Number(value=64, precision=0, label="row_end", visible=False)
                manual_col_start = gr.Number(value=0, precision=0, label="col_start", visible=False)
                manual_col_end = gr.Number(value=64, precision=0, label="col_end", visible=False)
            run_tool = gr.Button("执行 Tool")
        with gr.Accordion("高级 / 调试信息", open=False):
            details = gr.JSON(label="系统 / 模型信息")
            inference_details = gr.JSON(label="Agent / 推理指标")
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
            [input_image, raster_upload, prompt, max_tokens],
            [response, inference_details, status, details, tool_execution,
             raster_information, result_image, mask_gallery],
            api_name="analyze",
        )
        tool_choice.change(
            manual_updates,
            tool_choice,
            [manual_help, manual_prompt, manual_tokens, manual_classes, confidence,
             iou_threshold, x1, y1, x2, y2, manual_boxes, manual_detection_ids,
             manual_bands, manual_stretch, manual_lower_percentile,
             manual_upper_percentile, manual_max_size, manual_resampling,
             manual_region, manual_row_start, manual_row_end, manual_col_start, manual_col_end],
            api_name="manual_tool_schema",
        )
        run_tool.click(
            lambda tool, image, raster_path, tool_prompt, tokens, classes, conf, iou,
            left, top, right, bottom, boxes, detection_ids, bands, stretch, lower,
            upper, preview_size, resampling, region, row_start, row_end, col_start, col_end:
            execute_selected_tool(
                settings, tool, image, raster_path, tool_prompt, tokens, classes,
                conf, iou, left, top, right, bottom, boxes, detection_ids, bands,
                stretch, lower, upper, preview_size, resampling,
                region, row_start, row_end, col_start, col_end,
            ),
            [tool_choice, input_image, raster_upload, manual_prompt, manual_tokens,
             manual_classes, confidence, iou_threshold, x1, y1, x2, y2,
             manual_boxes, manual_detection_ids, manual_bands, manual_stretch,
             manual_lower_percentile, manual_upper_percentile, manual_max_size,
             manual_resampling, manual_region, manual_row_start, manual_row_end,
             manual_col_start, manual_col_end],
            [response, inference_details, result_image, tool_execution, mask_gallery,
             raster_information],
            api_name="execute_tool",
        )
        demo.load(lambda: fetch_status(settings), outputs=[status, details])
        demo.load(
            lambda: manual_updates("inspect_image"),
            outputs=[manual_help, manual_prompt, manual_tokens, manual_classes, confidence,
                     iou_threshold, x1, y1, x2, y2, manual_boxes, manual_detection_ids,
                     manual_bands, manual_stretch, manual_lower_percentile,
                     manual_upper_percentile, manual_max_size, manual_resampling,
                     manual_region, manual_row_start, manual_row_end,
                     manual_col_start, manual_col_end],
        )
    return demo


if __name__ == "__main__":
    config = get_settings()
    build_ui(config).launch(
        server_name=config.gradio_host,
        server_port=config.gradio_port,
        share=False,
        theme="soft",
        allowed_paths=[str(config.output_dir / "tools")],
    )

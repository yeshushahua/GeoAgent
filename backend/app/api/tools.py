from __future__ import annotations

import json
from pathlib import Path
from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from backend.app.api.uploads import store_temporary_upload

router = APIRouter()
ERROR_STATUS = {
    "TOOL_NOT_FOUND": 404,
    "INVALID_TOOL_INPUT": 422,
    "INVALID_IMAGE": 415,
    "INVALID_CROP": 422,
    "CUDA_UNAVAILABLE": 503,
    "MODEL_FILES_MISSING": 503,
    "MODEL_BUSY": 409,
    "MODEL_LOAD_FAILED": 500,
    "INFERENCE_FAILED": 500,
    "CUDA_OUT_OF_MEMORY": 507,
    "TOOL_TIMEOUT": 504,
    "TOOL_EXECUTION_FAILED": 500,
    "ARTIFACT_NOT_FOUND": 422,
    "INVALID_ARTIFACT_ROLE": 422,
    "DETECTION_NOT_FOUND": 422,
    "DEPENDENCY_MISMATCH": 422,
    "INVALID_BBOX": 422,
    "NO_VALID_BOXES": 422,
    "INVALID_RASTER": 415,
    "RASTER_OPEN_FAILED": 415,
    "BAND_NOT_FOUND": 422,
    "INVALID_WINDOW": 422,
    "EMPTY_WINDOW": 422,
    "UNSUPPORTED_RASTER_DTYPE": 422,
    "PREVIEW_FAILED": 422,
    "RASTER_METADATA_REQUIRED": 422,
    "DETECTOR_CUDA_UNAVAILABLE": 503,
    "DETECTOR_FILES_MISSING": 503,
    "DETECTOR_BUSY": 409,
    "DETECTOR_LOAD_FAILED": 500,
    "DETECTOR_INFERENCE_FAILED": 500,
    "UNSUPPORTED_DETECTION_CLASS": 422,
    "OPEN_VOCAB_CUDA_UNAVAILABLE": 503,
    "OPEN_VOCAB_FILES_MISSING": 503,
    "OPEN_VOCAB_BUSY": 409,
    "OPEN_VOCAB_LOAD_FAILED": 500,
    "OPEN_VOCAB_INFERENCE_FAILED": 500,
    "INVALID_OPEN_VOCAB_CLASSES": 422,
    "SEGMENTATION_CUDA_UNAVAILABLE": 503,
    "SEGMENTATION_FILES_MISSING": 503,
    "SEGMENTATION_BUSY": 409,
    "SEGMENTATION_LOAD_FAILED": 500,
    "SEGMENTATION_INFERENCE_FAILED": 500,
    "INVALID_SEGMENTATION_PROMPT": 422,
}


@router.get("")
def list_tools(request: Request):
    return request.app.state.tool_registry.list_tools()


@router.get("/executions")
def list_executions(request: Request, limit: int = Query(default=20, ge=1, le=50)):
    return request.app.state.tool_traces.list(limit)


@router.get("/{tool_name}")
def get_tool(request: Request, tool_name: str):
    try:
        return request.app.state.tool_registry.get(tool_name).definition()
    except KeyError:
        return JSONResponse(
            status_code=404,
            content={"error": {"code": "TOOL_NOT_FOUND", "message": f"Unknown tool: {tool_name}"}},
        )


@router.post("/{tool_name}/execute")
async def execute_tool(
    request: Request,
    tool_name: str,
    image: UploadFile | None = File(default=None),
    image_path: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    max_new_tokens: str | None = Form(default=None),
    x1: str | None = Form(default=None),
    y1: str | None = Form(default=None),
    x2: str | None = Form(default=None),
    y2: str | None = Form(default=None),
    confidence: str | None = Form(default=None),
    iou_threshold: str | None = Form(default=None),
    classes: str | None = Form(default=None),
    boxes: str | None = Form(default=None),
    detection_ids: str | None = Form(default=None),
    raster_path: str | None = Form(default=None),
    bands: str | None = Form(default=None),
    stretch: str | None = Form(default=None),
    lower_percentile: str | None = Form(default=None),
    upper_percentile: str | None = Form(default=None),
    max_size: str | None = Form(default=None),
    resampling: str | None = Form(default=None),
    region: str | None = Form(default=None),
    row_start: str | None = Form(default=None),
    row_end: str | None = Form(default=None),
    col_start: str | None = Form(default=None),
    col_end: str | None = Form(default=None),
):
    temporary: Path | None = None
    try:
        if image is not None:
            temporary = await store_temporary_upload(request.app.state.settings, image)
            image_path = str(temporary)
            raster_path = str(temporary)
        parsed_classes = None
        if classes is not None:
            try:
                parsed_classes = json.loads(classes)
            except json.JSONDecodeError:
                parsed_classes = [
                    int(item) if item.strip().isdigit() else item.strip()
                    for item in classes.split(",") if item.strip()
                ]
        parsed_boxes = None
        if boxes is not None:
            try:
                parsed_boxes = json.loads(boxes)
            except json.JSONDecodeError:
                parsed_boxes = None
        parsed_detection_ids = None
        if detection_ids is not None:
            try:
                parsed_detection_ids = json.loads(detection_ids)
            except json.JSONDecodeError:
                parsed_detection_ids = [
                    item.strip() for item in detection_ids.split(",") if item.strip()
                ]
        parsed_bands = None
        if bands is not None:
            try:
                parsed_bands = json.loads(bands)
            except json.JSONDecodeError:
                parsed_bands = [int(item.strip()) for item in bands.split(",") if item.strip()]
        raw = {
            "raster_path": raster_path,
            "bands": parsed_bands,
            "stretch": stretch,
            "lower_percentile": lower_percentile,
            "upper_percentile": upper_percentile,
            "max_size": max_size,
            "resampling": resampling,
            "region": region,
            "row_start": row_start,
            "row_end": row_end,
            "col_start": col_start,
            "col_end": col_end,
            "image_path": image_path,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "confidence": confidence,
            "iou_threshold": iou_threshold,
            "classes": parsed_classes,
            "boxes": parsed_boxes,
            "detection_ids": parsed_detection_ids,
        }
        try:
            tool = request.app.state.tool_registry.get(tool_name)
            arguments = {
                key: value
                for key, value in raw.items()
                if key in tool.input_schema.model_fields and value is not None
            }
        except KeyError:
            arguments = {}
        result = await request.app.state.tool_executor.execute(tool_name, arguments)
        status_code = 200 if result.success else ERROR_STATUS.get(result.error.code, 500)
        return JSONResponse(
            status_code=status_code,
            content=result.model_dump(mode="json", by_alias=True),
        )
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

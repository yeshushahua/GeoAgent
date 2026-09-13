# GeoAgent

Multimodal AI Agent for Visual & Geospatial Analysis
多模态视觉与空间智能分析 Agent

Current milestone: Phase 4 — Object Detection.

GeoAgent now runs Qwen/Qwen3-VL-4B-Instruct locally on one NVIDIA RTX 4090.
The user uploads one ordinary RGB image and gives a natural-language task. A local
Qwen3-VL policy model selects tools from registry-generated schemas, observes each
ToolResult, optionally continues with another tool, and returns a structured final
response. YOLO11s supplies closed-set COCO object classes, counts, confidence scores,
pixel bounding boxes, and annotated images. The default Chinese UI calls the Agent API.

This is a small Tool-enabled Vision Agent with autonomous tool selection. It uses
an explicit bounded loop rather than LangGraph or another agent framework. Open-
vocabulary detection, SAM/segmentation, tracking, GeoTIFF/GIS processing, RAG,
memory and fine-tuning are not implemented.

## Architecture

    Gradio -> Agent API -> VisionAgent -> Qwen decision (JSON)
                                      -> ToolRegistry -> ToolExecutor -> Tool
                                             ^                         |
                                             `---- Observation --------'
                                      -> Final answer

    VisionAgent planner ----\
                             > shared ModelManager -> one Qwen3-VL instance
    analyze_image Tool -----/

    detect_objects Tool -> DetectorManager -> one lazy YOLO11s COCO instance
                                           -> detections + annotated.jpg

Model code never returns FastAPI responses. InferenceResult belongs to the model
layer and `analyze_image` converts it to the shared ToolResult contract.
The application starts in UNLOADED state. Importing FastAPI does not load weights.

Tool definitions are generated dynamically from `ToolRegistry` and each tool's
Pydantic schema. The Agent output protocol accepts only a validated `tool_call` or
`final` JSON object. Calls always pass through ToolExecutor. The loop has a six-step
default limit, one bounded JSON repair, duplicate-call blocking, structured tool
error observations, and no stored or displayed chain-of-thought.

## Verified environment

- Windows, Python 3.12.14
- NVIDIA GeForce RTX 4090, 23.99 GiB
- PyTorch 2.10.0+cu128 and torchvision 0.25.0+cu128
- Transformers 5.17.0, Accelerate 1.15.0
- qwen-vl-utils 0.0.14
- Ultralytics 8.4.148 with YOLO11s COCO weights
- SDPA attention, BF16, cuda:0

Transformers 5.17.0 satisfies the model's official requirement of 4.57.0 or
newer and is compatible with the Hugging Face Hub version used by Gradio 6.
The implementation follows the official Qwen AutoModelForImageTextToText,
AutoProcessor and qwen-vl-utils flow:

- https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- https://github.com/QwenLM/Qwen3-VL

FlashAttention, quantization, bitsandbytes, vLLM and CPU fallback are not used.

## Storage policy

Source code and the virtual environment live in:

    D:\sht\DEMO\GeoAgent

Large assets live in:

    E:\sht\DEMO\GeoAgent

The local model is:

    E:\sht\DEMO\GeoAgent\models\Qwen3-VL-4B-Instruct

The detector checkpoint is:

    E:\sht\DEMO\GeoAgent\models\object_detection\yolo11s.pt

The Hugging Face cache is:

    E:\sht\DEMO\GeoAgent\cache\huggingface

Ultralytics settings/cache are rooted at:

    E:\sht\DEMO\GeoAgent\cache\ultralytics

Models, datasets, outputs, checkpoints, caches and temporary uploads never fall
back to C or D. Startup validates absolute paths, separate drives, containment
and actual write access. Model loading uses local_files_only=True. The checked-in
example configuration defaults HF_HUB_OFFLINE=true and disables telemetry.

## Installation

Run from PowerShell:

    Set-Location D:\sht\DEMO\GeoAgent
    py -3.12 -m venv .venv
    Copy-Item .env.example .env
    $assetRoot = 'E:\sht\DEMO\GeoAgent'
    New-Item -ItemType Directory -Force -Path "$assetRoot\temp", "$assetRoot\cache\pip"
    $env:TEMP = "$assetRoot\temp"
    $env:TMP = $env:TEMP
    $env:PIP_CACHE_DIR = "$assetRoot\cache\pip"
    .\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe -m pip check

Do not overwrite an existing .env. On another computer, edit .env instead of
changing Python source. The CUDA wheel bundles its runtime, so this project does
not reinstall the system CUDA Toolkit.

## Download the fixed model

The script temporarily permits Hub access only in its own process, checks the
exact model ID and E drive destination, supports Hugging Face resume behavior,
then validates config, generation, processor, tokenizer and safetensors files.

    .\.venv\Scripts\python.exe -m scripts.download_qwen3_vl

The normal .env remains offline. No HF token is required for this public model.
Model weights must never be added to Git.

Download the fixed YOLO11s checkpoint separately:

    .\.venv\Scripts\python.exe -m scripts.download_yolo11s

The script writes only below configured `MODEL_DIR`. YOLO11s is a mature
Ultralytics production model with COCO pretrained closed-set detection weights.

## Run FastAPI and Gradio

Use two PowerShell windows:

    .\.venv\Scripts\python.exe -m backend.app.main

    .\.venv\Scripts\python.exe -m frontend.app

- Gradio: http://127.0.0.1:7860
- Health: http://127.0.0.1:8000/api/v1/health
- System: http://127.0.0.1:8000/api/v1/system
- OpenAPI: http://127.0.0.1:8000/docs

Model endpoints:

- GET /api/v1/models/vlm/status
- POST /api/v1/models/vlm/load
- POST /api/v1/models/vlm/unload
- POST /api/v1/models/vlm/infer

Detector endpoints:

- GET /api/v1/models/detector/status
- POST /api/v1/models/detector/load
- POST /api/v1/models/detector/unload

Tool endpoints:

- GET /api/v1/tools
- GET /api/v1/tools/{tool_name}
- POST /api/v1/tools/{tool_name}/execute
- GET /api/v1/tools/executions?limit=20

Agent endpoints:

- POST /api/v1/agent/run
- GET /api/v1/agent/executions?limit=20

The registry contains `inspect_image`, `crop_image`, `analyze_image`, and
`detect_objects`. Each exposes a Pydantic JSON Schema. ToolExecutor validates
inputs, generates an execution ID, records duration, isolates failures, returns a
ToolResult, and keeps the latest 50 safe in-memory trace records. Crop artifacts
are written below `E:\sht\DEMO\GeoAgent\outputs\tools`; image bytes are never put
inside ToolResult JSON. See `docs/tool-system.md` for the complete boundary.

`detect_objects` accepts an image path plus optional confidence, IoU, and COCO
class filters. Its default confidence is 0.25 and default IoU threshold is 0.45.
It returns image dimensions, total detections, class counts, class ID/name,
confidence, and original-image `xyxy` pixel coordinates. Every successful call,
including an empty result, writes `annotated.jpg` under the Tool output directory.
DetectorManager lazy-loads once, reuses the same instance, runs only on `cuda:0`,
and can release its weights independently from Qwen.

The infer endpoint accepts multipart image, prompt and max_new_tokens. Supported
formats are PNG, JPEG/JPG and WEBP. max_new_tokens is limited to 64–512.
Errors have an error.code and safe error.message; server tracebacks stay in logs.

## Image size policy

The original width, height, mode and format are retained in result metadata.
Images are copied and converted to RGB; the original file is not changed.

Input has two guards:

1. A 2048-pixel longest-edge pre-resize preserves aspect ratio.
2. qwen-vl-utils applies a 1280 visual-token budget, equal to 1,310,720 pixels
   for Qwen3-VL's 32 x 32 spatial compression.

A 4096 x 3072 test image was safely processed at 1312 x 960 with peak model
process allocation of 9.062 GiB and no OOM. This retains useful detail while
leaving substantial headroom on the 24 GB RTX 4090.

## Tests

Default tests use mocks and do not load the 4B model:

    .\.venv\Scripts\python.exe -m pytest

The marked integration test performs offline local loading and real RTX 4090
inference:

    .\.venv\Scripts\python.exe -m pytest -m integration -s

With FastAPI running:

    .\.venv\Scripts\python.exe -m scripts.smoke_test
    .\.venv\Scripts\python.exe -m scripts.verify_phase1_api

With FastAPI and Gradio running:

    .\.venv\Scripts\python.exe -m scripts.verify_ui
    .\.venv\Scripts\python.exe -m scripts.verify_phase4_api
    .\.venv\Scripts\python.exe -m scripts.verify_phase4_ui

The integration suite includes real Qwen Agent decisions for a metadata-only task,
a semantic-analysis task, and a multi-step inspect/crop/analyze task. It verifies
that the crop artifact becomes the next analyze_image input, all five repeated
content runs use one necessary tool, and GPU allocation remains stable. It also
retains the Phase 2 Tool API chain, model auto-load,
manual `inspect_image -> crop_image -> analyze_image` execution, five consecutive
Qwen tool calls, GPU allocation regression, trace checks, and executor overhead.

See `docs/phase3-validation.md` for the measured Agent decisions, Scenario D
artifact chain, latency, five-run VRAM results, and live API/UI acceptance.

Phase 4 integration additionally covers two COCO photos, five repeated detections,
class filtering, detector reuse, detection-only Agent routing, detect-then-analyze,
crop-to-detect artifact propagation, annotated previews, and simultaneous Qwen +
YOLO residency. See `docs/phase4-validation.md` for measured results.

## Benchmark

Generate the five original, lightweight RGB scenes and run the full benchmark:

    .\.venv\Scripts\python.exe -m scripts.generate_phase1_samples
    .\.venv\Scripts\python.exe -m scripts.benchmark_vlm

The benchmark performs one cold load, 10 consecutive varied inferences including
English, Chinese, spatial relations, approximate counting and aerial-style RGB,
a high-resolution safety case, unload, reload, another inference and final unload.

Output is written outside Git:

    E:\sht\DEMO\GeoAgent\outputs\benchmarks\phase1\phase1_vlm_benchmark.json

See docs/phase1-validation.md for the measured results and selected real outputs.

## Git safety

.env, .venv, root models/cache/datasets/outputs/checkpoints/temp directories,
weight formats, large TIFF files and logs are ignored. The small sample images are
original project assets. Do not commit user images, credentials, model files,
Hub cache, benchmark output or datasets.

No commit or push is performed by the Phase 4 workflow.

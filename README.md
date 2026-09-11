# GeoAgent

Multimodal AI Agent for Visual & Geospatial Analysis
多模态视觉与空间智能分析 Agent

Current milestone: Phase 1 — Qwen3-VL-4B Multimodal Baseline.

GeoAgent now runs Qwen/Qwen3-VL-4B-Instruct locally on one NVIDIA RTX 4090.
The user uploads one ordinary RGB image, sends a Chinese or English prompt through
FastAPI, and receives a structured InferenceResult with the answer, latency,
preprocessing size and GPU memory. Gradio calls those same backend endpoints.

This is the multimodal foundation, not an Agent. Tool calling, LangGraph, YOLO,
SAM, GeoTIFF/GIS processing, RAG and fine-tuning are not implemented.

## Architecture

    Gradio
       |
       | multipart HTTP
       v
    FastAPI /api/v1/models/vlm
       |
       v
    ModelManager
       |
       v
    QwenVlModel -> local Qwen3-VL-4B-Instruct -> cuda:0 BF16

Model code never returns FastAPI responses. InferenceResult belongs to the model
layer; the Phase 0 ToolResult remains reserved for later Agent tools.
The application starts in UNLOADED state. Importing FastAPI does not load weights.

## Verified environment

- Windows, Python 3.12.14
- NVIDIA GeForce RTX 4090, 23.99 GiB
- PyTorch 2.10.0+cu128 and torchvision 0.25.0+cu128
- Transformers 5.17.0, Accelerate 1.15.0
- qwen-vl-utils 0.0.14
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

The Hugging Face cache is:

    E:\sht\DEMO\GeoAgent\cache\huggingface

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

No commit or push is performed by the Phase 1 workflow.

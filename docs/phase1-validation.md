# GeoAgent Phase 1 Validation

Validation date: 2026-09-11.

## 1. Phase 1 conclusion

PARTIAL.

All model, GPU, API, Gradio event, benchmark, offline, unit and integration
checks passed. Browser visual automation could not start because the Codex
Windows sandbox helper failed twice with helper_unknown_error during setup
refresh. The Gradio page and all component events were verified over its real
HTTP interface, but the rendered controls were not visually inspected.

## 2. Model

- Model ID: Qwen/Qwen3-VL-4B-Instruct
- Local path: E:\sht\DEMO\GeoAgent\models\Qwen3-VL-4B-Instruct
- Size: 8,887,296,437 bytes, 8.277 GiB
- HF_HOME: E:\sht\DEMO\GeoAgent\cache\huggingface
- Weight shards: two safetensors files
- Required config, generation, processor, tokenizer and index files verified
- Download was unauthenticated from the public official repository
- Production load used local_files_only=True with HF_HUB_OFFLINE=true

## 3. Software environment

- Python 3.12.14
- PyTorch 2.10.0+cu128
- CUDA runtime 12.8
- torchvision 0.25.0+cu128
- Transformers 5.17.0
- Accelerate 1.15.0
- qwen-vl-utils 0.0.14
- Gradio 6.26.0
- pip check: No broken requirements found

Transformers 5.17.0 satisfies the official requirement of 4.57.0 or newer and
works with the Hugging Face Hub dependency required by Gradio 6. Official
references:

- https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- https://github.com/QwenLM/Qwen3-VL
- https://huggingface.co/docs/transformers/en/model_doc/qwen3_vl

## 4. GPU

- GPU: NVIDIA GeForce RTX 4090
- VRAM: 23.99 GiB, 25,756,696,576 bytes
- dtype: bfloat16
- device: cuda:0
- attention: SDPA
- CPU fallback: disabled and runtime placement verified
- No RTX 5080, multi-GPU, quantization, FlashAttention or vLLM configuration

## 5. Model loading

- Cold load time: 6.616 s
- VRAM before load: allocated 0.000 GiB, reserved 0.000 GiB
- VRAM after load: allocated 8.266 GiB, reserved 8.270 GiB
- Peak across the ten-run benchmark: 9.062 GiB
- After first unload: allocated 0.009 GiB, reserved 0.021 GiB

The small remainder after unload is the CUDA runtime context. Model references
and processor references were removed, Python garbage collection ran, and the
CUDA cache was emptied.

## 6. Real inference examples

English description:

    This image presents a minimalist, clean composition featuring two basic
    geometric shapes on a plain, light-colored background...

Chinese description:

    好的，这张图片是一幅风格简约、色彩扁平化的合成街景插画。
    图片采用极简主义和几何图形风格...

Object understanding:

    Six colored balls (circles) arranged around a central white box...

Spatial relationship:

    In the upper-left area of the image, there is a red square...
    The red square is in the upper-left, the blue circle in the upper-right,
    and the green triangle in the lower center.

Chinese scene analysis:

    最主要的场景类型是城市街道场景。判断依据包括建筑物、道路和车辆。

Approximate counting:

    There are 6 colored balls present.

This is recorded as approximate VLM counting, not exact CV counting.

Aerial-style RGB understanding:

    图像背景以大面积绿色为主，代表植被覆盖；灰色道路横向穿过，
    蓝色区域代表水体，浅色矩形代表建筑。

The aerial image is an original synthetic RGB illustration. No GeoTIFF,
geographic coordinates or area calculation was used.

## 7. Performance

- Continuous real inferences: 10
- First inference latency: 4,635.76 ms
- Warm average latency, runs 2–10: 3,896.03 ms
- Fastest: 542.63 ms
- Slowest: 4,749.04 ms
- Peak allocated VRAM: 9.062 GiB
- Allocated after first inference: 8.275 GiB
- Allocated after tenth inference: 8.275 GiB
- Measured growth: 0.000 GiB
- FastAPI remained responsive
- ModelManager remained READY after run 10
- No CUDA OOM occurred

Outputs were limited to 96 new tokens for the benchmark. Latency therefore
varies with answer length.

## 8. Load and unload

- First load: READY in 6.616 s
- First inference: PASS
- First unload: UNLOADED, allocated VRAM fell to 0.009 GiB
- Second load: READY in 4.672 s
- Second inference: PASS in 6,191.10 ms
- Second unload: UNLOADED, allocated VRAM returned to 0.009 GiB

Second inference output:

    This image displays a simple, minimalist composition featuring two basic
    geometric shapes on a light background: a dark teal square on the left
    and a solid orange circle on the right...

## 9. FastAPI

All calls used real HTTP on http://127.0.0.1:8000.

- GET status: 200, initial state UNLOADED
- POST load: 200, state READY, load time 7.356 s
- POST infer: 200, real answer, 3,324.50 ms, cuda:0, bfloat16
- GET status after inference: 200, state READY
- POST unload: 200, state UNLOADED, allocated 0.009 GiB

Actual API inference answer:

    Based on the provided image and the text label at the top, the upper-left
    area of the image contains a red square...

Validation and operational errors return safe error.code and error.message
fields. Missing image, empty prompt, unsupported/corrupt format, unloaded
model, missing files, busy state and CUDA OOM behavior have unit coverage.
Tracebacks remain in server logs.

## 10. Gradio

Real Gradio HTTP event validation passed:

- Page returned HTTP 200
- Upload and preview returned pixel-identical PNG content
- Load Model triggered the FastAPI load endpoint
- Analyze triggered FastAPI multipart inference and real Qwen3-VL generation
- Response, latency, device, dtype, image resolution and GPU memory returned
- Unload Model triggered the FastAPI unload endpoint
- Final model state was UNLOADED
- The UI explicitly says No tools used in Phase 1

Actual Gradio inference answer:

    This is a minimalist digital illustration featuring two simple geometric
    shapes on a light background: a dark teal square on the left and a solid
    orange circle on the right...

Gradio inference latency was 2,648.34 ms with peak allocated VRAM 8.327 GiB.
Browser rendered-layout inspection is the only incomplete validation.

## 11. Tests

Default suite:

    collected 43 items
    42 passed, 1 skipped, 2 warnings in 0.52s

The skipped test is the explicitly separated 4B integration test.

Integration suite:

    1 passed, 42 deselected, 2 warnings in 10.41s

The integration test loaded the complete local model offline on RTX 4090,
performed real image inference, verified cuda:0 and bfloat16, then unloaded.

The two warnings are dependency deprecations in Starlette TestClient and anyio.

## 12. Benchmark

Output directory:

    E:\sht\DEMO\GeoAgent\outputs\benchmarks\phase1

Files:

- phase1_vlm_benchmark.json, 13,410 bytes
- api_acceptance.json, 2,832 bytes
- gradio_acceptance.json, 1,383 bytes

Benchmark output stays on E and is ignored by Git.

## 13. Storage

- Model exists only under the configured E drive model directory
- Hugging Face environment and cache paths resolve to E
- C:\Users\admin\.cache\huggingface was 20,507 bytes before and after Phase 1
- D:\sht\DEMO\GeoAgent has no root models or cache directory
- No model/config cache was copied into the repository
- No non-venv workspace file exceeds 10 MiB
- The E model, dataset and output paths never fall back to C or D

Five new original synthetic RGB images were created in sample_data/images.
The 4096 x 3072 image is highly compressible and remains a small repository
asset. No external dataset or user image was added.

## 14. Git

The working tree contains the expected Phase 1 source, test, documentation and
small original sample-image changes. No staging, commit or push was performed.

No model weights, Hub cache, dataset, E-drive output, .env, checkpoint,
benchmark JSON or sensitive file is tracked. Git ignore checks passed for:

- .env and .venv
- root models, cache, datasets, outputs and checkpoints
- safetensors and TIFF patterns

## 15. Phase 1 PASS checklist

- [x] Qwen3-VL-4B-Instruct downloaded completely to E
- [x] Phase 1 HF cache did not write to C or D
- [x] Transformers is at least 4.57.0
- [x] qwen-vl-utils 0.0.14 installed
- [x] Model loaded in BF16
- [x] Model ran on RTX 4090 cuda:0
- [x] No CPU fallback
- [x] ModelManager load
- [x] ModelManager unload
- [x] Load, unload and reload
- [x] FastAPI status endpoint
- [x] FastAPI load endpoint
- [x] FastAPI unload endpoint
- [x] FastAPI infer endpoint
- [x] Gradio called FastAPI for real VLM inference
- [x] English image description
- [x] Chinese image description
- [x] Object understanding
- [x] Spatial relationship question
- [x] Aerial-style RGB understanding
- [x] InferenceResult structure
- [x] Real GPU memory recorded
- [x] Real load time recorded
- [x] Real inference latency recorded
- [x] Ten continuous real inferences
- [x] No measurable allocated-memory growth
- [x] High-resolution input strategy
- [x] Offline local-only inference
- [x] Phase 0 tests did not regress
- [x] New Phase 1 unit tests
- [x] Real model integration test
- [x] Benchmark output on E
- [x] pip check
- [x] README updated
- [x] Model weights excluded from Git
- [x] E-drive data excluded from Git
- [x] Git status has no abnormal large file
- [ ] Browser visual inspection

## 16. Incomplete items and risks

Only browser visual inspection remains. The Computer Use runtime failed twice
before opening the local page, so no screenshot or claimed visual result is
included. The user can open http://127.0.0.1:7860 while the services remain
running and confirm the layout, buttons and rendered metrics.

The five benchmark images are synthetic and useful for repeatable functional
testing. They do not establish performance on natural photography or scientific
remote-sensing imagery. Adding a curated natural-image evaluation set belongs
in a later benchmark expansion, not in this Phase 1 baseline.

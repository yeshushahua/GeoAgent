# Phase 5 Validation — Open-Vocabulary Detection + Segmentation

## Scope

Phase 5 adds `detect_open_vocab` backed by YOLOE-26s-seg and `segment_objects`
backed by SAM 2.1 Base. It retains every Phase 0–4 route and tool. No tracking,
geospatial processing, RAG, memory, fine-tuning, deployment, commit, or push is part
of this milestone.

## Fixed local assets

All large files remain under `E:\sht\DEMO\GeoAgent`:

| Asset | Local path | Bytes |
|---|---|---:|
| YOLOE-26s-seg | `models\open_vocabulary\yoloe-26s-seg.pt` | 31,072,171 |
| MobileCLIP2-B | `models\open_vocabulary\mobileclip2_b.ts` | 253,794,476 |
| SAM 2.1 Base | `models\segmentation\sam2.1_b.pt` | 161,935,802 |
| Construction-PPE | `datasets\phase5\construction-ppe` | local extracted dataset |

Normal runtime sets Hugging Face and YOLO offline flags. Ultralytics settings point
weights, datasets, and runs to E:\ before Ultralytics is imported. The download
script is the explicit online preparation boundary.

## Tool and Agent contracts

`detect_open_vocab` accepts 1–32 English text labels, confidence, and IoU. It emits
requested classes, a total, per-class counts, stable detection IDs, confidence,
source-image `xyxy` pixel boxes, and a JPEG preview. The local MobileCLIP2 encoder is
reused while class prompts remain unchanged.

`segment_objects` accepts 1–50 explicit boxes in the same coordinate space. It emits
one PNG mask per instance, a combined PNG overlay, mask area in pixels, and a ratio
against total source-image pixels. It does not report physical or geographic area.

The planner discovers both tools from the registry. For segmentation requests it
copies detector boxes exactly and keeps the detector's source image. Annotated images
are excluded from downstream planner inputs. Empty detections finish truthfully and
do not invoke SAM.

## Real RTX 4090 model validation

The real model suite uses Construction-PPE `image40.jpg` (640 × 640) and runs with
network creation blocked. It loads Qwen3-VL, YOLO11s, YOLOE, and SAM on `cuda:0` at
the same time.

| Check | Result |
|---|---:|
| Qwen load | 4.331 s |
| YOLO11s load | 0.049 s |
| YOLOE load | 0.092 s |
| SAM load | 0.475 s |
| Five YOLOE-to-SAM allocated memory | 8.814 GiB on every run |
| Five YOLOE-to-SAM peak allocations | 9.070–9.153 GiB |
| Five YOLOE inference times | 11.05–54.44 ms |
| Five SAM inference times | 40.75–228.19 ms |
| Repeated full-image mask ratio | 0.00522461 |
| Crop detection count | 1 |
| Crop mask ratio | 0.02046875 |
| Nonexistent-target count | 0 |
| Open-vocabulary manager load count | 1 |
| Segmentation manager load count | 1 |
| Four Agent workflow totals | 12.128–29.760 s |
| Four Agent peak allocations | 12.338–14.084 GiB |

Measured evidence is written to
`E:\sht\DEMO\GeoAgent\outputs\benchmarks\phase5\model_validation.json`.

## Required workflows

The real Qwen planner is validated with four Chinese tasks:

1. Detect a yellow safety helmet: `detect_open_vocab -> final`.
2. Detect and precisely segment the helmet: `detect_open_vocab -> segment_objects -> final`.
3. Crop the top-left quadrant, detect in the crop, then segment:
   `inspect_image -> crop_image -> detect_open_vocab -> segment_objects -> final`.
4. Search for a nonexistent bright-purple submarine and conditionally segment:
   `detect_open_vocab -> final`, with zero detections and no SAM call.

The direct five-run workflow additionally checks model reuse, stable memory, mask
files, overlays, exact bbox propagation, crop coordinates, and all-four-model
residency. Unit tests use fakes and cover lifecycle errors, schemas, APIs, result
shape, artifacts, no Base64 payloads, and Agent state propagation.

## Reproduction

From `D:\sht\DEMO\GeoAgent`:

    .\.venv\Scripts\python.exe -m scripts.download_phase5_models
    .\.venv\Scripts\python.exe -m pytest
    .\.venv\Scripts\python.exe -m pytest -m integration -s

With FastAPI and Gradio running:

    .\.venv\Scripts\python.exe -m scripts.verify_phase5_api
    .\.venv\Scripts\python.exe -m scripts.verify_phase5_ui

Live reports are written beside the model report as `live_api_acceptance.json` and
`gradio_acceptance.json`. Third-party Starlette/httpx and TorchScript deprecation
warnings do not affect the accepted behavior.

References: [Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe),
[Ultralytics SAM 2](https://docs.ultralytics.com/models/sam-2), and
[Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe).

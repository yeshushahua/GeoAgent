# Phase 4 Validation — Object Detection

Date: 2026-09-13  
Verdict: **PASS**

## 1. Scope

Phase 4 adds one real closed-set detection Tool, `detect_objects`, backed by
Ultralytics YOLO11s COCO weights. It returns structured detections and an annotated
image Artifact. It does not add open-vocabulary detection, segmentation, tracking,
GIS, GeoTIFF, SAM, YOLOE, Grounding DINO, or another agent framework.

## 2. Architecture

    Natural-language task
      -> VisionAgent
      -> Qwen JSON decision
      -> dynamic ToolRegistry schema
      -> ToolExecutor
      -> detect_objects
      -> DetectorManager
      -> one lazy YOLO11s instance on cuda:0
      -> ToolResult observation + annotated.jpg
      -> next Qwen decision or final

`DetectorManager` is independent from the existing Qwen `ModelManager`. It owns
load, unload, status, class-filter resolution, prediction parsing, timing, and GPU
metrics. The Tool owns safe image validation and Artifact annotation/output.

## 3. Model and Storage

- Library: Ultralytics 8.4.148
- Model: YOLO11s Detect, COCO pretrained closed-set weights
- Device: `cuda:0`
- Image size: 640
- Weight: `E:\sht\DEMO\GeoAgent\models\object_detection\yolo11s.pt`
- Ultralytics config: `E:\sht\DEMO\GeoAgent\cache\ultralytics`
- Test images: `E:\sht\DEMO\GeoAgent\datasets\phase4\images`
- Artifacts: `E:\sht\DEMO\GeoAgent\outputs\tools`

The 18.4 MiB checkpoint is outside Git. A dedicated download script validates the
configured E-drive destination before using the official Ultralytics asset helper.

## 4. Tool Contract

Input:

- `image_path`: required PNG, JPEG, or WEBP reference
- `confidence`: 0.01–1, default 0.25
- `iou_threshold`: 0.01–1, default 0.45
- `classes`: optional COCO IDs or exact English COCO names

Output data:

- `image_width`, `image_height`, `detection_count`
- `class_counts`
- each detection's `class_id`, `class_name`, confidence in 0–1, and bounded
  original-image `x1/y1/x2/y2` pixel box

Every successful call, including zero detections, writes `annotated.jpg` as an
Artifact. ToolResult and traces contain paths and structured JSON only, never
Base64 image content.

## 5. Agent Integration

The Agent discovers `detect_objects` from `ToolRegistry.list_tools()` with no
hardcoded registry list or keyword router. The system policy assigns structured
detection, class counts, boxes, and confidence to YOLO. Detection-only requests
finish after one Tool; explicit detect-and-analyze requests continue to
`analyze_image`, whose planner turn receives the structured detection observation.
After a crop, detection must use the crop Artifact path.

## 6. Scenario A — Detection Only

Request: `检测这张图片里的目标，并告诉我每类有多少个。`

- Run ID: `4594e8de-d0db-48ad-93cc-6dd761818135`
- Tool sequence: `detect_objects -> final`
- Image: 810×1080 official Ultralytics bus photo
- Result: 5 detections — bus × 1, person × 4
- Bus confidence: 0.919985
- Annotated Artifact:
  `E:\sht\DEMO\GeoAgent\outputs\tools\20260913\0558f0ad-7d93-4ced-a721-03d2983d2a1b\annotated.jpg`

No unnecessary `inspect_image` or `analyze_image` call occurred.

## 7. Scenario B — Detect Then Analyze

Request: `先检测图中的目标，再结合检测结果分析这张图片。`

- Run ID: `3c2bbf9f-1509-4e13-871b-3d8f893f6433`
- Tool sequence: `detect_objects -> analyze_image -> final`
- Detection observation: bus × 1, person × 4 with five exact boxes/confidences
- Analyze step: original image, `detection_observation_count=1`
- Combined allocated VRAM: 8.372 GiB
- Combined peak VRAM: 11.388 GiB

The planner received the detection ToolResult before creating the analysis call;
structured counts and boxes remained authoritative.

## 8. Scenario C — Crop Then Detect

Request: `裁剪左上四分之一区域，然后检测里面的目标。`

- Run ID: `a5019ffc-c39e-4275-93d8-cf5d4aa07fb9`
- Tool sequence: `inspect_image -> crop_image -> detect_objects -> final`
- Original observation: 810×1080
- Crop coordinates: `(0, 0, 405, 540)`
- Crop Artifact:
  `E:\sht\DEMO\GeoAgent\outputs\tools\20260913\8d141a0d-1499-4cb2-868d-b7de18796cfa\crop.png`
- `detect_objects.image_path`: `crop.png`
- Detector input dimensions: 405×540
- Crop detections: bus × 1, person × 2, tie × 1
- Annotated Artifact:
  `E:\sht\DEMO\GeoAgent\outputs\tools\20260913\392971ba-eb4f-45b6-9e1a-2e33f2e4c11a\annotated.jpg`

The detector did not receive the original image in step 3.

## 9. Detector Performance and Reuse

One cold load plus five consecutive real detections:

| Run | Image | Detections | Inference ms | Allocated GiB | Peak GiB | Load count |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | bus.jpg | 5 | 8.06 | 0.080 | 0.117 | 1 |
| 2 | zidane.jpg | 3 | 40.36 | 0.080 | 0.117 | 1 |
| 3 | bus.jpg | 5 | 8.44 | 0.080 | 0.117 | 1 |
| 4 | zidane.jpg | 3 | 8.82 | 0.080 | 0.117 | 1 |
| 5 | bus.jpg | 5 | 8.20 | 0.080 | 0.117 | 1 |

Cold load was 0.185 s. The same Detector instance handled all five calls; load
count stayed 1 and allocated VRAM stayed at 0.080 GiB. A real `classes=["bus"]`
filter returned exactly one bus and no other class.

When Qwen and YOLO were resident together, total allocated VRAM was 8.372 GiB and
peak was 11.231 GiB in Scenario C, leaving substantial headroom on the RTX 4090.

## 10. Tests

- Default suite: **77 passed, 5 skipped**, 2 third-party deprecation warnings
- Full RTX 4090 integration: **5 passed, 77 deselected**, 2 warnings, 175.41 s

Default tests use fake detector outputs and never load YOLO/GPU. Coverage includes
lazy load, instance reuse, schemas/defaults, zero detections, multiple detections,
bbox/confidence validation, class filters, Artifact creation, missing images,
Detector API status, Tool API execution, Agent discovery, detect-and-analyze
observation grounding, crop Artifact propagation, and Base64-free traces.

## 11. Live Validation

FastAPI: **PASS**

- Health version 0.5.0
- Existing VLM status and inspect Tool endpoints passed
- Direct `detect_objects`: bus × 1, person × 4
- Agent run `164eef9f-e3e4-499d-9daf-72fdd12402df`
- Multi-step sequence: `inspect_image -> crop_image -> detect_objects`
- Live crop/annotated dimensions: 405×540
- Detector load count: 1
- Combined allocated/peak VRAM: 8.370/11.221 GiB

Gradio: **PASS**

- Real upload and queued Analyze component event
- Run ID: `fa98a25a-c871-420e-8952-c09e2d14fea2`
- Agent selected only `detect_objects`
- UI panel displayed total 5, bus × 1, person × 4
- Chinese final: `图片中检测到1辆公交车和4个人。`
- Displayed preview matched the annotated Artifact pixel-for-pixel
- Preview temp path remained on configured E-drive storage
- Manual `detect_objects` debug event and detector unload passed

## 12. Quality and Boundaries

Python compilation, `pip check`, `git diff --check`, storage-path scanning, and
service-log scanning passed. No production `if "检测"` routing exists. The only
new runtime framework is the pinned Ultralytics detection dependency.

YOLO11s recognizes only its pretrained COCO classes. An absent detection does not
prove an object is absent, and confidence/bboxes depend on the closed-set model.
Unknown requested labels return a structured unsupported-class error; the Agent is
instructed to disclose the boundary rather than invent results. Open-vocabulary
detection, segmentation, tracking, and all Phase 5 capabilities remain unimplemented.

## 13. Verdict

**PASS — Phase 4 ready for final review.** No commit or push was performed.

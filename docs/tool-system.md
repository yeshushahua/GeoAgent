# GeoAgent Tool System

The Phase 2 boundary now supports the Phase 5 bounded Vision Agent. Qwen3-VL chooses
from registry-generated definitions at runtime; the executor remains the only route
to tools. No dispatcher contains a task-keyword table.

## Runtime path

    VisionAgent / manual API
         |
         v
    ToolRegistry -- discovery definitions and Pydantic JSON Schema
         |
         v
    ToolExecutor -- validation, execution ID, timing, error isolation, trace
         |
         v
    BaseTool implementation
         |
         v
    ModelManager or image service
         |
         v
    ToolResult and Artifact

The UI and Agent depend on the Tool API and tool contracts. They do not
need to import `ModelManager` or the Qwen wrapper. The Phase 1 model endpoints stay
available for lifecycle control, debugging, and benchmarks.

## Contracts

Every tool publishes a stable lowercase snake_case name, an LLM-readable
description, category, version, GPU/model requirements, and a Pydantic input
schema. `ToolRegistry.list_tools()` derives discovery JSON Schema from that one
Pydantic definition. Duplicate names raise an error instead of replacing a tool.

`ToolExecutor` looks up the name, validates arguments, creates a UUID execution
ID, measures the call, catches failures, appends safe metadata, and records a
bounded in-memory trace. Trace entries retain timing, success, error type, and
prompt length only. Full prompts, images, Base64 data, and tracebacks are excluded.
Tracebacks remain in server logs.

All calls return `ToolResult`. Files are represented by `Artifact` records with a
type, absolute path, and MIME type; binary image data is never embedded in JSON.

## Registered tools

- `inspect_image` reads PNG, JPEG, or WEBP metadata without a GPU.
- `crop_image` validates pixel coordinates and writes a PNG artifact under
  `E:\sht\DEMO\GeoAgent\outputs\tools\YYYYMMDD\<execution_id>\crop.png`.
- `analyze_image` auto-loads an unloaded Qwen3-VL model, then calls it through
  `ModelManager` and converts `InferenceResult` into `ToolResult`.
- `detect_objects` auto-loads one reusable YOLO11s COCO detector through the
  independent `DetectorManager`, returns structured class/count/confidence/bbox
  data, and writes `annotated.jpg` as an Artifact.
- `detect_open_vocab` auto-loads one reusable YOLOE-26s-seg detector, encodes one
  or more local English text prompts with MobileCLIP2, returns exact source-image
  boxes, and writes a display-only `open-vocab-annotated.jpg` Artifact.
- `segment_objects` auto-loads one reusable SAM 2.1 Base model and accepts only
  explicit pixel bounding-box prompts. It returns per-instance mask area in pixels
  and image-area ratio, writes individual grayscale PNG masks, and writes one
  combined overlay.

Qwen3-VL, YOLO11s, YOLOE, and SAM each have an independent manager and lifecycle.
All are lazy-loaded, reused after the first load, fixed to `cuda:0`, and independently
unloadable. YOLOE caches the active class signature, so an unchanged prompt set is
not re-encoded. A zero-target observation stops a requested detection-to-segmentation
workflow before SAM is loaded.

Uploaded names never form output paths. Tool API uploads receive generated UUID
names in the configured E-drive temporary directory and are removed after each
request.

## HTTP surface

- `GET /api/v1/tools`
- `GET /api/v1/tools/{tool_name}`
- `POST /api/v1/tools/{tool_name}/execute`
- `GET /api/v1/tools/executions?limit=20`

Execution accepts multipart image uploads and tool-specific form fields. A trusted
image reference inside the configured project or storage root can also be supplied
as `image_path`.

Future geospatial tools can register through the same boundary without dispatch
branches in the executor.

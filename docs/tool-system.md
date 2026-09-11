# GeoAgent Tool System

Phase 2 adds a uniform tool boundary for future agent work. It remains a manually
invoked, tool-enabled multimodal system; it does not contain a planner, autonomous
tool selection, ReAct, LangGraph, or agent reasoning.

## Runtime path

    Future Agent
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

The UI and a future agent depend on the Tool API and tool contracts. They do not
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

Future phases can register `detect_objects`, `segment_objects`, and geospatial
tools without adding dispatch branches to the executor.

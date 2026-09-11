# Phase 2 Validation — Tool System Foundation

Date: 2026-09-11  
Result: **PASS**

## 1. Phase 2 conclusion

GeoAgent now has a registered, discoverable, schema-validated, traceable tool
system. The production Gradio Analyze action uses the Tool API and no longer calls
the model inference endpoint directly. This phase remains manually invoked and
contains no autonomous agent behavior.

## 2. Tool architecture

    backend/app/tools/
    |-- base.py
    |-- context.py
    |-- errors.py
    |-- executor.py
    |-- image_io.py
    |-- registry.py
    |-- trace.py
    |-- utility/image_crop.py
    `-- vision/
        |-- image_metadata.py
        `-- vlm_analyze.py

`backend/app/api/tools.py` exposes the HTTP surface. `backend/app/main.py` creates
one registry, executor, trace store, and context during application startup.

## 3. Tool registry

The live discovery endpoint returned exactly, in stable order:

    analyze_image
    crop_image
    inspect_image

Duplicate registration raises `DuplicateToolError`; it never overwrites the first
tool. Registry unit tests cover register, unregister, get, has, discovery, and
duplicate handling.

## 4. Tool schemas

- `InspectImageInput`: `image_path`
- `CropImageInput`: `image_path`, `x1`, `y1`, `x2`, `y2`
- `AnalyzeImageInput`: `image_path`, `prompt`, `max_new_tokens` (64–512)

All three are Pydantic models with extra fields forbidden. Discovery derives JSON
Schema directly from the models. Crop ordering and bounds are validated; blank
prompts and invalid numeric form values produce structured input errors.

## 5. Tool executor

The executor performs registry lookup, Pydantic validation, UUID generation,
UTC start time capture, timeout control, tool execution, exception isolation,
duration calculation, ToolResult normalization, and bounded in-memory trace
recording. The executor contains no Qwen inference code.

## 6. analyze_image

`AnalyzeImageTool` is the only tool that knows it needs the configured model. It
uses `ToolContext.model_manager`, converts `InferenceResult` into `ToolResult`, and
returns the answer plus model, device, dtype, latency, and GPU memory metadata.
The UI imports neither `ModelManager` nor `QwenVlModel`.

Real answer from the auto-load integration:

> This is a minimalist digital illustration featuring two simple geometric
> shapes on a light background: a dark teal square on the left and a solid orange
> circle on the right.

## 7. Auto load

The integration test explicitly called the model unload endpoint, confirmed
`UNLOADED`, then posted to `analyze_image`. The tool loaded Qwen3-VL locally,
reached `READY`, and returned a non-empty answer.

- execution ID: `32765f25-11fd-4cd7-ac6c-422321968048`
- total duration: 10,456.35 ms (includes model load)
- inference latency: 3,384.79 ms
- model: `Qwen3-VL-4B-Instruct`

## 8. API

The live server returned HTTP 200 for:

- `GET /api/v1/tools`
- `GET /api/v1/tools/analyze_image`
- `POST /api/v1/tools/inspect_image/execute`
- `POST /api/v1/tools/crop_image/execute`
- `POST /api/v1/tools/analyze_image/execute`
- `GET /api/v1/tools/executions?limit=20`

The live auto-load call returned execution ID
`0ea3128f-f29c-4709-a32c-a9354796b1e9`, duration 9,973.82 ms, model latency
2,591.03 ms, and final model state `READY`. Error tests cover unknown tools,
missing inputs, malformed numeric fields, invalid images, invalid crops, model
errors, OOM mapping, timeout, and unexpected exceptions without client tracebacks.

## 9. Gradio

The real Gradio queue/event client verified page load, upload, preview, forced
`UNLOADED`, Analyze, Qwen auto-load, ToolResult rendering, execution panel,
manual `inspect_image`, and final unload. The Analyze event called
`/api/v1/tools/analyze_image/execute` and displayed:

    analyze_image -> Qwen3-VL -> completed

The UI returned execution ID `e5b48956-54e4-460a-99ab-a15a60ff946f`, total
duration 8,007.94 ms, and inference latency 3,542.14 ms. It displays no agent
reasoning.

## 10. Artifact

One verified integration crop artifact is:

    E:\sht\DEMO\GeoAgent\outputs\tools\20260911\e84ac8e6-19b6-4c59-8a61-f7ab6001b9ff\crop.png

One verified live-HTTP crop artifact is:

    E:\sht\DEMO\GeoAgent\outputs\tools\20260911\70066927-baa6-454a-a98e-55918973aa6a\crop.png

Artifacts contain a type, path, and MIME type. Upload names do not participate in
artifact paths, and no Base64 image is returned.

## 11. Trace

The live trace endpoint returned five recent records after API and UI acceptance.
The latest successful record was `inspect_image`, execution ID
`b1f6214e-e657-49ac-a9e9-1d768c2f0bf4`. Records contain execution ID, tool,
success, start/finish timestamps, duration, optional prompt length, and error type.
They do not retain images or prompt text.

## 12. Integration

The real manual sequence succeeded:

    inspect_image -> crop_image -> analyze_image -> Qwen3-VL -> ToolResult

The cropped image produced a non-empty Qwen answer. This was a manually ordered
tool test, not an agent workflow.

## 13. Performance

Five consecutive real `analyze_image` calls reported GPU allocation:

    8.275, 8.275, 8.275, 8.275, 8.275 GiB

First-to-last growth was **0.000 GiB**. Excluding the auto-load call, measured
ToolExecutor overheads were 1.95, 2.61, 1.94, and 2.04 ms; average **2.13 ms**.
A direct model API call measured 2,709.62 ms total and 2,706.97 ms model latency,
or about 2.65 ms surrounding HTTP overhead. No anomalous tool-layer overhead was
observed.

## 14. Tests

- Default suite: **52 passed, 2 skipped**, 2 dependency deprecation warnings
- Real integrations: **2 passed, 52 deselected**, 2 warnings, 30.64 s
- `pip check`: **No broken requirements found**
- Python compile check: passed
- Real live API acceptance: passed
- Real Gradio queue/event acceptance: passed

The two default skips are the explicitly marked local RTX 4090 integration tests;
both passed when run with `-m integration`.

## 15. Storage

The model and Hugging Face cache remain on E:

    E:\sht\DEMO\GeoAgent\models\Qwen3-VL-4B-Instruct
    E:\sht\DEMO\GeoAgent\cache\huggingface

Tool artifacts and validation outputs are also on E. No tracked file and no
non-virtual-environment workspace file exceeds 10 MiB. `.env` and `.venv` remain
ignored; models, caches, datasets, outputs, checkpoints, and temp roots remain
excluded by `.gitignore`.

## 16. Git

The worktree contains only the intended Phase 2 source, tests, scripts, and docs.
`git diff --check` passed, with Windows LF-to-CRLF notices only. No commit or push
was performed.

## 17. PASS checklist

- [x] Phase 0 and Phase 1 tests have no regression
- [x] BaseTool and ToolContext
- [x] ToolRegistry, discovery, JSON Schema, and duplicate protection
- [x] ToolExecutor, execution ID, ToolResult, safe errors, timeout, and trace
- [x] inspect_image
- [x] crop_image and E-drive artifact
- [x] analyze_image wraps Qwen behind the tool boundary
- [x] UNLOADED auto-loads and continues inference
- [x] Tool and trace APIs
- [x] Gradio Analyze uses the Tool API and shows real tool execution state
- [x] No false agent reasoning or autonomous-agent claim
- [x] Real Qwen Tool API integration
- [x] Manual inspect -> crop -> analyze sequence
- [x] Five calls without GPU allocation growth
- [x] Low executor overhead
- [x] Tests, integration, dependency check, README, and tool-system docs
- [x] Storage and Git safety

## 18. Unfinished issues

No Phase 2 product requirement remains unfinished. The optional Windows visual
screenshot helper could not attach a screenshot because its trusted Node process
exited twice. The separate real Gradio page, upload, preview, event-stream, model,
ToolResult, tool panel, and unload acceptance completed successfully.

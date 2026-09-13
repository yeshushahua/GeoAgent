# Phase 3 Validation — Vision Agent

Date: 2026-09-11  
Verdict: **PASS**

## 1. Phase 3 Objective

GeoAgent now accepts a natural-language image task, lets the local Qwen3-VL
policy model select tools from dynamic registry definitions, executes every call
through the Phase 2 ToolExecutor, feeds structured observations back to the model,
and stops with a validated final answer. This is one bounded Vision Agent, without
an agent framework, long-term memory, or multiple agents.

## 2. Architecture

    Gradio
      -> POST /api/v1/agent/run
      -> VisionAgent
      -> QwenAgentPlanner (text-only decision on shared model)
      -> AgentOutputParser
      -> ToolRegistry
      -> ToolExecutor
      -> inspect_image | crop_image | analyze_image
      -> ToolResult observation
      -> next Qwen decision or final answer

Agent modules are separated into schemas, state, prompt, planner, parser, errors,
trace, and service. The planner and `analyze_image` reference the same
`ModelManager`; only one Qwen model instance is loaded.

## 3. Agent Loop

The explicit asynchronous loop initializes one `AgentState`, auto-loads Qwen when
needed, requests one structured decision, validates it, executes a tool when
selected, appends the observation, and asks Qwen to decide again. A final decision
ends the run. The default maximum is six steps and the public range is 1–10.

The system policy now defines each named corner quarter as one quadrant from a
2×2 split. Midpoints use `floor(width / 2)` and `floor(height / 2)` consistently
when a dimension is odd. Coordinate selection still comes from Qwen after it sees
the `inspect_image` observation; the Agent service contains no phrase matcher or
predefined crop workflow.

## 4. Agent Schemas

Pydantic models define `AgentRequest`, `AgentToolDefinition`, `AgentToolCall`,
`AgentFinal`, the discriminated `AgentDecision`, `AgentObservation`, `AgentStep`,
`AgentMetadata`, `AgentResponse`, `AgentErrorDetail`, `AgentRunTrace`, and
`AgentState`. Extra fields are rejected on all decision and response contracts.

## 5. Tool Discovery

`VisionAgent` calls `ToolRegistry.list_tools()` for every run and converts each
registered tool's name, LLM-readable description, and Pydantic-generated JSON
Schema into `AgentToolDefinition`. The Agent loop contains no list of specific
tools and requires no dispatch change when a future tool is registered.

The Phase 3 registry still contains exactly:

    analyze_image
    crop_image
    inspect_image

## 6. Decision Protocol

Qwen receives the policy in an actual system role and returns exactly one of:

    {"type":"tool_call","tool_name":"inspect_image","arguments":{"image_path":"..."}}

    {"type":"final","answer":"..."}

The parser extracts JSON using `json.JSONDecoder`, never `eval` or `exec`, then
applies the discriminated decision schema, registry membership validation, and the
selected tool's input schema. Pure JSON, fenced JSON, and lightly wrapped JSON are
supported. Raw model output and hidden reasoning are not returned or retained.

## 7. Tool Execution

Every accepted tool call goes through `ToolRegistry` and the existing
`ToolExecutor`. The Agent never imports or invokes a concrete tool function.
Tool execution IDs remain visible in Agent steps and correlate with the Phase 2
tool trace.

## 8. Observation Handling

The next planner turn receives bounded structured observations containing tool
name, success, data, artifacts, and safe errors. Failed ToolResults become
observations instead of exceptions escaping the Agent API. The model can correct
the next call or return a user-facing limitation.

Agent run traces retain run ID, timestamps, duration, step count, tool sequence,
artifact count, prompt length, and error type. They exclude user prompt text,
system prompts, raw model decisions, images, Base64, tracebacks, and chain-of-thought.

## 9. Artifact Propagation

Scenario D proved the complete artifact chain. `crop_image` wrote a persistent
Artifact, the following Qwen decision used that artifact as
`analyze_image.image_path`, and the Agent response/UI preserved the crop preview.
No image bytes entered Agent state.

Final integration artifact:

    E:\sht\DEMO\GeoAgent\outputs\tools\20260911\dc824dfd-6225-4cfb-9acd-ddc6d6e76143\crop.png

## 10. Error Handling

Defined errors include parse, validation, unknown tool, maximum steps, duplicate
call, and model errors. API errors are structured and client-safe. Unit tests cover
malformed output, unknown tools, invalid tool arguments, missing images, blank
tasks, failed crops, repair success, maximum steps, and unexpected failures.
Server tracebacks remain in logs.

## 11. Loop Protection

The loop has three independent controls:

1. Maximum 1–10 steps, default 6.
2. Normalized `tool_name + arguments` signature detection blocks identical calls.
3. The system policy requires a final answer immediately after a successful
   semantic analysis for a content-only request, preventing prompt-rephrased
   repeat analyses.

Malformed decisions receive one configured repair attempt. Repair is bounded to
0–2 attempts and cannot create an infinite loop.

## 12. API

New endpoints:

- `POST /api/v1/agent/run`
- `GET /api/v1/agent/executions?limit=20`

The run endpoint accepts an image, message, optional max steps, and optional token
limit. It never accepts a caller-selected tool name. Model and Tool APIs from
Phases 1 and 2 remain available.

Live FastAPI validation returned version `0.4.0`, executed Scenario D with run ID
`e4224668-3123-4bd5-8384-64c49d9a04f3`, returned the exact three-tool sequence,
used crop coordinates `(0, 0, 240, 150)`, created a verified 240×150 artifact,
passed that artifact to `analyze_image`, exposed an Agent trace, and kept the Tool
API functional.

## 13. Gradio UI

The primary Analyze event now calls only `/api/v1/agent/run`. User-facing labels
are Chinese, while tool names and internal schemas remain English. The UI displays
observable steps, success/failure, tool execution time, final answer, Agent/Planner/
Tool metrics, and the crop artifact preview. Full system/model JSON and manual Tool
controls are in closed advanced accordions. Model auto-load remains automatic.

The live Gradio queue/event acceptance passed upload, Chinese Scenario D, auto-load,
three model-selected tools, Chinese answer, step panel, verified 240×150 crop
preview, manual Tool debugging, and unload. Run ID:
`0df7958c-248b-46f8-b692-6c993698be80`; total 20,761.28 ms, planner
14,429.55 ms, tools 1,881.03 ms, peak VRAM 9.576 GiB.

## 14. Unit Tests

Default suite: **64 passed, 3 skipped**, 2 third-party deprecation warnings.

The new tests cover dynamic discovery, decision parsing, fenced/wrapped JSON,
repair, unknown tools, invalid arguments, max steps, duplicate blocking, tool error
observations, final responses, artifact propagation, safe traces, model auto-load,
shared planner behavior, Agent API schemas, and preservation of Tool/Model APIs.
The three default skips are the marked RTX 4090 tests.

## 15. RTX4090 Integration

Final all-phase integration: **3 passed, 64 deselected**, 2 warnings, 114.84 s.

Environment remained Qwen3-VL-4B-Instruct, RTX 4090, cuda:0, BF16, SDPA,
`local_files_only=true`, single GPU, no quantization, no FlashAttention, and no
vLLM. Phase 1 direct model, Phase 2 Tool system, and Phase 3 Agent integration all
passed in the same final run.

## 16. Scenario A

Request: `告诉我这张图片的宽度、高度和格式。`

- Run ID: `4d56756b-b7e5-4862-a1d5-744cf12e6ef6`
- Tool sequence: `inspect_image`
- Final: `这张图片的宽度是480像素，高度是300像素，格式是PNG。`
- Total: 11,609.84 ms
- Planner: 4,457.89 ms
- Tool: 9.58 ms

No semantic analysis tool was called.

## 17. Scenario B

Request: `分析一下这张图片主要有什么内容。`

- Run ID: `c2239cf3-2057-49a2-8195-f606fea68014`
- Tool sequence: `analyze_image`
- Final: `这张图片主要包含两个几何图形：左侧是一个深绿色的正方形，右侧是一个橙色的圆形，两者在颜色和形状上形成视觉对比。`
- Total: 8,573.97 ms
- Planner: 5,519.15 ms
- Tool: 3,051.44 ms

No unnecessary metadata inspection was called.

## 18. Scenario C

Request: `先检查图片尺寸，然后裁剪左上四分之一区域。`

- Run ID: `9dc3efa2-9f5e-4593-b361-ca8c879602fb`
- Qwen-selected sequence: `inspect_image -> crop_image -> final`
- Observation: width 480, height 300
- Crop: `x1=0, y1=0, x2=240, y2=150`
- Verified artifact size: 240×150
- Artifact: `E:\sht\DEMO\GeoAgent\outputs\tools\20260911\ad73cd3e-2144-4117-be21-995ea26bf6c0\crop.png`

The previous validation incorrectly treated the top-left quarter as
`width/4 × height/4`, recorded `(0, 0, 120, 75)`, and therefore produced only
1/16 of the image area. This is corrected to image-quadrant semantics:
`width/2 × height/2`, which gives 240×150 and approximately 1/4 of the area.

## 19. Scenario D

Request: `先检查图片尺寸，裁剪左上四分之一区域，然后分析裁剪后的内容。`

Actual Qwen-selected sequence:

1. `inspect_image(image_path=original_image)` observed width 480 and height 300,
   execution `a783a70d-0427-4437-9aa6-9b4e4e82d0cc`.
2. `crop_image(image_path=original_image, x1=0, y1=0, x2=240, y2=150)` created
   a verified 240×150 artifact, execution `dc824dfd-6225-4cfb-9acd-ddc6d6e76143`.
3. `analyze_image(image_path=crop.png, prompt_length=17, max_new_tokens=64)` used
   that crop artifact, execution `9b93c0a3-cb81-487b-895f-a6fdd72e025c`.
4. A validated `final` decision returned a Chinese answer describing only the
   blank/light-gray crop region.

## 20. VRAM

Five warm content Agent runs each selected exactly one `analyze_image` call:

| Run | Allocated GiB | Peak GiB |
| --- | ---: | ---: |
| 1 | 8.275 | 9.354 |
| 2 | 8.275 | 9.359 |
| 3 | 8.275 | 9.357 |
| 4 | 8.275 | 9.354 |
| 5 | 8.275 | 9.348 |

First-to-last allocated growth: **0.000 GiB**. There was no monotonic allocation
growth and no second resident Qwen model. Integration Scenario D allocated
8.275 GiB and peaked at 9.570 GiB; live FastAPI D allocated 8.275 GiB and peaked
at 9.572 GiB.

## 21. Latency

| Scenario | Total ms | Planner ms | Tool ms |
| --- | ---: | ---: | ---: |
| A | 12,331.53 | 4,625.15 | 11.13 |
| B | 8,755.45 | 5,727.03 | 3,024.81 |
| C | 9,173.00 | 9,162.17 | 5.77 |
| D | 15,796.43 | 13,936.06 | 1,855.43 |

Repeated single-tool content runs ranged from 8.76 to 10.40 seconds. The measured
multi-step latency is expected because each action and final answer is a separate
real Qwen decision. No tool or planner result was faked or cached.

## 22. Known Limitations

Phase 3 cannot perform object detection, open-vocabulary detection, segmentation,
precise object localization/counting, GeoTIFF processing, GIS analysis, RAG,
fine-tuning, long-term memory, streaming, or multi-agent work. Crop operations
require explicit coordinates or dimension-derived regions. A 4B policy model can
occasionally produce malformed JSON; bounded repair handles one failure, after
which the run stops safely.

The Windows screenshot helper failed twice because its trusted Node process exited.
This prevented attaching a screenshot, while the independent live Gradio page,
upload, queue event, Chinese task, Agent result, artifact download/preview, metrics,
manual Tool control, and unload checks all passed.

## 23. Final Verdict

**PASS — Phase 3 ready for final review.** Qwen3-VL made the Tool decisions; Python
contains no task-keyword router or predefined Scenario C/D workflow. The corrected
quadrant policy, 240×150 crop, observation feedback, artifact propagation, live
HTTP, live Gradio, RTX 4090, and all prior phases were verified. No Phase 4
capability was added, and no commit or push was performed.

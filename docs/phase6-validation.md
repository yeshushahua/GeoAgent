# Phase 6 Validation — Multi-step Workflow

## 1. Phase 6 Goal

Phase 6 upgrades the existing bounded Qwen tool loop into a dependency-aware
multi-step workflow. It does not add a model or framework. The six Phase 0–5 tools,
their registry-generated schemas, `ToolExecutor`, model managers, and Agent trace
remain the execution boundary.

The accepted end-to-end workflow is:

    inspect_image -> crop_image -> detect_open_vocab -> segment_objects
                  -> ResultAggregator -> final

The planner still selects every requested capability. The workflow layer validates
and resolves its artifact and detection references before normal Tool execution.

## 2. Architecture Changes

`AgentState` now owns workflow artifacts, detections, segmentations, original and
active artifact IDs, completed and failed actions, pending goals, and warnings.
`WorkflowController` supplies identity, dependency resolution, state transitions,
compact observations, and normalized duplicate signatures. `ResultAggregator`
builds the authoritative public summary and deterministic Chinese category result.

No LangGraph, external workflow engine, database, Redis, Celery, or multi-agent
layer was introduced.

## 3. Workflow State

State is scoped to one Agent run and keeps:

- the original user request, current step, maximum steps, and observations;
- `original-image-001` and the current analysis source such as `crop-001`;
- completed and failed tool actions, pending completion policy, and warnings;
- structured detections and segmentations with stable per-run IDs;
- full local artifacts for execution and path-free public artifact views.

Planner observations contain artifact IDs, dimensions, category counts, detection
IDs, segment counts, and structured errors. They omit absolute paths, full boxes,
Base64 data, the system prompt, and hidden reasoning. Full geometry remains in state.

## 4. Artifact Management

The original upload is registered as an `analysis_source`. A crop is registered with
the source as its parent and becomes `active_image`. Detection overlays,
segmentation overlays, and masks are `visualization` artifacts; they never replace
the active analysis source and are rejected as model inputs.

Planner references support `original_image`, `active_image`, or a stable artifact
ID. `WorkflowController` resolves that reference to the validated local path only
immediately before `ToolExecutor` runs. An unknown ID yields
`ARTIFACT_NOT_FOUND`. Agent crop calls require inspected width and height metadata,
which prevents guessed fractional coordinates.

## 5. Detection / Segmentation Linking

Every detector instance becomes `det-001`, `det-002`, and so on, with source
artifact, class, confidence, and exact `xyxy` box. `segment_objects` accepts these
IDs in Agent workflows. The controller verifies all IDs exist and share the chosen
source, resolves their exact boxes internally, and removes the IDs before the
existing SAM Tool schema reaches inference.

SAM returns an `input_index` for every success or failure. The workflow records a
`seg-...` item with its source detection ID, source artifact, box, mask ID, overlay
ID, pixel area, and pixel ratio. Multi-box inference is attempted once. If the
batch raises a SAM error, the Tool retries each box with the same loaded manager so
successful instances survive an isolated failure.

## 6. Failure Recovery

Validated cases include:

- one requested class with zero detections while other classes continue to SAM;
- an invalid artifact producing `ARTIFACT_NOT_FOUND`, followed by a corrected call;
- a fractional crop attempted without dimensions producing
  `IMAGE_METADATA_REQUIRED`, followed by inspect and a correct crop;
- one SAM box failure preserving its detection, successful sibling masks, and a
  category-level failed segmentation count;
- empty boxes producing `NO_VALID_BOXES` and all out-of-bounds boxes producing
  `INVALID_BBOX` without an unhandled exception.

Artifact-aware duplicate keys include tool name, source artifact, normalized class
labels, detection IDs or boxes, and other arguments. Calls on two images are not
duplicates.

## 7. Result Aggregation

The aggregator reports detected, segmented, and failed counts; detection
confidences; per-instance mask areas; instance area sum; union mask area and ratio;
and source artifact IDs for every requested category, including zero-result classes.

The category total is calculated by loading its masks and taking a logical union per
source artifact. A synthetic overlap test has two four-pixel masks: the instance sum
is 8 pixels and the union is 6 pixels, proving overlap is not counted twice.
User-facing area is always described as an image-pixel ratio, never square metres,
hectares, or geographic area.

## 8. Tests

Phase 6 unit coverage includes state updates, artifact parents and roles, active
image transitions, detection and segmentation mapping, zero categories, partial SAM
failure, crop metadata dependency, invalid artifact recovery, normalized duplicate
calls, mask union area, compact planner context, UI trace, and path redaction.

Validation commands:

    .\.venv\Scripts\python.exe -m compileall backend
    .\.venv\Scripts\python.exe -m pytest
    .\.venv\Scripts\python.exe -m pytest -m integration -s
    .\.venv\Scripts\python.exe -m pip check
    git diff --check

The final command results are recorded in the completion report. The third-party
Starlette/httpx and TorchScript deprecation warnings do not affect behavior.

Final results: 114 passed, 11 skipped for the ordinary suite and
11 passed, 114 deselected for the RTX4090 integration suite. compileall,
pip check, and git diff --check also passed.

## 9. RTX4090 Integration

`test_phase6_integration.py` loads one Qwen, YOLOE, and SAM manager set and runs four
Chinese Agent tasks sequentially on `test/img/1.jpg`. YOLOE and SAM each report
`load_count=1`; both remain on `cuda:0`. The Phase 6 run reached a 17.174 GiB peak.

| Case | Successful Tool sequence | Result |
|---|---|---|
| Inspect + two-class detection | inspect, detect-open-vocab | helmet 3, vest 2, no SAM |
| Two-class detection + segmentation | detect-open-vocab, segment | 5/5 segmented |
| Inspect + left-half crop + detect + segment | inspect, crop, detect-open-vocab, segment | crop source retained, 4/4 segmented |
| Helmet + vest + excavator | detect-open-vocab, segment | 5/5 present targets segmented; excavator 0 |

The left-half result used `crop-001`: yellow safety helmet 2 with 4.131904% union
area, and reflective vest 2 with 17.683015% union area. The original-image result
used 2.792446% for three helmets and 12.339450% for two vests. Full evidence is in
`E:\sht\DEMO\GeoAgent\outputs\benchmarks\phase6\integration.json`.

## 10. Manual Validation

FastAPI Live passed the complete left-half workflow. It verified the exact tool
sequence, `crop-001` as the detector and segmenter source, every detection-to-mask
box link, public artifact path omission, workflow metrics, and Agent trace. Peak
memory was 17.203 GiB.

Gradio Live passed the helmet, vest, and absent excavator task through the real
`/analyze` event. The Workflow panel contained steps, input summaries, detection
IDs, artifact dependencies, aggregate results, durations, and final output. Five
masks and the overlay were readable. The returned debug JSON contained no D: or E:
absolute path. The real `/execute_tool` event also passed manual `inspect_image`.

Reports:

    E:\sht\DEMO\GeoAgent\outputs\benchmarks\phase6\live_api_acceptance.json
    E:\sht\DEMO\GeoAgent\outputs\benchmarks\phase6\gradio_acceptance.json

## 11. Known Limitations

- Workflow state and IDs are per run and in memory; workflows cannot be resumed
  after a process restart.
- The bounded loop allows at most 10 decisions. A planner correction consumes one
  of those decisions.
- SAM uses detector boxes as prompts. Point prompts and interactive mask refinement
  are outside this phase.
- Pixel ratios have no physical scale or geospatial meaning.
- Qwen planning is generative. Dependency validation prevents invalid execution and
  enables a bounded correction, but does not make every first decision deterministic.

## 12. Final Result

Phase 6: PASS

The existing six tools and four fixed local model stacks remain intact. Multi-step
state, stable artifact dependencies, multi-class batching, instance links, partial
recovery, union aggregation, workflow metrics, and Gradio observability are present.

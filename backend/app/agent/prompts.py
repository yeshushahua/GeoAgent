from __future__ import annotations

import json

from backend.app.agent.state import AgentState

AGENT_SYSTEM_PROMPT = """You are GeoAgent's visual task control agent.
Choose actions only from the tool definitions supplied at runtime. Never invent a
tool, argument, observation, image fact, or external capability. Use inspect_image
for exact dimensions/format, crop_image for pixel-coordinate cropping, and
analyze_image for visual semantic understanding. Use detect_objects for precise
closed-set COCO object classes, counts, confidence scores, and bounding boxes. Use
detect_open_vocab for arbitrary user-named text categories outside or more specific
than COCO. Use segment_objects only for precise instance masks from explicit bbox
prompts already present in a detection observation. Translate requested targets into
visually grounded English noun phrases when calling detect_open_vocab. For a wearable
or object part, include its immediate visible carrier or relation when that clarifies
the visual concept: for example, use "a person wearing a yellow helmet" rather than
the isolated words "yellow helmet". Do not pass an imperative or the entire user
request as a class. Use the fewest necessary tools.
Mandatory selection policy:
- A request only about width, height, mode, format, file size, or aspect ratio uses
  inspect_image and does not use analyze_image.
- A request only about visible content or meaning uses analyze_image immediately
  and does not use inspect_image first.
- A crop described as a fraction of the image uses inspect_image first only when
  dimensions are not already present in observations.
- A request to detect objects, count each object class, locate objects with boxes,
  or report detection confidence for ordinary COCO categories MUST use
  detect_objects. Do not substitute analyze_image to estimate these facts.
- A request for a dynamic, non-COCO, visually specific, or user-defined target such
  as a yellow safety helmet, tower crane, or tent on a riverbank MUST use
  detect_open_vocab with classes containing the requested English target phrases.
  Do not reject such labels as unsupported COCO classes.
- A request that only says detect, find, locate, count, or report boxes for an
  open-vocabulary target MUST call detect_open_vocab and then return final. It MUST
  NOT call segment_objects unless the user explicitly asks for segmentation, a
  mask, a precise outline, extraction, or pixel-area proportion. The phrase
  "yellow safety helmet" describes a target class; it does not request a mask.
- A request to segment, extract a precise outline, create a mask, or compute the
  fraction of image pixels occupied by a target requires detection first and then
  segment_objects. For dynamic targets use detect_open_vocab; for ordinary COCO
  targets detect_objects may be used. Copy every requested detection bbox exactly
  from the detection observation into segment_objects.boxes. Use the exact same
  image_path examined by the detector. SAM must never invent or relocate boxes.
  Detector annotated images such as open-vocab-annotated.jpg and annotated.jpg are
  display-only artifacts and MUST NEVER be passed to segment_objects. For example,
  after crop_image(path=crop.png) then detect_open_vocab(image_path=crop.png), the
  required call is segment_objects(image_path=crop.png, boxes=<exact detection
  observation boxes>), never a detector annotation artifact.
- If the relevant detector returns detection_count=0, return a truthful final answer
  immediately. Do not call segment_objects with fabricated or empty boxes.
- A detection-only request uses detect_objects directly and then returns final.
  Do not call inspect_image or analyze_image unless the user also requests their
  distinct capability.
- When the user explicitly requests detection followed by broader visual analysis,
  call detect_objects first, then call analyze_image. The analyze_image prompt and
  final answer must treat the structured detection observation as authoritative
  for classes, counts, boxes, and confidence; never guess replacements.
  A successful detection observation is NOT enough to finish an explicitly
  requested two-stage detect-and-analyze task. The required sequence is
  detect_objects -> analyze_image -> final. For example, Chinese "先检测图中的目标，
  再结合检测结果分析这张图片" requires exactly those two tools in that order.
  analyze_image must use the same image_path that detect_objects examined and must
  receive a prompt grounded in the detection observation. The annotated image is
  a display artifact, not a replacement analysis input.
For region requests whose coordinates depend on image dimensions, inspect first,
then compute coordinates from the observation. After crop_image, any requested
analysis of the crop MUST use the returned crop artifact path, not the original.
Likewise, after crop_image, any requested detection of the crop MUST call
detect_objects or detect_open_vocab with the returned crop artifact path, not the
original image. Any following segment_objects call MUST use that same crop artifact
path and the bboxes returned for the crop coordinate space.
Spatial rules for rectangular corner regions:
- "top-left quarter" (Chinese: "左上四分之一") means the top-left quadrant: split
  the image into 2 equal parts horizontally and 2 equal parts vertically. It is
  x1=0, y1=0, x2=floor(width / 2), y2=floor(height / 2).
- "top-right quarter" is x1=floor(width / 2), y1=0, x2=width,
  y2=floor(height / 2).
- "bottom-left quarter" is x1=0, y1=floor(height / 2),
  x2=floor(width / 2), y2=height.
- "bottom-right quarter" is x1=floor(width / 2),
  y1=floor(height / 2), x2=width, y2=height.
Always use integer pixel coordinates. When a half is not an integer, use floor
for both horizontal and vertical midpoints consistently. A corner quarter is one
quadrant with approximately 1/4 of the image area. NEVER use width / 4 together
with height / 4; that produces only 1/16 of the image area. Example: after an
inspect_image observation of width=480 and height=300, the top-left-quarter crop
coordinates are exactly x1=0, y1=0, x2=240, y2=150.
These region rules apply only when the user requests a crop or region. A request
only about visible content still MUST call analyze_image immediately and MUST NOT
call inspect_image first, even if the request includes a run number or label.
After every observation, decide again whether another tool is needed or a final
answer is supported. Do not repeat an identical tool call. If a tool fails, either
correct the call or explain the limitation. Do not reveal chain-of-thought.
After one successful analyze_image observation for a content-only request, the
next decision MUST be final. Do not call analyze_image again with a rephrased
prompt to inspect the same image.
detect_objects remains a closed-set COCO detector. detect_open_vocab is text-prompt
open-vocabulary localization, and segment_objects reports pixel masks and pixel-area
ratios only. Never describe mask area as square metres, hectares, or geographic area.
Never fabricate a detection or mask, and never claim tracking capability.

Each response MUST contain exactly one JSON object and nothing else:
{"type":"tool_call","tool_name":"name","arguments":{...}}
or
{"type":"final","answer":"user-facing answer"}

Default final answers to Simplified Chinese. Use English only when the user
explicitly requests English. The controller selects and sequences tools; it must
not substitute an unsupported visual claim for an analyze_image observation."""


def build_planner_prompt(
    state: AgentState,
    repair_output: str | None = None,
    repair_error: str | None = None,
) -> str:
    definitions = [item.model_dump(mode="json") for item in state.tool_definitions]
    observations = []
    for item in state.observations[-state.max_steps:]:
        result = item.result
        # Detector renderings are presentation artifacts, never inputs to another model.
        # Omitting them here prevents a small planner model from confusing a preview
        # path with the authoritative source image recorded in result.data.
        planner_artifacts = [] if result.tool in {
            "detect_objects", "detect_open_vocab", "segment_objects"
        } else [
            artifact.model_dump(mode="json", by_alias=True) for artifact in result.artifacts
        ]
        observations.append({
            "step": item.step,
            "tool_name": item.tool_name,
            "success": result.success,
            "data": result.data,
            "artifacts": planner_artifacts,
            "error": result.error.model_dump(mode="json") if result.error else None,
        })
    context = {
        "user_request": state.user_message,
        "original_image_path": state.original_image_path.replace("\\", "/"),
        "max_new_tokens_for_analyze_image": state.max_new_tokens,
        "available_tools": definitions,
        "observations": observations,
        "remaining_steps": state.max_steps - state.step_count,
    }
    if state.observations:
        latest = state.observations[-1].result
        if latest.success and latest.tool in {"detect_objects", "detect_open_vocab"}:
            source = latest.data.get("source_image_path", state.original_image_path)
            count = latest.data.get("detection_count", 0)
            if count == 0:
                context["next_action_contract"] = (
                    "The latest detector found zero targets. The next decision must be final."
                )
            else:
                context["next_action_contract"] = (
                    "Bounding boxes being available does not itself authorize segmentation. "
                    "Re-read the original user request. If it asks only to detect/find/locate/count, "
                    "the next decision must be final. Only an explicit request for masks, precise "
                    "segmentation/outlines, extraction, or pixel area authorizes segment_objects. "
                    f"If authorized, segment_objects.image_path must be exactly {source!s}; copy the "
                    "latest detection boxes exactly. Detector preview artifacts are unavailable as "
                    "model inputs."
                )
    parts = ["Runtime context:", json.dumps(context, ensure_ascii=False)]
    if repair_output is not None:
        parts.extend([
            "Your previous response was not valid for the required Agent schema.",
            f"Validation error: {repair_error or 'invalid output'}",
            "Previous response:",
            repair_output,
            "Repair it now. Output only one valid JSON object.",
        ])
    else:
        parts.append("Choose the next tool call or final answer now. Output JSON only.")
    return "\n\n".join(parts)

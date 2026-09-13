from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from uuid import uuid4

import anyio

from backend.app.agent.errors import (
    AgentError,
    AgentModelError,
    AgentParseError,
    AgentValidationError,
    DuplicateToolCallError,
    MaxStepsExceededError,
    UnknownToolError,
)
from backend.app.agent.parser import AgentOutputParser
from backend.app.agent.planner import AgentPlanner, QwenAgentPlanner
from backend.app.agent.schemas import (
    AgentErrorDetail,
    AgentFinal,
    AgentMetadata,
    AgentObservation,
    AgentRequest,
    AgentResponse,
    AgentRunTrace,
    AgentStep,
    AgentToolCall,
    AgentToolDefinition,
)
from backend.app.agent.state import AgentState
from backend.app.agent.trace import AgentTraceStore
from backend.app.models.errors import VlmError
from backend.app.models.manager import ModelManager, ModelState
from backend.app.models.qwen_vl import get_gpu_memory
from backend.app.schemas.tool_result import Artifact, ToolError, ToolResult
from backend.app.tools.executor import ToolExecutor
from backend.app.tools.registry import ToolRegistry


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class VisionAgent:
    def __init__(
        self,
        registry: ToolRegistry,
        executor: ToolExecutor,
        model_manager: ModelManager,
        trace_store: AgentTraceStore,
        logger,
        repair_attempts: int = 1,
        planner_max_new_tokens: int = 256,
        planner: AgentPlanner | None = None,
    ):
        self.registry = registry
        self.executor = executor
        self.model_manager = model_manager
        self.trace_store = trace_store
        self.logger = logger
        self.repair_attempts = repair_attempts
        self.parser = AgentOutputParser()
        self.planner = planner or QwenAgentPlanner(model_manager, planner_max_new_tokens)

    async def run(self, request: AgentRequest) -> AgentResponse:
        run_id = str(uuid4())
        started_at = _utc_now()
        timer = time.perf_counter()
        definitions = [
            AgentToolDefinition.model_validate({
                "name": item["name"],
                "description": item["description"],
                "input_schema": item["input_schema"],
            })
            for item in self.registry.list_tools()
        ]
        state = AgentState(
            run_id=run_id,
            user_message=request.message,
            original_image_path=request.image_path,
            tool_definitions=definitions,
            max_steps=request.max_steps,
            max_new_tokens=request.max_new_tokens,
            started_at=started_at,
        )
        planner_ms = 0.0
        tool_ms = 0.0
        agent_model_load_ms = 0.0
        allocated = 0.0
        peak = 0.0
        try:
            try:
                safe_path = self.executor.context.validate_read_path(Path(request.image_path))
                state.original_image_path = str(safe_path)
            except (OSError, ValueError) as exc:
                raise AgentValidationError("Image is missing or outside configured storage") from exc
            if self.model_manager.state == ModelState.ERROR:
                raise AgentModelError("Qwen3-VL is in ERROR state; unload it before retrying")
            if self.model_manager.state == ModelState.UNLOADED:
                self.logger.info("[%s] Auto-loading Qwen3-VL for VisionAgent", run_id)
                load_timer = time.perf_counter()
                await anyio.to_thread.run_sync(self.model_manager.load_model)
                agent_model_load_ms = (time.perf_counter() - load_timer) * 1000

            for index in range(1, request.max_steps + 1):
                state.step_count = index
                decision, decision_ms, turn_allocated, turn_peak = await self._decide(state)
                planner_ms += decision_ms
                allocated = turn_allocated
                peak = max(peak, turn_peak)
                if isinstance(decision, AgentFinal):
                    state.final_answer = decision.answer.strip()
                    state.steps.append(AgentStep(
                        index=index,
                        decision_type="final",
                        success=True,
                        duration_ms=decision_ms,
                        planner_duration_ms=decision_ms,
                        state_transition="DECIDING -> COMPLETED",
                    ))
                    return self._finish(
                        state, timer, planner_ms, tool_ms, allocated, peak, success=True,
                        agent_model_load_ms=agent_model_load_ms,
                    )

                signature = json.dumps(
                    {"tool_name": decision.tool_name, "arguments": decision.arguments},
                    sort_keys=True,
                    ensure_ascii=False,
                )
                if signature in state.call_signatures:
                    duplicate = DuplicateToolCallError(
                        f"Duplicate tool call blocked: {decision.tool_name}"
                    )
                    result = ToolResult(
                        success=False,
                        tool=decision.tool_name,
                        error=ToolError(
                            code=duplicate.code,
                            type=duplicate.code,
                            message=str(duplicate),
                        ),
                    )
                    state.observations.append(AgentObservation(
                        step=index, tool_name=decision.tool_name, result=result
                    ))
                    state.steps.append(AgentStep(
                        index=index,
                        decision_type="blocked",
                        tool_name=decision.tool_name,
                        arguments_summary=self._safe_arguments(state, decision),
                        observation_summary=self._safe_observation(result),
                        success=False,
                        duration_ms=decision_ms,
                        planner_duration_ms=decision_ms,
                        error_type=duplicate.code,
                        state_transition="DECIDING -> DUPLICATE_BLOCKED -> DECIDING",
                    ))
                    continue

                state.call_signatures.add(signature)
                state.tool_calls.append(decision)
                self.logger.info(
                    "[%s] step=%d decision=tool_call tool=%s",
                    run_id, index, decision.tool_name,
                )
                result = await self.executor.execute(decision.tool_name, decision.arguments)
                duration = float(result.metadata.get("duration_ms", 0))
                tool_ms += duration
                tool_peak = result.metadata.get("gpu_peak_gb")
                if isinstance(tool_peak, (int, float)):
                    peak = max(peak, float(tool_peak))
                tool_allocated = result.metadata.get("gpu_allocated_gb")
                if isinstance(tool_allocated, (int, float)):
                    allocated = float(tool_allocated)
                for artifact in result.artifacts:
                    if artifact.path not in {item.path for item in state.artifacts}:
                        state.artifacts.append(artifact)
                state.observations.append(AgentObservation(
                    step=index, tool_name=decision.tool_name, result=result
                ))
                state.steps.append(AgentStep(
                    index=index,
                    decision_type="tool_call",
                    tool_name=decision.tool_name,
                    arguments_summary=self._executed_arguments_summary(
                        state, decision, result
                    ),
                    observation_summary=self._safe_observation(result),
                    success=result.success,
                    execution_id=str(result.metadata.get("execution_id", "")) or None,
                    duration_ms=round(decision_ms + duration, 2),
                    planner_duration_ms=round(decision_ms, 2),
                    tool_duration_ms=round(duration, 2),
                    model_load_duration_ms=float(result.metadata.get("model_load_ms", 0)),
                    prompt_encoding_duration_ms=float(
                        result.metadata.get("prompt_encoding_ms", 0)
                    ),
                    model_inference_duration_ms=float(result.metadata.get("inference_ms", 0)),
                    tool_overhead_ms=float(result.metadata.get("tool_overhead_ms", 0)),
                    artifacts=result.artifacts,
                    error_type=result.error.type if result.error else None,
                    state_transition="DECIDING -> TOOL_EXECUTION -> OBSERVING -> DECIDING",
                ))
            raise MaxStepsExceededError(
                f"Agent stopped safely after the maximum of {request.max_steps} steps"
            )
        except (AgentError, VlmError) as exc:
            self.logger.exception("[%s] Agent run failed: %s", run_id, exc.code)
            return self._finish(
                state, timer, planner_ms, tool_ms, allocated, peak,
                success=False, error=AgentErrorDetail(type=exc.code, message=str(exc)),
                agent_model_load_ms=agent_model_load_ms,
            )
        except Exception:
            self.logger.exception("[%s] Unexpected agent failure", run_id)
            return self._finish(
                state, timer, planner_ms, tool_ms, allocated, peak,
                success=False,
                error=AgentErrorDetail(
                    type="AGENT_ERROR", message="Agent execution failed; see server log"
                ),
                agent_model_load_ms=agent_model_load_ms,
            )

    async def _decide(self, state: AgentState):
        planner_ms = 0.0
        allocated = 0.0
        peak = 0.0
        repair_output = None
        repair_error = None
        for attempt in range(self.repair_attempts + 1):
            turn = await self.planner.decide(state, repair_output, repair_error)
            planner_ms += turn.latency_ms
            allocated = turn.allocated_vram_gib
            peak = max(peak, turn.peak_vram_gib)
            try:
                return self.parser.parse(turn.output, self.registry), planner_ms, allocated, peak
            except (AgentParseError, AgentValidationError, UnknownToolError) as exc:
                if attempt >= self.repair_attempts:
                    raise
                repair_output = turn.output
                repair_error = str(exc)
                self.logger.warning(
                    "[%s] step=%d repairing invalid agent decision type=%s",
                    state.run_id, state.step_count, exc.code,
                )
        raise AgentParseError("Agent decision repair failed")

    @staticmethod
    def _safe_arguments(state: AgentState, call: AgentToolCall) -> dict:
        summary = {}
        for key, value in call.arguments.items():
            if key == "prompt" and isinstance(value, str):
                summary["prompt_length"] = len(value)
                detection_observations = sum(
                    1 for item in state.observations
                    if item.tool_name in {"detect_objects", "detect_open_vocab"}
                    and item.result.success
                )
                if detection_observations:
                    summary["detection_observation_count"] = detection_observations
            elif key == "image_path" and isinstance(value, str):
                resolved = str(Path(value))
                summary[key] = (
                    "original_image" if resolved == state.original_image_path else Path(value).name
                )
            else:
                summary[key] = value
        return summary

    @classmethod
    def _executed_arguments_summary(
        cls, state: AgentState, call: AgentToolCall, result: ToolResult
    ) -> dict:
        """Merge executed defaults without losing Agent artifact/context markers."""
        summary = cls._safe_arguments(state, call)
        executed = result.metadata.get("arguments_summary", {})
        if isinstance(executed, dict):
            for key, value in executed.items():
                if key not in {"image_path", "prompt_length"}:
                    summary[key] = value
        return summary

    @staticmethod
    def _safe_observation(result: ToolResult) -> dict:
        if not result.success:
            return {}
        if result.tool == "analyze_image":
            answer = result.data.get("answer")
            return {"answer_length": len(answer)} if isinstance(answer, str) else {}
        allowed = {
            "inspect_image": {"width", "height", "mode", "format", "file_size", "aspect_ratio"},
            "crop_image": {"width", "height"},
            "detect_objects": {
                "source_image_path", "image_width", "image_height", "detection_count",
                "class_counts", "detections"
            },
            "detect_open_vocab": {
                "source_image_path", "image_width", "image_height", "requested_classes",
                "detection_count", "class_counts", "detections",
            },
            "segment_objects": {
                "image_width", "image_height", "segment_count", "segments",
                "overlay_artifact_path",
            },
        }.get(result.tool, set())
        summary = {key: value for key, value in result.data.items() if key in allowed}
        for key in ("source_image_path", "overlay_artifact_path"):
            if isinstance(summary.get(key), str):
                summary[key] = Path(summary[key]).name
        return summary

    def _finish(
        self,
        state: AgentState,
        timer: float,
        planner_ms: float,
        tool_ms: float,
        allocated: float,
        peak: float,
        success: bool,
        error: AgentErrorDetail | None = None,
        agent_model_load_ms: float = 0.0,
    ) -> AgentResponse:
        state.finished_at = _utc_now()
        wall_duration_ms = (time.perf_counter() - timer) * 1000
        # Preserve a coherent public breakdown even for injected planners whose
        # reported latency can exceed their test-double wall time.
        state.duration_ms = round(max(
            wall_duration_ms, planner_ms + tool_ms + agent_model_load_ms
        ), 2)
        state.status = "COMPLETED" if success else "FAILED"
        state.error = error
        if allocated == 0:
            memory = get_gpu_memory()
            allocated = memory.allocated_gb
            peak = max(peak, memory.peak_allocated_gb)
        framework_stage_ms = max(0.0, state.duration_ms - planner_ms - tool_ms)
        tool_model_load_ms = sum(step.model_load_duration_ms for step in state.steps)
        metadata = AgentMetadata(
            model="Qwen3-VL-4B-Instruct",
            device=self.model_manager.settings.vlm_device,
            step_count=len(state.steps),
            tool_call_count=len(state.tool_calls),
            total_duration_ms=state.duration_ms,
            planner_duration_ms=round(planner_ms, 2),
            tool_duration_ms=round(tool_ms, 2),
            framework_overhead_ms=round(framework_stage_ms, 2),
            model_load_duration_ms=round(agent_model_load_ms + tool_model_load_ms, 2),
            agent_model_load_duration_ms=round(agent_model_load_ms, 2),
            tool_model_load_duration_ms=round(tool_model_load_ms, 2),
            framework_runtime_overhead_ms=round(
                max(0.0, framework_stage_ms - agent_model_load_ms), 2
            ),
            prompt_encoding_duration_ms=round(sum(
                step.prompt_encoding_duration_ms for step in state.steps
            ), 2),
            model_inference_duration_ms=round(sum(
                step.model_inference_duration_ms for step in state.steps
            ), 2),
            tool_overhead_ms=round(sum(step.tool_overhead_ms for step in state.steps), 2),
            allocated_vram_gib=round(allocated, 3),
            peak_vram_gib=round(peak, 3),
        )
        response = AgentResponse(
            success=success,
            answer=state.final_answer or "",
            run_id=state.run_id,
            status=state.status,
            steps=state.steps,
            artifacts=state.artifacts,
            metadata=metadata,
            error=error,
        )
        self.trace_store.add(AgentRunTrace(
            run_id=state.run_id,
            success=success,
            started_at=state.started_at,
            finished_at=state.finished_at,
            duration_ms=state.duration_ms,
            step_count=len(state.steps),
            tool_sequence=[step.tool_name for step in state.steps if step.tool_name],
            artifact_count=len(state.artifacts),
            prompt_length=len(state.user_message),
            error_type=error.type if error else None,
        ))
        self.logger.info(
            "[%s] Agent completed success=%s steps=%d tools=%d duration_ms=%.2f",
            state.run_id, success, len(state.steps), len(state.tool_calls), state.duration_ms,
        )
        return response

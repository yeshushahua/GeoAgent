from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import anyio

from backend.app.agent.prompts import AGENT_SYSTEM_PROMPT, build_planner_prompt
from backend.app.agent.state import AgentState
from backend.app.models.manager import ModelManager


@dataclass(frozen=True)
class AgentPlannerTurn:
    output: str
    latency_ms: float
    allocated_vram_gib: float
    peak_vram_gib: float


class AgentPlanner(Protocol):
    async def decide(
        self,
        state: AgentState,
        repair_output: str | None = None,
        repair_error: str | None = None,
    ) -> AgentPlannerTurn: ...


class QwenAgentPlanner:
    def __init__(self, model_manager: ModelManager, planner_max_new_tokens: int = 256):
        self.model_manager = model_manager
        self.planner_max_new_tokens = planner_max_new_tokens

    async def decide(self, state, repair_output=None, repair_error=None):
        prompt = build_planner_prompt(state, repair_output, repair_error)
        result = await anyio.to_thread.run_sync(
            self.model_manager.plan,
            prompt,
            self.planner_max_new_tokens,
            AGENT_SYSTEM_PROMPT,
        )
        return AgentPlannerTurn(
            output=result.text,
            latency_ms=result.latency_ms,
            allocated_vram_gib=result.gpu.allocated_gb,
            peak_vram_gib=result.gpu.peak_allocated_gb,
        )

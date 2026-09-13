from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from backend.app.core.config import Settings
from backend.app.models.manager import ModelManager
from backend.app.tools.trace import ToolTraceStore

if TYPE_CHECKING:
    from backend.app.detection.manager import DetectorManager


@dataclass(frozen=True)
class ToolContext:
    settings: Settings
    model_manager: ModelManager
    logger: logging.Logger
    trace_store: ToolTraceStore
    detector_manager: DetectorManager | None = None

    def validate_read_path(self, value: Path) -> Path:
        path = value.expanduser().resolve()
        allowed = (self.settings.project_root.resolve(), self.settings.storage_root.resolve())
        if not any(path == root or path.is_relative_to(root) for root in allowed):
            raise ValueError("Image path must be inside the project or configured storage root")
        if not path.is_file():
            raise FileNotFoundError(f"Image does not exist: {path}")
        return path

    def tool_output_dir(self, execution_id: str, started_date: str) -> Path:
        root = (self.settings.output_dir / "tools").resolve()
        if not root.is_relative_to(self.settings.storage_root.resolve()):
            raise RuntimeError("Tool output directory escapes configured storage root")
        target = root / started_date / execution_id
        target.mkdir(parents=True, exist_ok=False)
        return target

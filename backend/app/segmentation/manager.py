from __future__ import annotations

from enum import StrEnum
import gc
import logging
from pathlib import Path
from threading import RLock
import time
from typing import Any, Callable

import numpy as np
from PIL import Image
import torch

from backend.app.core.config import Settings
from backend.app.detection.manager import get_detector_gpu_memory
from backend.app.detection.schemas import BoundingBox
from backend.app.segmentation.errors import (
    InvalidSegmentationPromptError,
    SegmentationBusyError,
    SegmentationCudaUnavailableError,
    SegmentationFilesMissingError,
    SegmentationInferenceError,
    SegmentationLoadError,
)
from backend.app.segmentation.schemas import (
    SegmentationPrediction,
    SegmentationRun,
    SegmentInstance,
)


class SegmentationState(StrEnum):
    UNLOADED = "UNLOADED"
    LOADING = "LOADING"
    READY = "READY"
    ERROR = "ERROR"


def _default_factory(path: Path):
    from ultralytics import SAM

    return SAM(str(path))


def _mask_array(value: Any) -> np.ndarray:
    for method in ("detach", "cpu"):
        if hasattr(value, method):
            value = getattr(value, method)()
    return np.asarray(value)


class SegmentationManager:
    def __init__(
        self,
        settings: Settings,
        logger: logging.Logger | None = None,
        factory: Callable[[Path], Any] | None = None,
        cuda_available: Callable[[], bool] = torch.cuda.is_available,
    ):
        self.settings = settings
        self.logger = logger or logging.getLogger("geoagent")
        self._factory = factory or _default_factory
        self._cuda_available = cuda_available
        self._model: Any | None = None
        self._state = SegmentationState.UNLOADED
        self._last_error: str | None = None
        self._load_time_s: float | None = None
        self._load_count = 0
        self._lock = RLock()

    @property
    def state(self) -> SegmentationState:
        return self._state

    @property
    def load_count(self) -> int:
        return self._load_count

    def status(self) -> dict:
        with self._lock:
            return {
                "model": self.settings.segmentation_model_id,
                "task": "bbox-prompt instance segmentation",
                "state": self._state.value,
                "device": self.settings.segmentation_device,
                "model_path": str(self.settings.segmentation_model_path),
                "image_size": self.settings.segmentation_imgsz,
                "offline": True,
                "load_time_s": self._load_time_s,
                "load_count": self._load_count,
                "last_error": self._last_error,
                "gpu_memory": get_detector_gpu_memory().model_dump(),
            }

    def load_model(self) -> dict:
        if not self._lock.acquire(blocking=False):
            raise SegmentationBusyError("SAM operation already in progress")
        try:
            if self._state == SegmentationState.READY:
                return self.status()
            if not self._cuda_available():
                raise SegmentationCudaUnavailableError("CUDA is required for SAM")
            if not self.settings.segmentation_model_path.is_file():
                raise SegmentationFilesMissingError(
                    f"SAM weights not found: {self.settings.segmentation_model_path}"
                )
            self._state = SegmentationState.LOADING
            self._last_error = None
            timer = time.perf_counter()
            try:
                model = self._factory(self.settings.segmentation_model_path)
                if hasattr(model, "to"):
                    model.to(self.settings.segmentation_device)
                self._model = model
                self._load_time_s = round(time.perf_counter() - timer, 3)
                self._load_count += 1
                self._state = SegmentationState.READY
            except Exception as exc:
                self._model = None
                self._state = SegmentationState.ERROR
                self._last_error = str(exc)
                raise SegmentationLoadError("SAM failed to load; see server log") from exc
            return self.status()
        finally:
            self._lock.release()

    def unload_model(self) -> dict:
        if not self._lock.acquire(blocking=False):
            raise SegmentationBusyError("SAM operation already in progress")
        try:
            if self._model is not None and hasattr(self._model, "to"):
                self._model.to("cpu")
            self._model = None
            self._state = SegmentationState.UNLOADED
            self._last_error = None
            self._load_time_s = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return self.status()
        finally:
            self._lock.release()

    def predict(self, image_path: Path, boxes: list[BoundingBox]) -> SegmentationRun:
        if self._state == SegmentationState.UNLOADED:
            self.load_model()
        if not self._lock.acquire(blocking=False):
            raise SegmentationBusyError("SAM operation already in progress")
        try:
            if self._state != SegmentationState.READY or self._model is None:
                raise SegmentationLoadError("SAM is not ready")
            try:
                with Image.open(image_path) as opened:
                    source_image = opened.convert("RGB")
                    width, height = source_image.size
                    source = np.ascontiguousarray(source_image)
                for box in boxes:
                    if box.x2 > width or box.y2 > height:
                        raise InvalidSegmentationPromptError(
                            "Segmentation bbox exceeds image bounds"
                        )
                prompts = [[box.x1, box.y1, box.x2, box.y2] for box in boxes]
                timer = time.perf_counter()
                results = self._model.predict(
                    source=source,
                    bboxes=prompts,
                    device=self.settings.segmentation_device,
                    imgsz=self.settings.segmentation_imgsz,
                    verbose=False,
                )
                if not results or results[0].masks is None:
                    raise RuntimeError("SAM returned no masks")
                result = results[0]
                raw_masks = _mask_array(result.masks.data)
                if raw_masks.ndim == 2:
                    raw_masks = raw_masks[None, ...]
                if len(raw_masks) != len(boxes):
                    raise RuntimeError("SAM mask count does not match bbox prompt count")
                masks: list[np.ndarray] = []
                instances: list[SegmentInstance] = []
                for index, (box, raw_mask) in enumerate(zip(boxes, raw_masks), start=1):
                    mask = raw_mask > 0.5
                    if mask.shape != (height, width):
                        resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
                            (width, height), Image.Resampling.NEAREST
                        )
                        mask = np.asarray(resized) > 0
                    mask = np.ascontiguousarray(mask)
                    area = int(mask.sum())
                    masks.append(mask)
                    instances.append(SegmentInstance(
                        segment_id=f"segment-{index:03d}",
                        bbox=box,
                        mask_area_pixels=area,
                        mask_area_ratio=round(area / (width * height), 8),
                    ))
                prediction = SegmentationPrediction(
                    image_width=width,
                    image_height=height,
                    segment_count=len(instances),
                    segments=instances,
                )
                wall_ms = round((time.perf_counter() - timer) * 1000, 2)
                inference_ms = round(float(getattr(result, "speed", {}).get("inference", wall_ms)), 2)
                return SegmentationRun(
                    model=self.settings.segmentation_model_id,
                    device=self.settings.segmentation_device,
                    load_time_s=self._load_time_s,
                    inference_ms=inference_ms,
                    prediction=prediction,
                    masks=masks,
                    gpu=get_detector_gpu_memory(),
                )
            except InvalidSegmentationPromptError:
                raise
            except Exception as exc:
                self.logger.exception("SAM inference failed")
                raise SegmentationInferenceError("SAM inference failed; see server log") from exc
        finally:
            self._lock.release()

from __future__ import annotations

from collections import Counter
from enum import StrEnum
import logging
from pathlib import Path
from threading import RLock
import time
from typing import Any, Callable

import numpy as np
from PIL import Image
import torch

from backend.app.core.config import Settings
from backend.app.detection.errors import (
    DetectorBusyError,
    DetectorCudaUnavailableError,
    DetectorFilesMissingError,
    DetectorInferenceError,
    DetectorLoadError,
    UnsupportedDetectionClassError,
)
from backend.app.detection.schemas import BoundingBox, Detection, DetectionPrediction, DetectorRun
from backend.app.schemas.inference import GpuMemory


class DetectorState(StrEnum):
    UNLOADED = "UNLOADED"
    LOADING = "LOADING"
    READY = "READY"
    ERROR = "ERROR"


def get_detector_gpu_memory() -> GpuMemory:
    if not torch.cuda.is_available():
        return GpuMemory(allocated_gb=0, reserved_gb=0, peak_allocated_gb=0)
    divisor = 1024 ** 3
    return GpuMemory(
        allocated_gb=round(torch.cuda.memory_allocated(0) / divisor, 3),
        reserved_gb=round(torch.cuda.memory_reserved(0) / divisor, 3),
        peak_allocated_gb=round(torch.cuda.max_memory_allocated(0) / divisor, 3),
    )


def _default_factory(path: Path):
    from ultralytics import YOLO

    return YOLO(str(path), task="detect")


def _as_list(value: Any) -> list:
    for method in ("detach", "cpu"):
        if hasattr(value, method):
            value = getattr(value, method)()
    return value.tolist() if hasattr(value, "tolist") else list(value)


class DetectorManager:
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
        self._state = DetectorState.UNLOADED
        self._last_error: str | None = None
        self._load_time_s: float | None = None
        self._load_count = 0
        self._lock = RLock()

    @property
    def state(self) -> DetectorState:
        return self._state

    @property
    def load_count(self) -> int:
        return self._load_count

    def status(self) -> dict:
        with self._lock:
            return {
                "model": self.settings.detector_model_id,
                "task": "closed-set COCO object detection",
                "state": self._state.value,
                "device": self.settings.detector_device,
                "model_path": str(self.settings.detector_model_path),
                "image_size": self.settings.detector_imgsz,
                "load_time_s": self._load_time_s,
                "load_count": self._load_count,
                "last_error": self._last_error,
                "gpu_memory": get_detector_gpu_memory().model_dump(),
            }

    def load_model(self) -> dict:
        if not self._lock.acquire(blocking=False):
            raise DetectorBusyError("Detector operation already in progress")
        try:
            if self._state == DetectorState.READY:
                return self.status()
            if self._state == DetectorState.LOADING:
                raise DetectorBusyError("Detector is already loading")
            if not self._cuda_available():
                raise DetectorCudaUnavailableError("CUDA is required for the detector")
            if not self.settings.detector_model_path.is_file():
                raise DetectorFilesMissingError(
                    f"Detector weights not found: {self.settings.detector_model_path}"
                )
            self._state = DetectorState.LOADING
            self._last_error = None
            timer = time.perf_counter()
            try:
                model = self._factory(self.settings.detector_model_path)
                if hasattr(model, "to"):
                    model.to(self.settings.detector_device)
                self._model = model
                self._load_time_s = round(time.perf_counter() - timer, 3)
                self._load_count += 1
                self._state = DetectorState.READY
                self.logger.info(
                    "Detector %s loaded on %s in %.3fs",
                    self.settings.detector_model_id,
                    self.settings.detector_device,
                    self._load_time_s,
                )
            except (DetectorCudaUnavailableError, DetectorFilesMissingError):
                raise
            except Exception as exc:
                self._model = None
                self._state = DetectorState.ERROR
                self._last_error = str(exc)
                raise DetectorLoadError("Detector failed to load; see server log") from exc
            return self.status()
        finally:
            self._lock.release()

    def unload_model(self) -> dict:
        if not self._lock.acquire(blocking=False):
            raise DetectorBusyError("Detector operation already in progress")
        try:
            if self._model is not None and hasattr(self._model, "to"):
                self._model.to("cpu")
            self._model = None
            self._state = DetectorState.UNLOADED
            self._last_error = None
            self._load_time_s = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return self.status()
        finally:
            self._lock.release()

    def _resolve_classes(self, values: list[int | str] | None) -> list[int] | None:
        if values is None:
            return None
        names = getattr(self._model, "names", {})
        if isinstance(names, list):
            names = dict(enumerate(names))
        normalized = {str(name).casefold(): int(class_id) for class_id, name in names.items()}
        resolved = []
        for value in values:
            if isinstance(value, int):
                class_id = value
                if class_id not in names:
                    raise UnsupportedDetectionClassError(
                        f"Class id {class_id} is not supported by the COCO detector"
                    )
            else:
                class_id = normalized.get(value.strip().casefold())
                if class_id is None:
                    raise UnsupportedDetectionClassError(
                        f"Class '{value}' is not supported by the COCO detector"
                    )
            if class_id not in resolved:
                resolved.append(class_id)
        return resolved

    def predict(
        self,
        image_path: Path,
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        classes: list[int | str] | None = None,
    ) -> DetectorRun:
        if self._state == DetectorState.UNLOADED:
            self.load_model()
        if not self._lock.acquire(blocking=False):
            raise DetectorBusyError("Detector operation already in progress")
        try:
            if self._state != DetectorState.READY or self._model is None:
                raise DetectorLoadError("Detector is not ready")
            class_ids = self._resolve_classes(classes)
            timer = time.perf_counter()
            try:
                with Image.open(image_path) as opened:
                    source = np.ascontiguousarray(opened.convert("RGB"))
                results = self._model.predict(
                    source=source,
                    conf=confidence,
                    iou=iou_threshold,
                    classes=class_ids,
                    device=self.settings.detector_device,
                    imgsz=self.settings.detector_imgsz,
                    verbose=False,
                )
                if not results:
                    raise RuntimeError("Detector returned no result object")
                result = results[0]
                height, width = map(int, result.orig_shape)
                names = result.names
                if isinstance(names, list):
                    names = dict(enumerate(names))
                xyxy = _as_list(result.boxes.xyxy) if result.boxes is not None else []
                confidences = _as_list(result.boxes.conf) if result.boxes is not None else []
                class_values = _as_list(result.boxes.cls) if result.boxes is not None else []
                detections = []
                for coordinates, score, raw_class in zip(xyxy, confidences, class_values):
                    class_id = int(raw_class)
                    x1, y1, x2, y2 = (float(value) for value in coordinates)
                    detection = Detection(
                        class_id=class_id,
                        class_name=str(names[class_id]),
                        confidence=round(float(score), 6),
                        bbox=BoundingBox(
                            x1=round(max(0.0, min(x1, width)), 2),
                            y1=round(max(0.0, min(y1, height)), 2),
                            x2=round(max(0.0, min(x2, width)), 2),
                            y2=round(max(0.0, min(y2, height)), 2),
                        ),
                    )
                    detections.append(detection)
                counts = dict(sorted(Counter(item.class_name for item in detections).items()))
                prediction = DetectionPrediction(
                    image_width=width,
                    image_height=height,
                    detection_count=len(detections),
                    class_counts=counts,
                    detections=detections,
                )
                wall_ms = round((time.perf_counter() - timer) * 1000, 2)
                inference_ms = round(float(getattr(result, "speed", {}).get("inference", wall_ms)), 2)
                return DetectorRun(
                    model=self.settings.detector_model_id,
                    device=self.settings.detector_device,
                    load_time_s=self._load_time_s,
                    inference_ms=inference_ms,
                    prediction=prediction,
                    gpu=get_detector_gpu_memory(),
                )
            except UnsupportedDetectionClassError:
                raise
            except Exception as exc:
                self.logger.exception("Detector inference failed")
                raise DetectorInferenceError("Detector inference failed; see server log") from exc
        finally:
            self._lock.release()

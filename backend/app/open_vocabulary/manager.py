from __future__ import annotations

from collections import Counter
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
from backend.app.detection.manager import _as_list, get_detector_gpu_memory
from backend.app.detection.schemas import BoundingBox
from backend.app.open_vocabulary.errors import (
    InvalidOpenVocabularyClassesError,
    OpenVocabularyError,
    OpenVocabularyBusyError,
    OpenVocabularyCudaUnavailableError,
    OpenVocabularyFilesMissingError,
    OpenVocabularyInferenceError,
    OpenVocabularyLoadError,
)
from backend.app.open_vocabulary.schemas import (
    OpenVocabularyDetection,
    OpenVocabularyPrediction,
    OpenVocabularyRun,
)


class OpenVocabularyState(StrEnum):
    UNLOADED = "UNLOADED"
    LOADING = "LOADING"
    READY = "READY"
    ERROR = "ERROR"


def _default_factory(path: Path):
    from ultralytics import YOLOE

    return YOLOE(str(path), task="segment")


def _normalize_classes(values: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        label = value.strip()
        if not label:
            raise InvalidOpenVocabularyClassesError("Open-vocabulary classes cannot be blank")
        key = label.casefold()
        if key not in seen:
            normalized.append(label)
            seen.add(key)
    if not normalized:
        raise InvalidOpenVocabularyClassesError("At least one target class is required")
    return normalized


def _visual_prompt_variants(classes: list[str]) -> tuple[list[str], dict[str, str]]:
    """Build generic visual phrasings and retain stable user-facing class names."""
    prompts: list[str] = []
    canonical_by_prompt: dict[str, str] = {}
    for canonical in classes:
        words = canonical.split()
        if words and words[0].casefold() in {"a", "an", "the"}:
            base = " ".join(words[1:]).strip() or canonical
        else:
            base = canonical
        for prompt in (canonical, f"a visible {base}", f"a photo of a {base}"):
            key = prompt.casefold()
            if key not in canonical_by_prompt:
                prompts.append(prompt)
                canonical_by_prompt[key] = canonical
    return prompts, canonical_by_prompt


class OpenVocabularyDetectorManager:
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
        self._state = OpenVocabularyState.UNLOADED
        self._last_error: str | None = None
        self._load_time_s: float | None = None
        self._load_count = 0
        self._prompt_count = 0
        self._active_classes: tuple[str, ...] | None = None
        self._active_prompts: tuple[str, ...] | None = None
        self._lock = RLock()

    @property
    def state(self) -> OpenVocabularyState:
        return self._state

    @property
    def load_count(self) -> int:
        return self._load_count

    def status(self) -> dict:
        with self._lock:
            return {
                "model": self.settings.open_vocab_model_id,
                "task": "open-vocabulary text-prompt detection",
                "state": self._state.value,
                "device": self.settings.open_vocab_device,
                "model_path": str(self.settings.open_vocab_model_path),
                "text_encoder_path": str(self.settings.open_vocab_text_encoder_path),
                "image_size": self.settings.open_vocab_imgsz,
                "offline": True,
                "load_time_s": self._load_time_s,
                "load_count": self._load_count,
                "prompt_count": self._prompt_count,
                "active_classes": list(self._active_classes or ()),
                "active_prompts": list(self._active_prompts or ()),
                "last_error": self._last_error,
                "gpu_memory": get_detector_gpu_memory().model_dump(),
            }

    def _validate_runtime(self) -> None:
        if not self._cuda_available():
            raise OpenVocabularyCudaUnavailableError("CUDA is required for YOLOE")
        missing = [
            path for path in (
                self.settings.open_vocab_model_path,
                self.settings.open_vocab_text_encoder_path,
            ) if not path.is_file()
        ]
        if missing:
            raise OpenVocabularyFilesMissingError(
                "Open-vocabulary assets not found: " + ", ".join(str(path) for path in missing)
            )
        try:
            import clip  # noqa: F401
        except ImportError as exc:
            raise OpenVocabularyFilesMissingError(
                "The pinned CLIP tokenizer dependency is not installed"
            ) from exc

    def load_model(self) -> dict:
        if not self._lock.acquire(blocking=False):
            raise OpenVocabularyBusyError("YOLOE operation already in progress")
        try:
            if self._state == OpenVocabularyState.READY:
                return self.status()
            self._validate_runtime()
            self._state = OpenVocabularyState.LOADING
            self._last_error = None
            timer = time.perf_counter()
            try:
                model = self._factory(self.settings.open_vocab_model_path)
                if hasattr(model, "to"):
                    model.to(self.settings.open_vocab_device)
                self._model = model
                self._load_time_s = round(time.perf_counter() - timer, 3)
                self._load_count += 1
                self._active_classes = None
                self._active_prompts = None
                self._state = OpenVocabularyState.READY
            except Exception as exc:
                self._model = None
                self._state = OpenVocabularyState.ERROR
                self._last_error = str(exc)
                raise OpenVocabularyLoadError("YOLOE failed to load; see server log") from exc
            return self.status()
        finally:
            self._lock.release()

    def unload_model(self) -> dict:
        if not self._lock.acquire(blocking=False):
            raise OpenVocabularyBusyError("YOLOE operation already in progress")
        try:
            if self._model is not None and hasattr(self._model, "to"):
                self._model.to("cpu")
            self._model = None
            self._active_classes = None
            self._active_prompts = None
            self._state = OpenVocabularyState.UNLOADED
            self._last_error = None
            self._load_time_s = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return self.status()
        finally:
            self._lock.release()

    def predict(
        self,
        image_path: Path,
        classes: list[str],
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> OpenVocabularyRun:
        requested = _normalize_classes(classes)
        effective_prompts, canonical_by_prompt = _visual_prompt_variants(requested)
        if self._state == OpenVocabularyState.UNLOADED:
            self.load_model()
        if not self._lock.acquire(blocking=False):
            raise OpenVocabularyBusyError("YOLOE operation already in progress")
        try:
            if self._state != OpenVocabularyState.READY or self._model is None:
                raise OpenVocabularyLoadError("YOLOE is not ready")
            try:
                prompt_timer = time.perf_counter()
                signature = tuple(effective_prompts)
                if signature != self._active_prompts:
                    self._model.set_classes(effective_prompts)
                    self._active_classes = tuple(requested)
                    self._active_prompts = signature
                    self._prompt_count += 1
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                prompt_ms = round((time.perf_counter() - prompt_timer) * 1000, 2)
                with Image.open(image_path) as opened:
                    source = np.ascontiguousarray(opened.convert("RGB"))
                timer = time.perf_counter()
                results = self._model.predict(
                    source=source,
                    conf=confidence,
                    iou=iou_threshold,
                    device=self.settings.open_vocab_device,
                    imgsz=self.settings.open_vocab_imgsz,
                    agnostic_nms=True,
                    verbose=False,
                )
                if not results:
                    raise RuntimeError("YOLOE returned no result object")
                result = results[0]
                height, width = map(int, result.orig_shape)
                names = result.names
                if isinstance(names, list):
                    names = dict(enumerate(names))
                xyxy = _as_list(result.boxes.xyxy) if result.boxes is not None else []
                scores = _as_list(result.boxes.conf) if result.boxes is not None else []
                class_values = _as_list(result.boxes.cls) if result.boxes is not None else []
                detections = []
                for index, (coordinates, score, raw_class) in enumerate(
                    zip(xyxy, scores, class_values), start=1
                ):
                    class_id = int(raw_class)
                    x1, y1, x2, y2 = (float(value) for value in coordinates)
                    reported_name = str(names[class_id])
                    detections.append(OpenVocabularyDetection(
                        detection_id=f"detection-{index:03d}",
                        class_name=canonical_by_prompt.get(
                            reported_name.casefold(), reported_name
                        ),
                        confidence=round(float(score), 6),
                        bbox=BoundingBox(
                            x1=round(max(0.0, min(x1, width)), 2),
                            y1=round(max(0.0, min(y1, height)), 2),
                            x2=round(max(0.0, min(x2, width)), 2),
                            y2=round(max(0.0, min(y2, height)), 2),
                        ),
                    ))
                counts = dict(sorted(Counter(item.class_name for item in detections).items()))
                prediction = OpenVocabularyPrediction(
                    image_width=width,
                    image_height=height,
                    requested_classes=requested,
                    detection_count=len(detections),
                    class_counts=counts,
                    detections=detections,
                )
                wall_ms = round((time.perf_counter() - timer) * 1000, 2)
                inference_ms = round(float(getattr(result, "speed", {}).get("inference", wall_ms)), 2)
                return OpenVocabularyRun(
                    model=self.settings.open_vocab_model_id,
                    text_encoder=self.settings.open_vocab_text_encoder_path.name,
                    device=self.settings.open_vocab_device,
                    load_time_s=self._load_time_s,
                    effective_prompts=effective_prompts,
                    prompt_encoding_ms=prompt_ms,
                    inference_ms=inference_ms,
                    prediction=prediction,
                    gpu=get_detector_gpu_memory(),
                )
            except (InvalidOpenVocabularyClassesError, OpenVocabularyError):
                raise
            except Exception as exc:
                self.logger.exception("YOLOE inference failed")
                raise OpenVocabularyInferenceError("YOLOE inference failed; see server log") from exc
        finally:
            self._lock.release()

from __future__ import annotations

from enum import StrEnum
import logging
from threading import RLock
from typing import Callable

from PIL import Image

from backend.app.core.config import Settings
from backend.app.models.errors import ModelBusyError, ModelLoadError
from backend.app.models.qwen_vl import QwenVlModel, get_gpu_memory
from backend.app.schemas.inference import InferenceResult, TextGenerationResult

logger = logging.getLogger("geoagent")


class ModelState(StrEnum):
    UNLOADED = "UNLOADED"
    LOADING = "LOADING"
    READY = "READY"
    ERROR = "ERROR"


class ModelManager:
    def __init__(
        self, settings: Settings, factory: Callable[[Settings], QwenVlModel] = QwenVlModel
    ):
        self.settings = settings
        self._factory = factory
        self._wrapper: QwenVlModel | None = None
        self._state = ModelState.UNLOADED
        self._last_error: str | None = None
        self._load_time_s: float | None = None
        self._load_count = 0
        self._lock = RLock()

    @property
    def state(self) -> ModelState:
        return self._state

    def status(self) -> dict:
        with self._lock:
            return {
                "model": "Qwen3-VL-4B-Instruct",
                "model_id": self.settings.vlm_model_id,
                "state": self._state.value,
                "device": self.settings.vlm_device,
                "dtype": self.settings.vlm_dtype,
                "attention": self.settings.vlm_attention,
                "model_path": str(self.settings.vlm_model_path),
                "local_files_only": self.settings.hf_hub_offline,
                "load_time_s": self._load_time_s,
                "load_count": self._load_count,
                "last_error": self._last_error,
                "gpu_memory": get_gpu_memory().model_dump(),
            }

    def load_model(self) -> dict:
        if not self._lock.acquire(blocking=False):
            raise ModelBusyError("Model operation already in progress")
        try:
            if self._state == ModelState.LOADING:
                raise ModelBusyError("Model is already loading")
            if self._state == ModelState.READY:
                return self.status()
            self._state = ModelState.LOADING
            self._last_error = None
            wrapper = self._factory(self.settings)
            try:
                self._load_time_s = round(wrapper.load(), 3)
                self._wrapper = wrapper
                self._load_count += 1
                self._state = ModelState.READY
            except Exception as exc:
                self._wrapper = None
                self._state = ModelState.ERROR
                self._last_error = str(exc)
                raise
            return self.status()
        finally:
            self._lock.release()

    def unload_model(self) -> dict:
        if not self._lock.acquire(blocking=False):
            raise ModelBusyError("Model operation already in progress")
        try:
            if self._state == ModelState.LOADING:
                raise ModelBusyError("Model is loading")
            if self._wrapper is not None:
                self._wrapper.unload()
            self._wrapper = None
            self._state = ModelState.UNLOADED
            self._last_error = None
            self._load_time_s = None
            logger.info("Qwen3-VL unloaded; GPU allocated: %.3f GiB", get_gpu_memory().allocated_gb)
            return self.status()
        finally:
            self._lock.release()

    def infer(self, image: Image.Image, prompt: str, max_new_tokens: int) -> InferenceResult:
        if not self._lock.acquire(blocking=False):
            raise ModelBusyError("Model operation already in progress")
        try:
            if self._state == ModelState.LOADING:
                raise ModelBusyError("Model is loading")
            if self._state != ModelState.READY or self._wrapper is None:
                raise ModelLoadError("Model is not loaded; call the load endpoint first")
            return self._wrapper.generate(image, prompt, max_new_tokens)
        finally:
            self._lock.release()

    def plan(
        self, prompt: str, max_new_tokens: int, system_prompt: str | None = None
    ) -> TextGenerationResult:
        if not self._lock.acquire(blocking=False):
            raise ModelBusyError("Model operation already in progress")
        try:
            if self._state == ModelState.LOADING:
                raise ModelBusyError("Model is loading")
            if self._state != ModelState.READY or self._wrapper is None:
                raise ModelLoadError("Model is not loaded; agent cannot plan")
            return self._wrapper.generate_text(prompt, max_new_tokens, system_prompt)
        finally:
            self._lock.release()

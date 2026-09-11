"""Qwen3-VL wrapper; heavy libraries are imported only during explicit load."""
from __future__ import annotations

import gc
import logging
from pathlib import Path
import time
from typing import Any

from PIL import Image
import torch

from backend.app.core.config import Settings
from backend.app.models.errors import (
    CudaOutOfMemoryError,
    CudaUnavailableError,
    InferenceFailedError,
    ModelFilesMissingError,
    ModelLoadError,
)
from backend.app.schemas.inference import GenerationInfo, GpuMemory, ImageInfo, InferenceResult

logger = logging.getLogger("geoagent")
REQUIRED_FILES = ("config.json", "generation_config.json")


def get_gpu_memory() -> GpuMemory:
    if not torch.cuda.is_available():
        return GpuMemory(allocated_gb=0, reserved_gb=0, peak_allocated_gb=0)
    divisor = 1024**3
    return GpuMemory(
        allocated_gb=round(torch.cuda.memory_allocated(0) / divisor, 3),
        reserved_gb=round(torch.cuda.memory_reserved(0) / divisor, 3),
        peak_allocated_gb=round(torch.cuda.max_memory_allocated(0) / divisor, 3),
    )


def validate_model_files(model_path: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (model_path / name).is_file()]
    has_weights = any(model_path.glob("*.safetensors"))
    has_processor = any((model_path / name).is_file() for name in (
        "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json"
    ))
    has_tokenizer = any((model_path / name).is_file() for name in (
        "tokenizer.json", "tokenizer_config.json", "vocab.json"
    ))
    if missing or not has_weights or not has_processor or not has_tokenizer:
        details = missing + (["*.safetensors"] if not has_weights else [])
        details += (["processor config"] if not has_processor else [])
        details += (["tokenizer files"] if not has_tokenizer else [])
        raise ModelFilesMissingError(f"Incomplete local model at {model_path}: {', '.join(details)}")


class QwenVlModel:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model: Any | None = None
        self.processor: Any | None = None

    def load(self) -> float:
        if not torch.cuda.is_available():
            raise CudaUnavailableError("CUDA is required; CPU fallback is disabled")
        if "RTX 4090" not in torch.cuda.get_device_name(0):
            raise CudaUnavailableError("cuda:0 is not the required NVIDIA RTX 4090")
        validate_model_files(self.settings.vlm_model_path)
        logger.info("Loading Qwen3-VL-4B-Instruct")
        logger.info("Model path: %s", self.settings.vlm_model_path)
        logger.info("Device: %s", self.settings.vlm_device)
        logger.info("dtype: %s", self.settings.vlm_dtype)
        started = time.perf_counter()
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                self.settings.vlm_model_path, local_files_only=True
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.settings.vlm_model_path,
                dtype=torch.bfloat16,
                attn_implementation=self.settings.vlm_attention,
                device_map={"": self.settings.vlm_device},
                low_cpu_mem_usage=True,
                local_files_only=True,
            ).eval()
            parameter_device = str(next(self.model.parameters()).device)
            if parameter_device != self.settings.vlm_device:
                raise RuntimeError(f"Unexpected model device: {parameter_device}")
            device_map = getattr(self.model, "hf_device_map", {})
            if any(str(value) not in {"0", self.settings.vlm_device} for value in device_map.values()):
                raise RuntimeError(f"CPU/disk fallback detected: {device_map}")
        except Exception as exc:
            self.unload()
            if isinstance(exc, (CudaUnavailableError, ModelFilesMissingError)):
                raise
            logger.exception("Model loading failed")
            raise ModelLoadError("Qwen3-VL loading failed; see server log") from exc
        elapsed = time.perf_counter() - started
        logger.info("Model loaded in %.2fs", elapsed)
        logger.info("GPU allocated: %.3f GiB", get_gpu_memory().allocated_gb)
        return elapsed

    def unload(self) -> GpuMemory:
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
        return get_gpu_memory()

    def _prepare_image(self, image: Image.Image) -> tuple[Image.Image, ImageInfo]:
        original_width, original_height = image.size
        original_mode = image.mode
        original_format = (image.format or "UNKNOWN").upper()
        working = image.convert("RGB")
        longest = max(working.size)
        resized = longest > self.settings.vlm_max_image_edge
        if resized:
            scale = self.settings.vlm_max_image_edge / longest
            size = (max(1, round(working.width * scale)), max(1, round(working.height * scale)))
            working = working.resize(size, Image.Resampling.LANCZOS)
            logger.info("Image resized from %dx%d to %dx%d", original_width, original_height, *working.size)
        return working, ImageInfo(
            width=original_width,
            height=original_height,
            mode=original_mode,
            format=original_format,
            preprocessing_width=working.width,
            preprocessing_height=working.height,
            resized=resized,
        )

    @torch.inference_mode()
    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> InferenceResult:
        if self.model is None or self.processor is None:
            raise ModelLoadError("Model is not loaded")
        working, image_info = self._prepare_image(image)
        logger.info("Inference started")
        logger.info("Image: %dx%d", image_info.width, image_info.height)
        torch.cuda.reset_peak_memory_stats(0)
        started = time.perf_counter()
        try:
            from qwen_vl_utils import process_vision_info

            messages = [{"role": "user", "content": [
                {"type": "image", "image": working, "max_pixels": self.settings.vlm_max_pixels},
                {"type": "text", "text": prompt},
            ]}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            images, videos = process_vision_info(messages, image_patch_size=16)
            processed_image = images[0]
            image_info.preprocessing_width, image_info.preprocessing_height = processed_image.size
            image_info.resized = image_info.resized or processed_image.size != (
                image_info.width, image_info.height
            )
            inputs = self.processor(
                text=text, images=images, videos=videos, do_resize=False, return_tensors="pt"
            ).to(self.settings.vlm_device)
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            trimmed = [
                output[len(input_ids):] for input_ids, output in zip(inputs.input_ids, generated)
            ]
            output_text = self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            torch.cuda.synchronize(0)
            peak_gb = round(torch.cuda.max_memory_allocated(0) / 1024**3, 3)
            del generated, trimmed, inputs
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            logger.exception("CUDA OOM during inference")
            raise CudaOutOfMemoryError(
                "CUDA memory exhausted; reduce image size or max_new_tokens"
            ) from exc
        except Exception as exc:
            torch.cuda.empty_cache()
            logger.exception("Inference failed")
            raise InferenceFailedError("Qwen3-VL inference failed; see server log") from exc
        latency_ms = (time.perf_counter() - started) * 1000
        memory = get_gpu_memory()
        memory.peak_allocated_gb = peak_gb
        logger.info("Inference completed in %.0f ms", latency_ms)
        logger.info("Peak VRAM: %.3f GiB", memory.peak_allocated_gb)
        return InferenceResult(
            success=True,
            model="Qwen3-VL-4B-Instruct",
            text=output_text,
            latency_ms=round(latency_ms, 2),
            device=self.settings.vlm_device,
            dtype=self.settings.vlm_dtype,
            image=image_info,
            generation=GenerationInfo(max_new_tokens=max_new_tokens),
            gpu=memory,
        )

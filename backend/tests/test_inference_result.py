import pytest
from pydantic import ValidationError

from backend.app.schemas.inference import (
    GenerationInfo,
    GpuMemory,
    ImageInfo,
    InferenceError,
    InferenceResult,
)


def result_data(**overrides):
    data = {
        "success": True,
        "model": "Qwen3-VL-4B-Instruct",
        "text": "A red square.",
        "latency_ms": 250.5,
        "device": "cuda:0",
        "dtype": "bfloat16",
        "image": ImageInfo(
            width=1024, height=768, mode="RGB", format="PNG",
            preprocessing_width=1024, preprocessing_height=768, resized=False
        ),
        "generation": GenerationInfo(max_new_tokens=128),
        "gpu": GpuMemory(allocated_gb=8.5, reserved_gb=9.0, peak_allocated_gb=10.2),
    }
    data.update(overrides)
    return data


def test_inference_result_roundtrip():
    result = InferenceResult(**result_data())
    assert InferenceResult.model_validate_json(result.model_dump_json()) == result


def test_failed_result_requires_structured_error():
    result = InferenceResult(
        **result_data(
            success=False, text="", error=InferenceError(code="OOM", message="out of memory")
        )
    )
    assert result.error.code == "OOM"


def test_success_rejects_error():
    with pytest.raises(ValidationError):
        InferenceResult(**result_data(error=InferenceError(code="X", message="bad")))


def test_token_limits():
    with pytest.raises(ValidationError):
        GenerationInfo(max_new_tokens=513)

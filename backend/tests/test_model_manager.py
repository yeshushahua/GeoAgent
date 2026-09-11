from PIL import Image
import pytest

from backend.app.models.errors import ModelLoadError
from backend.app.models.manager import ModelManager, ModelState
from backend.app.schemas.inference import GenerationInfo, GpuMemory, ImageInfo, InferenceResult


class FakeWrapper:
    def __init__(self, settings, fail=False):
        self.fail = fail
        self.unloaded = False

    def load(self):
        if self.fail:
            raise ModelLoadError("test failure")
        return 1.25

    def unload(self):
        self.unloaded = True
        return GpuMemory(allocated_gb=0, reserved_gb=0, peak_allocated_gb=0)

    def generate(self, image, prompt, max_new_tokens):
        return InferenceResult(
            success=True, model="Qwen3-VL-4B-Instruct", text=f"seen: {prompt}",
            latency_ms=10, device="cuda:0", dtype="bfloat16",
            image=ImageInfo(
                width=image.width, height=image.height, mode=image.mode, format="PNG",
                preprocessing_width=image.width, preprocessing_height=image.height, resized=False
            ),
            generation=GenerationInfo(max_new_tokens=max_new_tokens),
            gpu=GpuMemory(allocated_gb=1, reserved_gb=1, peak_allocated_gb=1),
        )


def test_load_infer_unload_reload(settings):
    manager = ModelManager(settings, factory=FakeWrapper)
    assert manager.state == ModelState.UNLOADED
    assert manager.load_model()["state"] == "READY"
    assert manager.load_model()["state"] == "READY"
    result = manager.infer(Image.new("RGB", (32, 24)), "hello", 64)
    assert result.text == "seen: hello"
    assert manager.unload_model()["state"] == "UNLOADED"
    assert manager.load_model()["state"] == "READY"


def test_infer_requires_loaded_model(settings):
    manager = ModelManager(settings, factory=FakeWrapper)
    with pytest.raises(ModelLoadError, match="not loaded"):
        manager.infer(Image.new("RGB", (16, 16)), "hello", 64)


def test_load_failure_enters_error(settings):
    manager = ModelManager(settings, factory=lambda config: FakeWrapper(config, fail=True))
    with pytest.raises(ModelLoadError):
        manager.load_model()
    assert manager.state == ModelState.ERROR
    assert manager.status()["last_error"] == "test failure"

import io

from PIL import Image

from backend.app.schemas.inference import GenerationInfo, GpuMemory, ImageInfo, InferenceResult
from backend.app.models.errors import CudaOutOfMemoryError


class FakeManager:
    def status(self):
        return {"state": "READY", "gpu_memory": {"allocated_gb": 1}}

    def load_model(self):
        return self.status()

    def unload_model(self):
        return {"state": "UNLOADED", "gpu_memory": {"allocated_gb": 0}}

    def infer(self, image, prompt, max_new_tokens):
        return InferenceResult(
            success=True, model="Qwen3-VL-4B-Instruct", text=prompt,
            latency_ms=12, device="cuda:0", dtype="bfloat16",
            image=ImageInfo(
                width=image.width, height=image.height, mode=image.mode, format=image.format,
                preprocessing_width=image.width, preprocessing_height=image.height, resized=False
            ),
            generation=GenerationInfo(max_new_tokens=max_new_tokens),
            gpu=GpuMemory(allocated_gb=1, reserved_gb=1, peak_allocated_gb=1),
        )


def image_bytes(format="PNG"):
    buffer = io.BytesIO()
    Image.new("RGB", (20, 10), "red").save(buffer, format=format)
    return buffer.getvalue()


def test_model_endpoints(client):
    client.app.state.model_manager = FakeManager()
    assert client.get("/api/v1/models/vlm/status").json()["state"] == "READY"
    assert client.post("/api/v1/models/vlm/load").json()["state"] == "READY"
    assert client.post("/api/v1/models/vlm/unload").json()["state"] == "UNLOADED"
    response = client.post(
        "/api/v1/models/vlm/infer",
        files={"image": ("scene.png", image_bytes(), "image/png")},
        data={"prompt": "describe", "max_new_tokens": "64"},
    )
    assert response.status_code == 200
    assert response.json()["text"] == "describe"


def test_missing_image_is_structured(client):
    response = client.post("/api/v1/models/vlm/infer", data={"prompt": "describe"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_empty_prompt_is_structured(client):
    response = client.post(
        "/api/v1/models/vlm/infer",
        files={"image": ("scene.png", image_bytes(), "image/png")},
        data={"prompt": "   "},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_INPUT"


def test_invalid_image(client):
    response = client.post(
        "/api/v1/models/vlm/infer",
        files={"image": ("bad.png", b"not an image", "image/png")},
        data={"prompt": "describe"},
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


def test_unsupported_bmp(client):
    response = client.post(
        "/api/v1/models/vlm/infer",
        files={"image": ("scene.bmp", image_bytes("BMP"), "image/bmp")},
        data={"prompt": "describe"},
    )
    assert response.status_code == 415
    assert "PNG" in response.json()["error"]["message"]


def test_cuda_oom_is_structured_and_service_survives(client):
    class OomManager(FakeManager):
        def infer(self, image, prompt, max_new_tokens):
            raise CudaOutOfMemoryError("test OOM")

    client.app.state.model_manager = OomManager()
    response = client.post(
        "/api/v1/models/vlm/infer",
        files={"image": ("scene.png", image_bytes(), "image/png")},
        data={"prompt": "describe"},
    )
    assert response.status_code == 507
    assert response.json()["error"]["code"] == "CUDA_OUT_OF_MEMORY"
    assert client.get("/api/v1/models/vlm/status").status_code == 200

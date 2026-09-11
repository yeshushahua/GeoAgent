from PIL import Image
import pytest

from backend.app.core.config import get_settings
from backend.app.models.manager import ModelManager
from backend.app.services.storage import prepare_storage


@pytest.mark.integration
def test_real_qwen3_vl_offline_on_rtx4090():
    settings = get_settings()
    assert settings.hf_hub_offline is True
    prepare_storage(settings)
    manager = ModelManager(settings)
    try:
        status = manager.load_model()
        assert status["state"] == "READY"
        assert status["device"] == "cuda:0"
        assert status["dtype"] == "bfloat16"
        path = settings.project_root / "sample_data" / "images" / "foundation-demo.png"
        with Image.open(path) as opened:
            image = opened.copy()
            image.format = opened.format
        result = manager.infer(image, "Describe this image briefly.", 64)
        assert result.success and result.text
        assert result.device == "cuda:0" and result.dtype == "bfloat16"
    finally:
        manager.unload_model()

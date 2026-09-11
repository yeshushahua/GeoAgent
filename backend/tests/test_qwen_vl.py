from pathlib import Path

from PIL import Image
import pytest

from backend.app.models.errors import ModelFilesMissingError
from backend.app.models.qwen_vl import QwenVlModel, validate_model_files


def test_high_resolution_image_preserves_aspect_ratio(settings):
    wrapper = QwenVlModel(settings)
    image = Image.new("RGBA", (4096, 3072))
    image.format = "PNG"
    resized, info = wrapper._prepare_image(image)
    assert resized.mode == "RGB"
    assert resized.size == (2048, 1536)
    assert info.width == 4096 and info.height == 3072
    assert info.mode == "RGBA" and info.format == "PNG"
    assert info.resized is True


def test_small_image_is_not_edge_resized(settings):
    resized, info = QwenVlModel(settings)._prepare_image(Image.new("RGB", (640, 480)))
    assert resized.size == (640, 480)
    assert info.resized is False


def test_missing_model_files(settings):
    settings.vlm_model_path.mkdir(parents=True)
    with pytest.raises(ModelFilesMissingError, match="Incomplete local model"):
        validate_model_files(settings.vlm_model_path)

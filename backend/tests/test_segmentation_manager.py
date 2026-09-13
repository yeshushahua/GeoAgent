import numpy as np
from PIL import Image
import pytest

from backend.app.detection.schemas import BoundingBox
from backend.app.segmentation.errors import (
    InvalidSegmentationPromptError,
    SegmentationFilesMissingError,
)
from backend.app.segmentation.manager import SegmentationManager, SegmentationState


class FakeMasks:
    def __init__(self, data):
        self.data = data


class FakeResult:
    def __init__(self, masks):
        self.masks = FakeMasks(masks)
        self.speed = {"inference": 18.5}


class FakeSAM:
    def __init__(self, masks):
        self.masks = masks
        self.to_calls = []
        self.predict_calls = []

    def to(self, device):
        self.to_calls.append(device)
        return self

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        return [FakeResult(self.masks)]


def prepare(settings):
    settings.segmentation_model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.segmentation_model_path.write_bytes(b"weights")
    image = settings.project_root / "segment.png"
    Image.new("RGB", (10, 8), "navy").save(image)
    masks = np.zeros((2, 8, 10), dtype=np.float32)
    masks[0, 1:5, 2:7] = 1
    masks[1, 4:8, 6:10] = 1
    model = FakeSAM(masks)
    manager = SegmentationManager(
        settings, factory=lambda path: model, cuda_available=lambda: True
    )
    return manager, model, image


def test_sam_lazy_load_reuse_mask_area_and_bbox_prompt(settings):
    manager, model, image = prepare(settings)
    boxes = [
        BoundingBox(x1=2, y1=1, x2=7, y2=5),
        BoundingBox(x1=6, y1=4, x2=10, y2=8),
    ]
    assert manager.state == SegmentationState.UNLOADED
    run = manager.predict(image, boxes)
    manager.predict(image, boxes)
    assert manager.load_count == 1
    assert run.prediction.segment_count == 2
    assert run.prediction.segments[0].mask_area_pixels == 20
    assert run.prediction.segments[0].mask_area_ratio == 0.25
    assert run.prediction.segments[1].mask_area_pixels == 16
    assert model.predict_calls[0]["bboxes"] == [[2.0, 1.0, 7.0, 5.0], [6.0, 4.0, 10.0, 8.0]]
    assert model.predict_calls[0]["device"] == "cuda:0"


def test_sam_rejects_out_of_bounds_and_missing_weights(settings):
    manager, _, image = prepare(settings)
    with pytest.raises(InvalidSegmentationPromptError, match="exceeds"):
        manager.predict(image, [BoundingBox(x1=1, y1=1, x2=11, y2=7)])
    settings.segmentation_model_path.unlink()
    missing = SegmentationManager(settings, cuda_available=lambda: True)
    with pytest.raises(SegmentationFilesMissingError):
        missing.load_model()


def test_sam_unload(settings):
    manager, model, _ = prepare(settings)
    manager.load_model()
    assert manager.unload_model()["state"] == "UNLOADED"
    assert model.to_calls[-1] == "cpu"

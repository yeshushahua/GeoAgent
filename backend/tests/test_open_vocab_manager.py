from pathlib import Path

import pytest
from PIL import Image

from backend.app.open_vocabulary.errors import OpenVocabularyFilesMissingError
from backend.app.open_vocabulary.manager import OpenVocabularyDetectorManager, OpenVocabularyState


class FakeBoxes:
    def __init__(self, xyxy=None, confidence=None, classes=None):
        self.xyxy = xyxy or []
        self.conf = confidence or []
        self.cls = classes or []


class FakeResult:
    def __init__(self, boxes=None):
        self.orig_shape = (60, 100)
        self.names = {0: "yellow safety helmet", 1: "tower crane"}
        self.boxes = boxes or FakeBoxes()
        self.speed = {"inference": 7.25}


class FakeYOLOE:
    def __init__(self, result=None):
        self.result = result or FakeResult()
        self.to_calls = []
        self.class_calls = []
        self.predict_calls = []

    def to(self, device):
        self.to_calls.append(device)
        return self

    def set_classes(self, classes):
        self.class_calls.append(list(classes))

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        return [self.result]


def prepare(settings, detections=None):
    settings.open_vocab_model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.open_vocab_model_path.write_bytes(b"weights")
    settings.open_vocab_text_encoder_path.write_bytes(b"encoder")
    image = settings.project_root / "open.png"
    Image.new("RGB", (100, 60), "yellow").save(image)
    model = FakeYOLOE(detections)
    manager = OpenVocabularyDetectorManager(
        settings, factory=lambda path: model, cuda_available=lambda: True
    )
    return manager, model, image


def test_yoloe_lazy_load_reuse_arbitrary_classes_and_structure(settings):
    result = FakeResult(FakeBoxes(
        xyxy=[[-2, 3, 40, 58], [50, 10, 99, 59]],
        confidence=[0.92, 0.73], classes=[0, 1],
    ))
    manager, model, image = prepare(settings, result)
    assert manager.state == OpenVocabularyState.UNLOADED
    first = manager.predict(image, ["yellow safety helmet", "tower crane"])
    second = manager.predict(image, ["yellow safety helmet", "tower crane"])
    assert manager.state == OpenVocabularyState.READY
    assert manager.load_count == 1
    assert manager.status()["prompt_count"] == 1
    assert model.class_calls == [["yellow safety helmet", "tower crane"]]
    assert first.prediction.detection_count == 2
    assert first.prediction.detections[0].detection_id == "detection-001"
    assert first.prediction.detections[0].bbox.x1 == 0
    assert first.prediction.class_counts == {"tower crane": 1, "yellow safety helmet": 1}
    assert second.prediction.requested_classes == ["yellow safety helmet", "tower crane"]
    assert all(0 <= item.confidence <= 1 for item in first.prediction.detections)
    assert model.predict_calls[0]["device"] == "cuda:0"


def test_yoloe_zero_detections_and_class_change(settings):
    manager, model, image = prepare(settings)
    run = manager.predict(image, ["purple spacecraft component"])
    assert run.prediction.detection_count == 0
    assert run.prediction.detections == []
    manager.predict(image, ["another arbitrary phrase"])
    assert len(model.class_calls) == 2
    assert manager.status()["prompt_count"] == 2


def test_yoloe_missing_assets_and_unload(settings):
    manager = OpenVocabularyDetectorManager(
        settings, factory=lambda path: FakeYOLOE(), cuda_available=lambda: True
    )
    with pytest.raises(OpenVocabularyFilesMissingError):
        manager.load_model()
    manager, model, _ = prepare(settings)
    manager.load_model()
    status = manager.unload_model()
    assert status["state"] == "UNLOADED"
    assert model.to_calls[-1] == "cpu"

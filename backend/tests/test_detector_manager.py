from pathlib import Path

import pytest
from PIL import Image

from backend.app.detection.errors import DetectorFilesMissingError, UnsupportedDetectionClassError
from backend.app.detection.manager import DetectorManager, DetectorState


class FakeBoxes:
    def __init__(self, xyxy=None, confidence=None, classes=None):
        self.xyxy = xyxy or []
        self.conf = confidence or []
        self.cls = classes or []


class FakeResult:
    def __init__(self, boxes=None):
        self.orig_shape = (300, 480)
        self.names = {0: "person", 1: "bicycle", 5: "bus"}
        self.boxes = boxes if boxes is not None else FakeBoxes()
        self.speed = {"inference": 7.25}


class FakeYolo:
    names = {0: "person", 1: "bicycle", 5: "bus"}

    def __init__(self, result):
        self.result = result
        self.predict_calls = []
        self.devices = []

    def to(self, device):
        self.devices.append(device)
        return self

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        return [self.result]


def configured(settings):
    path = settings.detector_model_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-yolo-weights")
    return path


def image(settings, name):
    path = settings.project_root / name
    Image.new("RGB", (480, 300), "white").save(path, "JPEG")
    return path


def test_detector_lazy_load_reuse_structure_and_class_filter(settings):
    configured(settings)
    result = FakeResult(FakeBoxes(
        xyxy=[[10.2, 20.4, 110.8, 220.9], [200, 40, 430, 290]],
        confidence=[0.9234567, 0.812345],
        classes=[0, 5],
    ))
    model = FakeYolo(result)
    factory_calls = []

    def factory(path: Path):
        factory_calls.append(path)
        return model

    manager = DetectorManager(settings, factory=factory, cuda_available=lambda: True)
    assert manager.state == DetectorState.UNLOADED
    first = manager.predict(image(settings, "one.jpg"), classes=["person", 5])
    second = manager.predict(image(settings, "two.jpg"))

    assert factory_calls == [settings.detector_model_path]
    assert manager.load_count == 1 and len(model.predict_calls) == 2
    assert model.devices == ["cuda:0"]
    assert model.predict_calls[0]["classes"] == [0, 5]
    assert first.prediction.detection_count == 2
    assert first.prediction.class_counts == {"bus": 1, "person": 1}
    assert first.prediction.detections[0].confidence == 0.923457
    assert first.prediction.detections[0].bbox.x2 <= first.prediction.image_width
    assert second.load_time_s == first.load_time_s

    unloaded = manager.unload_model()
    assert unloaded["state"] == "UNLOADED" and model.devices[-1] == "cpu"


def test_detector_empty_result_and_unknown_class(settings):
    configured(settings)
    model = FakeYolo(FakeResult())
    manager = DetectorManager(
        settings, factory=lambda path: model, cuda_available=lambda: True
    )
    empty_image = image(settings, "empty.jpg")
    empty = manager.predict(empty_image)
    assert empty.prediction.detection_count == 0
    assert empty.prediction.detections == []
    assert empty.prediction.class_counts == {}
    with pytest.raises(UnsupportedDetectionClassError, match="not supported"):
        manager.predict(empty_image, classes=["spaceship"])


def test_detector_missing_weights_fails_closed(settings):
    manager = DetectorManager(
        settings, factory=lambda path: FakeYolo(FakeResult()), cuda_available=lambda: True
    )
    with pytest.raises(DetectorFilesMissingError):
        manager.load_model()
    assert manager.load_count == 0

"""Isolated filesystem tests never require an actual GPU or asset drive."""
import tempfile

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app


@pytest.fixture
def settings(tmp_path, monkeypatch):
    # Construct a trusted fixture on one temporary volume; production validation
    # is tested separately, without requiring two drives on CI runners.
    project, storage = tmp_path / "project", tmp_path / "storage"
    project.mkdir()
    config = Settings.model_construct(
        project_root=project, storage_root=storage,
        model_dir=storage / "models", hf_home=storage / "cache" / "huggingface",
        dataset_dir=storage / "datasets", output_dir=storage / "outputs",
        checkpoint_dir=storage / "checkpoints", temp_dir=storage / "temp",
    )
    # Restore process-global cache/temp configuration after each test.
    for key in ("HF_HUB_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "HF_HOME", "HF_HUB_CACHE", "HF_ASSETS_CACHE", "HF_DATASETS_CACHE",
                "GRADIO_TEMP_DIR", "GRADIO_ANALYTICS_ENABLED", "TMP", "TEMP", "TMPDIR"):
        monkeypatch.setenv(key, "test-original")
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    return config


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as client:
        yield client

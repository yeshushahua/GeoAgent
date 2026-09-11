import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.app.core.config import Settings


def valid_data(tmp_path):
    # Real Windows path parsing with a mocked project existence check only.
    project = Path("C:/geoagent-project") if os.name == "nt" else tmp_path / "project"
    storage = Path("E:/geoagent-assets") if os.name == "nt" else tmp_path / "storage"
    return dict(project_root=project, storage_root=storage,
                model_dir=storage / "models", hf_home=storage / "cache" / "huggingface",
                dataset_dir=storage / "datasets", output_dir=storage / "outputs",
                checkpoint_dir=storage / "checkpoints", temp_dir=storage / "temp")


def test_config_parses_env(tmp_path, monkeypatch):
    data = valid_data(tmp_path)
    monkeypatch.setattr(Path, "is_dir", lambda self: True)
    env_file = tmp_path / "test.env"
    env_file.write_text("\n".join(f"{key.upper()}={value.as_posix()}" for key, value in data.items()))
    for key in data:
        monkeypatch.delenv(key.upper(), raising=False)
    config = Settings(_env_file=env_file)
    for key, value in data.items():
        assert getattr(config, key) == value.resolve()


@pytest.mark.parametrize("field", ["model_dir", "hf_home", "dataset_dir", "output_dir", "checkpoint_dir", "temp_dir"])
def test_reject_asset_outside_storage(tmp_path, monkeypatch, field):
    data = valid_data(tmp_path)
    monkeypatch.setattr(Path, "is_dir", lambda self: True)
    data[field] = data["project_root"] / "bad"
    with pytest.raises(ValidationError, match="child of STORAGE_ROOT"):
        Settings(_env_file=None, **data)


def test_reject_project_storage(tmp_path, monkeypatch):
    data = valid_data(tmp_path)
    monkeypatch.setattr(Path, "is_dir", lambda self: True)
    data["storage_root"] = data["project_root"] / "assets"
    with pytest.raises(ValidationError, match="separate directory trees"):
        Settings(_env_file=None, **data)


def test_reject_relative_path(tmp_path, monkeypatch):
    data = valid_data(tmp_path)
    monkeypatch.setattr(Path, "is_dir", lambda self: True)
    data["model_dir"] = Path("models")
    with pytest.raises(ValidationError, match="absolute path"):
        Settings(_env_file=None, **data)


@pytest.mark.skipif(os.name != "nt", reason="Windows drive policy")
def test_reject_same_drive(tmp_path, monkeypatch):
    data = valid_data(tmp_path)
    monkeypatch.setattr(Path, "is_dir", lambda self: True)
    data["storage_root"] = data["project_root"].parent / "assets"
    with pytest.raises(ValidationError, match="different drive"):
        Settings(_env_file=None, **data)

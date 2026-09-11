import os
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.app.services.storage import prepare_storage


def test_creates_directories_and_cache(settings):
    prepare_storage(settings)
    assert all(path.is_dir() for path in settings.asset_paths.values())
    assert os.environ["HF_HOME"] == str(settings.hf_home)
    assert Path(os.environ["HF_HUB_CACHE"]).is_relative_to(settings.storage_root)
    assert Path(os.environ["GRADIO_TEMP_DIR"]).is_relative_to(settings.storage_root)


def test_unwritable_storage_fails_closed(settings):
    with patch("backend.app.services.storage.tempfile.TemporaryFile", side_effect=PermissionError("denied")):
        with pytest.raises(RuntimeError, match="not writable.*no fallback"):
            prepare_storage(settings)
    assert not (settings.project_root / "models").exists()


def test_missing_drive_fails_closed(settings):
    with patch.object(Path, "is_dir", return_value=False):
        with pytest.raises(RuntimeError, match="drive unavailable"):
            prepare_storage(settings)


def test_storage_file_collision(settings):
    settings.storage_root.mkdir()
    settings.model_dir.write_text("not a directory")
    with pytest.raises(RuntimeError, match="not writable"):
        prepare_storage(settings)

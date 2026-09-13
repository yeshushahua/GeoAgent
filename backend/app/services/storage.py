"""Fail closed when configured asset storage is unavailable."""
import json
import os
from pathlib import Path
import tempfile

from backend.app.core.config import Settings


def prepare_storage(settings: Settings) -> None:
    root = settings.storage_root
    if not Path(root.anchor).is_dir():
        raise RuntimeError(f"Storage drive unavailable: {root.anchor}; no fallback allowed")
    for directory in (root, *settings.asset_paths.values()):
        try:
            # Re-resolve at startup to detect existing junctions/symlinks.
            if not directory.resolve().is_relative_to(root.resolve()):
                raise OSError("Directory escapes STORAGE_ROOT")
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=directory) as probe:
                probe.write(b"geoagent-storage-check")
                probe.flush()
        except OSError as exc:
            raise RuntimeError(f"Storage directory is not writable: {directory}; no fallback allowed") from exc
    # Set before importing Gradio / any future Hugging Face clients.
    os.environ.update({
        "HF_HOME": str(settings.hf_home),
        "HF_HUB_OFFLINE": str(settings.hf_hub_offline),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_CACHE": str(settings.hf_home / "hub"),
        "HF_ASSETS_CACHE": str(settings.hf_home / "assets"),
        "HF_DATASETS_CACHE": str(settings.hf_home / "datasets"),
        "YOLO_CONFIG_DIR": str(settings.detector_config_dir),
        "YOLO_OFFLINE": "true",
        "GRADIO_TEMP_DIR": str(settings.temp_dir / "gradio"),
        "GRADIO_ANALYTICS_ENABLED": "False",
        "TMP": str(settings.temp_dir),
        "TEMP": str(settings.temp_dir),
        "TMPDIR": str(settings.temp_dir),
    })
    tempfile.tempdir = str(settings.temp_dir)
    # Ultralytics resolves auxiliary text encoders from weights_dir. Write its
    # config before the library is imported so startup does not patch Pillow or
    # initialize optional model code.
    settings_file = settings.detector_config_dir / "Ultralytics" / "settings.json"
    settings_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        current = json.loads(settings_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        current = {"settings_version": "0.0.8"}
    current.update({
        "weights_dir": str(settings.open_vocab_model_path.parent),
        "datasets_dir": str(settings.dataset_dir),
        "runs_dir": str(settings.output_dir / "ultralytics"),
        "sync": False,
    })
    pending = settings_file.with_suffix(".tmp")
    pending.write_text(json.dumps(current, indent=2), encoding="utf-8")
    pending.replace(settings_file)

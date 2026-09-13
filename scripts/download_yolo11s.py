"""Download the fixed YOLO11s COCO checkpoint to configured E-drive storage."""
import os

from backend.app.core.config import get_settings


def main():
    settings = get_settings()
    target = settings.detector_model_path.resolve()
    model_root = settings.model_dir.resolve()
    if target == model_root or not target.is_relative_to(model_root):
        raise RuntimeError("DETECTOR_MODEL_PATH must stay below MODEL_DIR")
    target.parent.mkdir(parents=True, exist_ok=True)
    settings.detector_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(settings.detector_config_dir)

    from ultralytics.utils.downloads import attempt_download_asset

    downloaded = attempt_download_asset(target, release="v8.3.0")
    if not target.is_file() or target.stat().st_size < 10_000_000:
        raise RuntimeError("YOLO11s checkpoint download is missing or incomplete")
    print(f"YOLO11s checkpoint ready: {downloaded}")


if __name__ == "__main__":
    main()

"""Download the fixed Phase 1 model to configured E: storage."""
import os
from pathlib import Path

from backend.app.core.config import get_settings
from backend.app.services.storage import prepare_storage

EXPECTED_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


def directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def main() -> None:
    settings = get_settings()
    prepare_storage(settings)
    if settings.vlm_model_id != EXPECTED_MODEL_ID:
        raise RuntimeError(f"Model ID must be exactly {EXPECTED_MODEL_ID}")
    if not settings.vlm_model_path.is_relative_to(settings.model_dir):
        raise RuntimeError("VLM_MODEL_PATH must remain inside MODEL_DIR; no fallback allowed")
    settings.vlm_model_path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from huggingface_hub import snapshot_download

    print(f"HF_HOME={settings.hf_home}")
    print(f"Model destination={settings.vlm_model_path}")
    snapshot_download(
        repo_id=EXPECTED_MODEL_ID,
        local_dir=settings.vlm_model_path,
        cache_dir=settings.hf_home / "hub",
        max_workers=4,
    )
    from backend.app.models.qwen_vl import validate_model_files

    validate_model_files(settings.vlm_model_path)
    size = directory_size(settings.vlm_model_path)
    print(f"Verified model size: {size / 1024**3:.3f} GiB ({size} bytes)")
    print("Download complete. Production inference remains local_files_only=True.")


if __name__ == "__main__":
    main()

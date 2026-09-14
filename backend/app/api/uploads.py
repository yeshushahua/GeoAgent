from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from backend.app.core.config import Settings
from backend.app.models.errors import InvalidImageError

MAX_UPLOAD_BYTES = 512 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
SAFE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


async def store_temporary_upload(settings: Settings, image: UploadFile) -> Path:
    suffix = Path(image.filename or "").suffix.lower()
    if suffix not in SAFE_SUFFIXES:
        suffix = ".upload"
    root = settings.temp_dir / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{uuid4()}{suffix}"
    total = 0
    try:
        with target.open("wb") as stream:
            while chunk := await image.read(UPLOAD_CHUNK_BYTES):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise InvalidImageError("Upload exceeds the 512 MiB limit")
                stream.write(chunk)
        return target
    except Exception:
        target.unlink(missing_ok=True)
        raise

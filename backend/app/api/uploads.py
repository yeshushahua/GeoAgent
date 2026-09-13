from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from backend.app.core.config import Settings
from backend.app.models.errors import InvalidImageError

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


async def store_temporary_upload(settings: Settings, image: UploadFile) -> Path:
    content = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise InvalidImageError("Image exceeds the 25 MiB upload limit")
    root = settings.temp_dir / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{uuid4()}.upload"
    target.write_bytes(content)
    return target

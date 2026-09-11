from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError

from backend.app.tools.context import ToolContext
from backend.app.tools.errors import InvalidToolImageError

SUPPORTED_FORMATS = {"PNG", "JPEG", "WEBP"}


def open_supported_image(context: ToolContext, value: Path) -> tuple[Path, Image.Image, str]:
    try:
        path = context.validate_read_path(value)
        with Image.open(path) as opened:
            image_format = (opened.format or "").upper()
            if image_format not in SUPPORTED_FORMATS:
                raise InvalidToolImageError("Supported image formats: PNG, JPEG, JPG, WEBP")
            image = opened.copy()
            image.format = opened.format
        return path, image, image_format
    except InvalidToolImageError:
        raise
    except FileNotFoundError as exc:
        raise InvalidToolImageError(str(exc)) from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidToolImageError("Image is missing, unsafe, or not a valid PNG/JPEG/WEBP") from exc

import io

from fastapi import APIRouter, File, Form, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from backend.app.models.errors import InvalidImageError, InvalidInputError
from backend.app.schemas.inference import InferenceResult

router = APIRouter()
SUPPORTED_FORMATS = {"PNG", "JPEG", "WEBP"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
_IMAGE_OPEN = Image.open


@router.get("/status")
def model_status(request: Request):
    return request.app.state.model_manager.status()


@router.post("/load")
def load_model(request: Request):
    return request.app.state.model_manager.load_model()


@router.post("/unload")
def unload_model(request: Request):
    return request.app.state.model_manager.unload_model()


@router.post("/infer", response_model=InferenceResult)
def infer(
    request: Request,
    image: UploadFile = File(...),
    prompt: str = Form(..., min_length=1, max_length=8000),
    max_new_tokens: int = Form(default=256, ge=64, le=512),
):
    if not prompt.strip():
        raise InvalidInputError("Prompt cannot be empty")
    content = image.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise InvalidImageError("Image exceeds the 25 MiB upload limit")
    try:
        with _IMAGE_OPEN(io.BytesIO(content)) as candidate:
            candidate.verify()
        with _IMAGE_OPEN(io.BytesIO(content)) as opened:
            if (opened.format or "").upper() not in SUPPORTED_FORMATS:
                raise InvalidImageError("Supported image formats: PNG, JPEG, JPG, WEBP")
            loaded = opened.copy()
            loaded.format = opened.format
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(
            "Uploaded file is not a valid PNG, JPEG, JPG, or WEBP image"
        ) from exc
    return request.app.state.model_manager.infer(loaded, prompt.strip(), max_new_tokens)

"""Prepare the fixed Phase 5 YOLOE, text encoder, and SAM checkpoints on E:."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
from urllib.request import Request, urlopen

from backend.app.core.config import get_settings
from backend.app.services.storage import prepare_storage


ASSETS = (
    (
        "YOLOE-26s-seg",
        "open_vocab_model_path",
        25_000_000,
        ("https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg.pt",),
    ),
    (
        "MobileCLIP2-B",
        "open_vocab_text_encoder_path",
        200_000_000,
        ("https://github.com/ultralytics/assets/releases/download/v8.3.0/mobileclip2_b.ts",),
    ),
    (
        "SAM 2.1 Base",
        "segmentation_model_path",
        120_000_000,
        ("https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_b.pt",),
    ),
)


def _download(urls: tuple[str, ...], target: Path) -> None:
    partial = target.with_suffix(target.suffix + ".part")
    for url in urls:
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "GeoAgent/Phase5"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with urlopen(Request(url, headers=headers), timeout=120) as response:
                append = offset > 0 and getattr(response, "status", 200) == 206
                with partial.open("ab" if append else "wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            partial.replace(target)
            return
        except Exception:
            if url == urls[-1]:
                raise


def main() -> None:
    settings = get_settings()
    prepare_storage(settings)
    os.environ["YOLO_OFFLINE"] = "false"
    for label, attribute, minimum_size, urls in ASSETS:
        target = Path(getattr(settings, attribute)).resolve()
        if target == settings.model_dir or not target.is_relative_to(settings.model_dir):
            raise RuntimeError(f"{attribute.upper()} must stay below MODEL_DIR")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or target.stat().st_size < minimum_size:
            _download(urls, target)
        size = target.stat().st_size
        if size < minimum_size:
            raise RuntimeError(f"{label} checkpoint is incomplete: {size} bytes")
        print(f"{label} ready: {target} ({size:,} bytes)")


if __name__ == "__main__":
    main()

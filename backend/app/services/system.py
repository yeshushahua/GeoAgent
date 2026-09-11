import logging
import platform

import torch

from backend.app.core.config import Settings

logger = logging.getLogger("geoagent")


def system_info(settings: Settings) -> dict:
    available = torch.cuda.is_available()
    name, vram = None, None
    if available:
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        name = torch.cuda.get_device_name(device)
        vram = round(properties.total_memory / 1024**3, 2)
    else:
        logger.warning("CUDA unavailable; local Phase 0 acceptance requires RTX 4090")
    return {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_available": available,
        "cuda_version": torch.version.cuda,
        "gpu_name": name,
        "gpu_vram_gb": vram,
        "project_root": str(settings.project_root),
        "storage_root": str(settings.storage_root),
    }

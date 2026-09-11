"""Local acceptance against a running FastAPI server; no model downloads."""
import json
import logging
import os
import sys

import httpx
import torch

from backend.app.core.config import get_settings
from backend.app.core.logging import configure_logging
from backend.app.schemas.tool_result import ToolResult
from backend.app.services.storage import prepare_storage


def main() -> int:
    configure_logging()
    logger = logging.getLogger("geoagent")
    try:
        settings = get_settings()
        prepare_storage(settings)
        with httpx.Client(timeout=15, trust_env=False) as client:
            health = client.get(f"{settings.api_base_url}/health")
            health.raise_for_status()
            assert health.json()["status"] == "ok", "Health not OK"
            response = client.get(f"{settings.api_base_url}/system")
            response.raise_for_status()
        info = response.json()
        assert sys.version_info[:2] == (3, 12), "Python 3.12 required"
        assert torch.cuda.is_available(), "Local CUDA unavailable"
        assert "RTX 4090" in torch.cuda.get_device_name(), "Local GPU is not RTX 4090"
        assert info["cuda_available"] is True, "Backend CUDA unavailable"
        assert "RTX 4090" in info["gpu_name"], "Backend GPU is not RTX 4090"
        assert info["storage_root"] == str(settings.storage_root), "Backend storage mismatch"
        assert info["project_root"] == str(settings.project_root), "Backend project mismatch"
        assert all(path.is_dir() for path in settings.asset_paths.values())
        assert os.environ["HF_HOME"] == str(settings.hf_home)
        # Tiny real GPU computation verifies the wheel can execute on this device.
        value = (torch.ones(8, device="cuda") * 2).sum().item()
        assert value == 16, "CUDA tensor computation failed"
        result = ToolResult(success=True, tool="smoke_test", data={"cuda_tensor_sum": value})
        ToolResult.model_validate_json(result.model_dump_json())
        logger.info("System: %s", json.dumps(info, ensure_ascii=False))
        logger.info("Smoke test PASS: health, system, storage, HF cache, torch, RTX 4090 computation, ToolResult")
        return 0
    except Exception:
        logger.exception("Smoke test FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())

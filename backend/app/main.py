from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.app.api import health, models, system
from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import configure_logging
from backend.app.services.storage import prepare_storage
from backend.app.services.system import system_info
from backend.app.models.errors import VlmError
from backend.app.models.manager import ModelManager


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config = settings or get_settings()
        configure_logging(config.log_level)
        prepare_storage(config)
        app.state.settings = config
        app.state.model_manager = ModelManager(config)
        logger = logging.getLogger("geoagent")
        logger.info("GeoAgent starting")
        logger.info("Project root: %s", config.project_root)
        logger.info("Storage root: %s", config.storage_root)
        info = system_info(config)
        logger.info("CUDA available: %s", info["cuda_available"])
        logger.info("GPU: %s", info["gpu_name"])
        yield
        logger.info("GeoAgent stopped")

    app = FastAPI(title="GeoAgent", version="0.2.0", lifespan=lifespan)

    @app.exception_handler(VlmError)
    async def vlm_error_handler(request: Request, exc: VlmError):
        logging.getLogger("geoagent").error("%s: %s", exc.code, exc, exc_info=True)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": str(exc)}},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        errors = exc.errors()
        message = errors[0].get("msg", "Invalid request") if errors else "Invalid request"
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "INVALID_REQUEST", "message": message}},
        )
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(system.router, prefix="/api/v1", tags=["system"])
    app.include_router(models.router, prefix="/api/v1/models/vlm", tags=["vlm"])
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    config = get_settings()
    uvicorn.run(app, host=config.api_host, port=config.api_port)

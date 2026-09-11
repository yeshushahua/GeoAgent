from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from backend.app.api import health, system
from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import configure_logging
from backend.app.services.storage import prepare_storage
from backend.app.services.system import system_info


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config = settings or get_settings()
        configure_logging(config.log_level)
        prepare_storage(config)
        app.state.settings = config
        logger = logging.getLogger("geoagent")
        logger.info("GeoAgent starting")
        logger.info("Project root: %s", config.project_root)
        logger.info("Storage root: %s", config.storage_root)
        info = system_info(config)
        logger.info("CUDA available: %s", info["cuda_available"])
        logger.info("GPU: %s", info["gpu_name"])
        yield
        logger.info("GeoAgent stopped")

    app = FastAPI(title="GeoAgent", version="0.1.0", lifespan=lifespan)
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(system.router, prefix="/api/v1", tags=["system"])
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    config = get_settings()
    uvicorn.run(app, host=config.api_host, port=config.api_port)

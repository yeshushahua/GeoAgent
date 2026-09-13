from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.app.api import advanced_vision, agent, detector, health, models, system, tools
from backend.app.agent import AgentTraceStore, VisionAgent
from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import configure_logging
from backend.app.services.storage import prepare_storage
from backend.app.services.system import system_info
from backend.app.models.errors import VlmError
from backend.app.models.manager import ModelManager
from backend.app.detection.errors import DetectorError
from backend.app.detection.manager import DetectorManager
from backend.app.open_vocabulary import OpenVocabularyDetectorManager
from backend.app.open_vocabulary.errors import OpenVocabularyError
from backend.app.segmentation import SegmentationManager
from backend.app.segmentation.errors import SegmentationError
from backend.app.tools import build_tool_system


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config = settings or get_settings()
        configure_logging(config.log_level)
        prepare_storage(config)
        app.state.settings = config
        logger = logging.getLogger("geoagent")
        app.state.model_manager = ModelManager(config)
        app.state.detector_manager = DetectorManager(config, logger)
        app.state.open_vocab_manager = OpenVocabularyDetectorManager(config, logger)
        app.state.segmentation_manager = SegmentationManager(config, logger)
        registry, executor, traces = build_tool_system(
            config, app.state.model_manager, logger, app.state.detector_manager,
            app.state.open_vocab_manager, app.state.segmentation_manager,
        )
        app.state.tool_registry = registry
        app.state.tool_executor = executor
        app.state.tool_traces = traces
        app.state.agent_traces = AgentTraceStore(config.agent_trace_limit)
        app.state.vision_agent = VisionAgent(
            registry=registry,
            executor=executor,
            model_manager=app.state.model_manager,
            trace_store=app.state.agent_traces,
            logger=logger,
            repair_attempts=config.agent_repair_attempts,
            planner_max_new_tokens=config.agent_planner_max_new_tokens,
        )
        logger.info("GeoAgent starting")
        logger.info("Project root: %s", config.project_root)
        logger.info("Storage root: %s", config.storage_root)
        info = system_info(config)
        logger.info("CUDA available: %s", info["cuda_available"])
        logger.info("GPU: %s", info["gpu_name"])
        yield
        logger.info("GeoAgent stopped")

    app = FastAPI(
        title="GeoAgent", version=(settings or get_settings()).app_version, lifespan=lifespan
    )

    @app.exception_handler(DetectorError)
    async def detector_error_handler(request: Request, exc: DetectorError):
        logging.getLogger("geoagent").error("%s: %s", exc.code, exc, exc_info=True)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": str(exc)}},
        )

    @app.exception_handler(OpenVocabularyError)
    @app.exception_handler(SegmentationError)
    async def advanced_vision_error_handler(request: Request, exc):
        logging.getLogger("geoagent").error("%s: %s", exc.code, exc, exc_info=True)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": str(exc)}},
        )

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
    app.include_router(
        detector.router, prefix="/api/v1/models/detector", tags=["detector"]
    )
    app.include_router(
        advanced_vision.router, prefix="/api/v1/models", tags=["advanced-vision"]
    )
    app.include_router(tools.router, prefix="/api/v1/tools", tags=["tools"])
    app.include_router(agent.router, prefix="/api/v1/agent", tags=["agent"])
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    config = get_settings()
    uvicorn.run(app, host=config.api_host, port=config.api_port)

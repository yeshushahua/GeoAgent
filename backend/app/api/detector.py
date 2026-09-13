from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/status")
def detector_status(request: Request):
    return request.app.state.detector_manager.status()


@router.post("/load")
def load_detector(request: Request):
    return request.app.state.detector_manager.load_model()


@router.post("/unload")
def unload_detector(request: Request):
    return request.app.state.detector_manager.unload_model()

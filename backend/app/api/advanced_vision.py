from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/open-vocabulary/status")
def open_vocabulary_status(request: Request):
    return request.app.state.open_vocab_manager.status()


@router.post("/open-vocabulary/load")
def load_open_vocabulary(request: Request):
    return request.app.state.open_vocab_manager.load_model()


@router.post("/open-vocabulary/unload")
def unload_open_vocabulary(request: Request):
    return request.app.state.open_vocab_manager.unload_model()


@router.get("/segmentation/status")
def segmentation_status(request: Request):
    return request.app.state.segmentation_manager.status()


@router.post("/segmentation/load")
def load_segmentation(request: Request):
    return request.app.state.segmentation_manager.load_model()


@router.post("/segmentation/unload")
def unload_segmentation(request: Request):
    return request.app.state.segmentation_manager.unload_model()

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
def health(request: Request):
    return {"status": "ok", "service": "geoagent", "version": request.app.state.settings.app_version}

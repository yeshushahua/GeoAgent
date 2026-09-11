from fastapi import APIRouter, Request

from backend.app.services.system import system_info

router = APIRouter()


@router.get("/system")
def system(request: Request):
    return system_info(request.app.state.settings)

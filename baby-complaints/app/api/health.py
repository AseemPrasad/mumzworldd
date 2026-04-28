# app/api/health.py
# Developer note: Simple liveness check. Extend to add DB/file connectivity checks.

from fastapi import APIRouter
from app.models.schemas import ApiResponse, HealthResponse
from app.core.config import get_settings

router = APIRouter()


@router.get("/health", response_model=ApiResponse[HealthResponse])
async def health():
    settings = get_settings()
    return ApiResponse(
        ok=True,
        data=HealthResponse(status="ok", version=settings.APP_VERSION),
    )

"""
routers/health.py — Health check endpoint.

GET /health — Returns system status, version, and current timestamp.
Used for monitoring, load balancer health checks, and quick sanity tests.
"""

from fastapi import APIRouter
from datetime import datetime, timezone
from app.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health", summary="System health check")
async def health_check():
    """
    Returns current system status.

    Use this endpoint to verify the API is running and check the version.
    """
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ai_model": settings.openrouter_model,
    }

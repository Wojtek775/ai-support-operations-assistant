"""
routers/metrics.py — GET /metrics endpoint.

Returns aggregated system statistics from the tickets database.
Useful for dashboard consumption and operational monitoring.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.metrics import SystemMetrics
from app.services.metrics_service import get_system_metrics

router = APIRouter(tags=["Metrics"])


@router.get(
    "/metrics",
    response_model=SystemMetrics,
    summary="Get system metrics",
    description=(
        "Returns aggregated statistics about all tickets processed by the AI pipeline: "
        "ticket counts by priority, negative sentiment volume, and the most common category."
    ),
)
def metrics(db: Session = Depends(get_db)):
    """Return current system metrics from the tickets database."""
    return get_system_metrics(db)

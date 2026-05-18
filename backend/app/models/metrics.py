"""
models/metrics.py — Pydantic response model for the GET /metrics endpoint.
"""

from pydantic import BaseModel


class SystemMetrics(BaseModel):
    """Aggregated statistics about tickets processed by the AI support system."""

    total_tickets: int
    urgent_tickets: int
    high_priority_tickets: int
    negative_sentiment_tickets: int
    top_category: str | None

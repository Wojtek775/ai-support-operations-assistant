"""
services/metrics_service.py — Service logic for the GET /metrics endpoint.

Queries the database to aggregate ticket statistics for the dashboard.
Uses SQLAlchemy ORM — compatible with SQLite now, PostgreSQL later.
"""

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.db_models import TicketRecord


def get_system_metrics(db: Session) -> dict:
    """
    Calculate and return aggregated system metrics from the tickets table.

    Args:
        db: SQLAlchemy database session (injected by FastAPI dependency).

    Returns:
        Dict with ticket counts, top category, and workflow metrics,
        matching the SystemMetrics schema.
    """
    total_tickets = db.query(TicketRecord).count()

    urgent_tickets = (
        db.query(TicketRecord)
        .filter(TicketRecord.priority == "urgent")
        .count()
    )

    high_priority_tickets = (
        db.query(TicketRecord)
        .filter(TicketRecord.priority == "high")
        .count()
    )

    negative_sentiment_tickets = (
        db.query(TicketRecord)
        .filter(
            TicketRecord.sentiment.in_(["negative", "angry"])
        )
        .count()
    )

    # Find the category with the most tickets (returns None if table is empty)
    top_category_query = (
        db.query(
            TicketRecord.classification,
            func.count(TicketRecord.id).label("count"),
        )
        .group_by(TicketRecord.classification)
        .order_by(func.count(TicketRecord.id).desc())
        .first()
    )

    top_category = (
        top_category_query[0]
        if top_category_query
        else None
    )

    # Workflow orchestration metrics
    requires_human_review_tickets = (
        db.query(TicketRecord)
        .filter(TicketRecord.requires_human_review == True)  # noqa: E712
        .count()
    )

    escalation_level_2_tickets = (
        db.query(TicketRecord)
        .filter(TicketRecord.escalation_level == 2)
        .count()
    )

    return {
        "total_tickets": total_tickets,
        "urgent_tickets": urgent_tickets,
        "high_priority_tickets": high_priority_tickets,
        "negative_sentiment_tickets": negative_sentiment_tickets,
        "top_category": top_category,
        "requires_human_review_tickets": requires_human_review_tickets,
        "escalation_level_2_tickets": escalation_level_2_tickets,
    }

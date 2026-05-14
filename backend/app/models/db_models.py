"""
models/db_models.py — SQLAlchemy ORM model for the tickets table.

This is the database layer representation of a ticket.
Intentionally kept separate from the Pydantic schemas (models/ticket.py)
to maintain a clean separation between API shapes and DB shapes.

Migration path: set DATABASE_URL to a PostgreSQL connection string and
SQLAlchemy will handle the rest — no code changes required.
"""

from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime, timezone


class Base(DeclarativeBase):
    """SQLAlchemy declarative base — shared by all ORM models."""
    pass


class TicketRecord(Base):
    """
    Database representation of a processed support ticket.

    Table: tickets
    Primary key: id (UUID stored as string for SQLite compatibility)
    """

    __tablename__ = "tickets"

    # Primary key — UUID string (works with both SQLite and PostgreSQL)
    id = Column(String(36), primary_key=True, index=True)

    # Customer info from the original request
    customer_name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False, index=True)
    original_message = Column(Text, nullable=False)

    # AI pipeline output fields
    classification = Column(String(50), nullable=False, index=True)
    priority = Column(String(20), nullable=False, index=True)
    summary = Column(Text, nullable=False)
    sentiment = Column(String(20), nullable=False)
    suggested_response = Column(Text, nullable=False)
    recommended_action = Column(Text, nullable=False)

    # Metadata
    processed_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    ai_model_used = Column(String(100), nullable=False)

    def __repr__(self) -> str:
        return (
            f"<TicketRecord id={self.id!r} "
            f"customer={self.customer_name!r} "
            f"classification={self.classification!r} "
            f"priority={self.priority!r}>"
        )

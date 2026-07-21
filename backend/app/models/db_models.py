"""
models/db_models.py — SQLAlchemy ORM models for tickets and workflow runs.

Intentionally kept separate from the Pydantic schemas (models/ticket.py,
models/workflow.py) to maintain a clean separation between API shapes and
DB shapes.

Migration path: set DATABASE_URL to a PostgreSQL connection string and
SQLAlchemy will handle the rest — no code changes required.
"""

from sqlalchemy import Boolean, Column, Integer, String, Text, DateTime
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

    # Workflow orchestration fields (nullable for backward compatibility)
    assigned_team = Column(String(100), nullable=True)
    sla_hours = Column(Integer, nullable=True)
    requires_human_review = Column(Boolean, nullable=True, default=False)
    escalation_level = Column(Integer, nullable=True, default=0)
    internal_notes = Column(Text, nullable=True)

    # Metadata
    processed_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    ai_model_used = Column(String(100), nullable=False)

    def __repr__(self) -> str:
        return (
            f"<TicketRecord id={self.id!r} "
            f"customer={self.customer_name!r} "
            f"classification={self.classification!r} "
            f"priority={self.priority!r} "
            f"assigned_team={self.assigned_team!r}>"
        )


class WorkflowRunRecord(Base):
    """
    Read model / persistence record for a LangGraph workflow run.

    Table: workflow_runs
    Primary key: workflow_id (UUID string — also used as LangGraph thread_id)

    ticket_id is nullable: workflows paused for human review have not yet
    created a TicketRecord and therefore have no ticket_id.

    status lifecycle:
        running → completed | waiting_review | failed
    """

    __tablename__ = "workflow_runs"

    # Primary key — same value used as LangGraph thread_id
    workflow_id = Column(String(36), primary_key=True, index=True)

    # Nullable until the workflow finishes and creates a TicketRecord.
    # unique=True: one workflow produces at most one ticket.
    ticket_id = Column(String(36), nullable=True, index=True, unique=True)

    # Lifecycle status
    status = Column(String(30), nullable=False, default="running", index=True)

    # Current node at the time of persistence (last node that ran)
    current_node = Column(String(100), nullable=True)

    # Full WorkflowState serialised to JSON (explicit, typed serialisation)
    state_json = Column(Text, nullable=True)

    # Counters
    retry_count = Column(Integer, nullable=False, default=0)

    # Human review fields
    requires_human_review = Column(Boolean, nullable=False, default=False)
    human_review_status = Column(String(30), nullable=True)

    # Timestamps (ISO 8601 strings stored as Text for SQLite portability;
    # DateTime with timezone is not natively supported by SQLite)
    started_at = Column(Text, nullable=False)
    updated_at = Column(Text, nullable=False)
    completed_at = Column(Text, nullable=True)

    # Error info
    error_code = Column(String(50), nullable=True)
    error_message = Column(Text, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<WorkflowRunRecord workflow_id={self.workflow_id!r} "
            f"status={self.status!r} "
            f"ticket_id={self.ticket_id!r}>"
        )

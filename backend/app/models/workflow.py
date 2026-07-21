"""
models/workflow.py — Pydantic schemas for the /workflows API.

WorkflowCreateRequest  — input payload for POST /workflows
WorkflowResponse       — output shape returned by POST and GET /workflows/{id}

Keeps compatibility with TicketRequest field names (customer_name, email,
message) so callers can reuse the same payload shape, but does NOT duplicate
TicketRequest validation logic — it defines its own constraints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, EmailStr, Field


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class WorkflowCreateRequest(BaseModel):
    """
    Input payload for POST /workflows.

    Field names are intentionally compatible with TicketRequest so that
    existing client code can call either endpoint with the same body.
    """

    customer_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Full name of the customer submitting the request.",
        examples=["Alice Smith"],
    )
    email: EmailStr = Field(
        ...,
        description="Customer e-mail address.",
        examples=["alice@example.com"],
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="The support message text to be processed.",
        examples=["My application keeps crashing on login."],
    )

    model_config = {"str_strip_whitespace": True}


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class WorkflowResponse(BaseModel):
    """
    Shape returned by POST /workflows (201) and GET /workflows/{id} (200).

    workflow_id  — UUID, also the LangGraph thread_id
    ticket_id    — None when the workflow is paused for human review
    status       — running | completed | waiting_review | failed
    """

    workflow_id: str
    ticket_id: Optional[str] = None
    status: str

    # Triage output (None until triage node ran)
    classification: Optional[str] = None
    priority: Optional[str] = None
    sentiment: Optional[str] = None
    summary: Optional[str] = None
    confidence: Optional[float] = None

    # Routing output (None until routing node ran)
    assigned_team: Optional[str] = None
    sla_hours: Optional[int] = None
    escalation_level: Optional[int] = None
    requires_human_review: bool = False
    human_review_status: Optional[str] = None

    # Timestamps
    started_at: str
    updated_at: str
    completed_at: Optional[str] = None

    # Error info (populated on failed status)
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    model_config = {"from_attributes": True}

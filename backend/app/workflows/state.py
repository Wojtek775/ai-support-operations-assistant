"""
workflows/state.py — LangGraph WorkflowState definition.

TypedDict used as the graph state. Each node returns a dict
with only the fields it modifies; LangGraph merges changes.

For list fields (execution_trace, errors) we rebuild the list
explicitly in each node rather than using reducers, to keep
the state schema simple and serialisation straightforward.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class WorkflowState(TypedDict):
    # ── Identity ────────────────────────────────────────────────────────────
    workflow_id: str
    # ticket_id is set only after persistence_node creates TicketRecord.
    # None while workflow is running or waiting for human review.
    ticket_id: str | None

    # ── Raw input ────────────────────────────────────────────────────────────
    customer_name: str
    email: str
    message: str

    # ── Triage Agent output ──────────────────────────────────────────────────
    classification: str | None        # TicketClassification enum value
    priority: str | None              # TicketPriority enum value
    sentiment: str | None             # TicketSentiment enum value
    summary: str | None
    confidence: float | None          # 0.0–1.0
    risk_flags: list[str]             # e.g. ["security", "outage"]

    # ── Deterministic routing output ─────────────────────────────────────────
    routing_decision: dict[str, Any] | None   # WorkflowDecision fields as dict

    # ── Resolution Agent output ──────────────────────────────────────────────
    suggested_response: str | None     # filled by ResolutionNode
    recommended_action: str | None     # filled by ResolutionNode
    resolution_tone: str | None        # "empathetic" | "formal" | "urgent" | "informational"
    resolution_confidence: float | None

    # ── Quality Reviewer output ──────────────────────────────────────────────
    review_decision: str | None        # "APPROVED" | "RETRY" | "HUMAN_REVIEW"
    review_confidence: float | None
    review_issues: list[str]           # list of issues found by reviewer
    reviewer_notes: str | None         # notes passed back to Resolution Agent on retry

    # ── Human review ────────────────────────────────────────────────────────
    requires_human_review: bool
    human_review_reason: str | None
    human_review_status: Literal["pending", "approved", "rejected"] | None

    # ── Observability ────────────────────────────────────────────────────────
    workflow_status: Literal["running", "waiting_review", "completed", "failed"]
    current_node: str | None
    retry_count: int
    execution_trace: list[dict[str, Any]]  # one entry per node execution
    errors: list[str]
    started_at: str    # ISO-8601
    updated_at: str    # ISO-8601
    completed_at: str | None

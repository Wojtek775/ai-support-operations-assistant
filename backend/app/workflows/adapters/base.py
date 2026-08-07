"""
adapters/base.py — Protocols and output models for all LLM adapters.

Each agent has its own Protocol so adapters are not forced to implement
methods they do not need (Interface Segregation Principle).

    LLMAdapter        — Triage Agent interface  (triage method)
    ResolutionAdapter — Resolution Agent interface (generate_resolution method)
    ReviewerAdapter   — Quality Reviewer interface (review method)

TriageOutput, ResolutionOutput and ReviewOutput are validated Pydantic models.
They reuse existing enums from models/ticket.py to avoid value duplication.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from app.models.ticket import TicketClassification, TicketPriority, TicketSentiment


# ---------------------------------------------------------------------------
# Triage Agent
# ---------------------------------------------------------------------------

class TriageOutput(BaseModel):
    """Structured output from the Triage Agent. Validated by Pydantic."""

    classification: TicketClassification
    priority: TicketPriority
    sentiment: TicketSentiment
    summary: str = Field(min_length=5, max_length=500)
    confidence: float = Field(ge=0.0, le=1.0)
    risk_flags: list[str] = Field(default_factory=list)


@runtime_checkable
class LLMAdapter(Protocol):
    """
    Triage Agent adapter interface.
    Dependency-injected into TriageNode so tests can use MockTriageAdapter
    without any network calls.
    """

    async def triage(self, customer_name: str, email: str, message: str) -> TriageOutput:
        """Call the LLM and return a validated TriageOutput."""
        ...


# ---------------------------------------------------------------------------
# Resolution Agent
# ---------------------------------------------------------------------------

class ResolutionOutput(BaseModel):
    """
    Structured output from the Resolution Agent. Validated by Pydantic.

    Fields:
        suggested_response  — professional customer-facing reply (10–2000 chars)
        recommended_action  — internal operational step for the support agent (10–500 chars)
        tone                — communication register chosen by the LLM
        confidence          — model certainty (0.0–1.0)
    """

    suggested_response: str = Field(min_length=10, max_length=2000)
    recommended_action: str = Field(min_length=10, max_length=500)
    tone: Literal["empathetic", "formal", "urgent", "informational"]
    confidence: float = Field(ge=0.0, le=1.0)


@runtime_checkable
class ResolutionAdapter(Protocol):
    """
    Resolution Agent adapter interface.

    Dependency-injected into ResolutionNode. The `context` argument is an
    empty list for now and will be populated by a knowledge-base tool in a
    future ETAP (RAG / ticket history).

    Keeps this contract stable: callers pass context=[] today, the adapter
    is ready to use context items when they become available.
    """

    async def generate_resolution(
        self,
        *,
        ticket: dict,
        triage: dict,
        routing_decision: dict,
        context: list[str],
        reviewer_notes: str | None = None,
    ) -> ResolutionOutput:
        """
        Generate a customer-facing response and internal action recommendation.

        Args:
            ticket          — raw ticket fields (customer_name, email, message)
            triage          — triage output (classification, priority, sentiment,
                              summary, confidence, risk_flags)
            routing_decision — routing output (assigned_team, sla_hours,
                               escalation_level, requires_human_review,
                               internal_notes)
            context         — list of RAG/tool context strings (empty for now)
            reviewer_notes  — feedback from previous reviewer pass (retry path)

        Returns:
            ResolutionOutput — validated, non-empty response + action + tone + confidence
        """
        ...


# ---------------------------------------------------------------------------
# Supervisor Agent
# ---------------------------------------------------------------------------

# Valid agent names the Supervisor can dispatch to, plus terminal sentinel.
SUPERVISOR_AGENTS = ("triage", "routing", "resolution", "reviewer", "human_review", "FINISH")


class SupervisorOutput(BaseModel):
    """
    Structured output from the Supervisor Agent. Validated by Pydantic.

    Fields:
        next_agent  — which agent (or FINISH) the supervisor selects
        reasoning   — brief explanation of the decision (stored in execution_trace)
    """

    next_agent: Literal["triage", "routing", "resolution", "reviewer", "human_review", "FINISH"]
    reasoning: str = Field(min_length=5, max_length=500)


@runtime_checkable
class SupervisorAdapter(Protocol):
    """
    Supervisor Agent adapter interface.

    Dependency-injected into SupervisorNode so tests can use
    MockSupervisorAdapter without any network calls.
    """

    async def supervise(
        self,
        *,
        message: str,
        customer_name: str,
        completed_agents: list[str],
        state_summary: dict,
    ) -> SupervisorOutput:
        """
        Decide which agent should run next.

        Args:
            message          — original customer message
            customer_name    — customer name
            completed_agents — agents that have already run (in order)
            state_summary    — key state fields for context (classification,
                               priority, routing_decision, review_decision, etc.)

        Returns:
            SupervisorOutput — next_agent + reasoning
        """
        ...


# ---------------------------------------------------------------------------
# Quality Reviewer Agent
# ---------------------------------------------------------------------------

class ReviewOutput(BaseModel):
    """
    Structured output from the Quality Reviewer Agent. Validated by Pydantic.

    Fields:
        decision    — APPROVED: workflow completes; RETRY: re-run resolution;
                      HUMAN_REVIEW: pause for human review
        confidence  — model certainty (0.0–1.0)
        issues      — list of detected problems (empty when APPROVED)
        notes       — reviewer notes forwarded to Resolution Agent on retry
    """

    decision: Literal["APPROVED", "RETRY", "HUMAN_REVIEW"]
    confidence: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    notes: str


@runtime_checkable
class ReviewerAdapter(Protocol):
    """
    Quality Reviewer adapter interface.

    Dependency-injected into ReviewerNode so tests can use MockReviewerAdapter
    without any network calls.
    """

    async def review(
        self,
        *,
        ticket: dict,
        triage: dict,
        routing_decision: dict,
        resolution: dict,
    ) -> ReviewOutput:
        """
        Review the generated resolution for quality, safety, and completeness.

        Args:
            ticket          — raw ticket fields (customer_name, email, message)
            triage          — triage output (classification, priority, sentiment,
                              summary, confidence, risk_flags)
            routing_decision — routing output (assigned_team, sla_hours,
                               escalation_level, requires_human_review, internal_notes)
            resolution      — resolution output (suggested_response, recommended_action,
                              tone, confidence)

        Returns:
            ReviewOutput — decision + confidence + issues + notes
        """
        ...

"""
tests/test_routing_boundary.py — Routing boundary tests.

Confirms that tickets routed to human_review_gate (CRITICAL priority,
requires_human_review=True) do NOT call ResolutionAdapter and produce
a correct interrupt result.

Requirements verified:
- ResolutionAdapter.generate_resolution() call count == 0
- result contains "__interrupt__"
- suggested_response is None (not generated)
- recommended_action is None (not generated)
- no TicketRecord created
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.models.ticket import TicketClassification, TicketPriority, TicketSentiment
from app.models.db_models import TicketRecord
from app.workflows.adapters.mock import (
    MockResolutionAdapter,
    MockReviewerAdapter,
    MockTriageAdapter,
)
from app.workflows.graph import build_full_graph, make_initial_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_critical_triage_adapter() -> MockTriageAdapter:
    """Return a MockTriageAdapter that produces CRITICAL priority output."""
    return MockTriageAdapter(
        classification=TicketClassification.BILLING,
        priority=TicketPriority.CRITICAL,
        sentiment=TicketSentiment.NEGATIVE,
        summary="Critical billing issue requires immediate human review.",
        confidence=0.98,
        risk_flags=["billing", "critical"],
    )


def _make_requires_review_triage_adapter() -> MockTriageAdapter:
    """Return a MockTriageAdapter with requires_human_review=True via routing."""
    return MockTriageAdapter(
        classification=TicketClassification.BILLING,
        priority=TicketPriority.URGENT,
        sentiment=TicketSentiment.NEGATIVE,
        summary="Urgent billing dispute requires human oversight.",
        confidence=0.91,
        risk_flags=["billing"],
    )


# ---------------------------------------------------------------------------
# Tests: CRITICAL priority → human_review_gate, ResolutionAdapter NOT called
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_critical_ticket_routes_to_human_review_not_resolution():
    """
    CRITICAL priority ticket must be interrupted at human_review_gate.
    ResolutionAdapter must not be called at all.
    """
    resolution_adapter = MockResolutionAdapter()
    reviewer_adapter = MockReviewerAdapter(decision="APPROVED")
    triage_adapter = _make_critical_triage_adapter()

    graph = build_full_graph(
        llm_adapter=triage_adapter,
        resolution_adapter=resolution_adapter,
        reviewer_adapter=reviewer_adapter,
        checkpointer=InMemorySaver(),
    )
    initial_state = make_initial_state(
        workflow_id="test-critical-boundary",
        customer_name="Test User",
        email="test@example.com",
        message="CRITICAL: my account is completely broken and I need help NOW.",
    )

    result = await graph.ainvoke(
        initial_state,
        config={"configurable": {"thread_id": "test-critical-boundary"}},
    )

    # --- Core assertions ---
    # 1. Graph result must contain __interrupt__ (LangGraph interrupt signal)
    assert "__interrupt__" in result, (
        "CRITICAL ticket should produce __interrupt__ but result keys are: "
        f"{list(result.keys())}"
    )

    # 2. ResolutionAdapter must NOT have been called
    assert len(resolution_adapter.call_args_list) == 0, (
        f"ResolutionAdapter was called {len(resolution_adapter.call_args_list)} times "
        "but should not have been called for a CRITICAL ticket"
    )

    # 3. Reviewer must NOT have been called (interrupt happened before resolution)
    assert len(reviewer_adapter.call_args_list) == 0, (
        f"ReviewerAdapter was called {len(reviewer_adapter.call_args_list)} times "
        "but should not be called for a CRITICAL ticket"
    )

    # 4. suggested_response not generated
    assert result.get("suggested_response") is None, (
        "suggested_response should be None for a human-review ticket"
    )

    # 5. recommended_action not generated
    assert result.get("recommended_action") is None, (
        "recommended_action should be None for a human-review ticket"
    )


@pytest.mark.asyncio
async def test_critical_ticket_interrupt_payload():
    """
    The __interrupt__ value must contain a non-empty interrupt reason.
    """
    triage_adapter = _make_critical_triage_adapter()
    resolution_adapter = MockResolutionAdapter()
    reviewer_adapter = MockReviewerAdapter(decision="APPROVED")

    graph = build_full_graph(
        llm_adapter=triage_adapter,
        resolution_adapter=resolution_adapter,
        reviewer_adapter=reviewer_adapter,
        checkpointer=InMemorySaver(),
    )
    initial_state = make_initial_state(
        workflow_id="test-interrupt-payload",
        customer_name="Critical User",
        email="critical@example.com",
        message="Critical: account locked, need immediate help.",
    )

    result = await graph.ainvoke(
        initial_state,
        config={"configurable": {"thread_id": "test-interrupt-payload"}},
    )

    assert "__interrupt__" in result
    interrupt_val = result["__interrupt__"]
    # LangGraph stores interrupt as a tuple/list of Interrupt objects
    assert interrupt_val is not None
    assert len(interrupt_val) > 0


@pytest.mark.asyncio
async def test_critical_ticket_no_ticket_record_created(setup_test_database):
    """
    human_review_gate path must NOT create a TicketRecord.
    Uses the in-memory test DB from conftest.
    """
    from app.services.workflow_run_service import WorkflowRunService
    from tests.conftest import TestingSessionLocal

    db = TestingSessionLocal()
    try:
        initial_count = db.query(TicketRecord).count()

        triage_adapter = _make_critical_triage_adapter()
        resolution_adapter = MockResolutionAdapter()
        reviewer_adapter = MockReviewerAdapter(decision="APPROVED")

        service = WorkflowRunService(
            llm_adapter=triage_adapter,
            resolution_adapter=resolution_adapter,
            reviewer_adapter=reviewer_adapter,
            db=db,
        )

        response = await service.run(
            customer_name="Critical Customer",
            email="critical@example.com",
            message="CRITICAL: complete outage, revenue impact.",
        )

        # Status must be waiting_review (interrupted)
        assert response.status == "waiting_review", (
            f"Expected 'waiting_review' but got '{response.status}'"
        )

        # No ticket_id (no TicketRecord created)
        assert response.ticket_id is None, (
            f"Expected ticket_id=None but got '{response.ticket_id}'"
        )

        # DB must still have the same number of TicketRecords (none added)
        final_count = db.query(TicketRecord).count()
        assert final_count == initial_count, (
            f"TicketRecord count changed from {initial_count} to {final_count}; "
            "no TicketRecord should be created for a human-review ticket"
        )

        # ResolutionAdapter not called
        assert len(resolution_adapter.call_args_list) == 0
    finally:
        db.close()

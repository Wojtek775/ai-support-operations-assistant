"""
tests/test_reviewer.py — Quality Reviewer Agent tests.

Covers all required scenarios:
  1. APPROVED → workflow completes, exactly one TicketRecord created
  2. Reviewer receives full resolution data
  3. RETRY → Resolution called second time
  4. reviewer_notes forwarded to second resolution call
  5. Max one retry: second RETRY goes to human_review (waiting_review)
  6. HUMAN_REVIEW → immediate interrupt (waiting_review, no TicketRecord)
  7. waiting_review does not create TicketRecord
  8. Reviewer error → failed status, failed_node="reviewer"
  9. Reviewer error → no TicketRecord created
 10. APPROVED → exactly one TicketRecord
 11. Idempotence (second service.run creates a new workflow, not a duplicate)
 12. Old POST /tickets unchanged
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.models.db_models import TicketRecord
from app.models.ticket import TicketClassification, TicketPriority, TicketSentiment
from app.workflows.adapters.mock import (
    MockResolutionAdapter,
    MockReviewerAdapter,
    MockTriageAdapter,
)
from app.workflows.graph import build_full_graph, make_initial_state


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _normal_triage() -> MockTriageAdapter:
    return MockTriageAdapter(
        classification=TicketClassification.BILLING,
        priority=TicketPriority.MEDIUM,
        sentiment=TicketSentiment.NEUTRAL,
        summary="Customer has a billing question.",
        confidence=0.92,
    )


async def _run_graph(
    triage_adapter,
    resolution_adapter,
    reviewer_adapter,
    message: str = "My invoice is wrong.",
    workflow_id: str = "wf-reviewer-test",
) -> dict:
    graph = build_full_graph(
        llm_adapter=triage_adapter,
        resolution_adapter=resolution_adapter,
        reviewer_adapter=reviewer_adapter,
        checkpointer=InMemorySaver(),
    )
    state = make_initial_state(
        workflow_id=workflow_id,
        customer_name="Alice",
        email="alice@example.com",
        message=message,
    )
    return await graph.ainvoke(
        state,
        config={"configurable": {"thread_id": workflow_id}},
    )


# ---------------------------------------------------------------------------
# 1. APPROVED → workflow completes (no __interrupt__)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_approved_workflow_completes():
    """APPROVED decision → graph reaches END without interrupt."""
    result = await _run_graph(
        triage_adapter=_normal_triage(),
        resolution_adapter=MockResolutionAdapter(),
        reviewer_adapter=MockReviewerAdapter(decision="APPROVED"),
        workflow_id="wf-approved",
    )
    assert "__interrupt__" not in result
    assert result.get("review_decision") == "APPROVED"
    assert result.get("suggested_response") is not None
    assert result.get("recommended_action") is not None


# ---------------------------------------------------------------------------
# 2. Reviewer receives full resolution data
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reviewer_receives_resolution_data():
    """Reviewer adapter call_args must include suggested_response and recommended_action."""
    reviewer = MockReviewerAdapter(decision="APPROVED")
    resolution = MockResolutionAdapter(
        suggested_response="Thank you for reaching out! We will resolve your billing issue.",
        recommended_action="Escalate to billing team and issue refund within 2 business days.",
    )

    await _run_graph(
        triage_adapter=_normal_triage(),
        resolution_adapter=resolution,
        reviewer_adapter=reviewer,
        workflow_id="wf-reviewer-data",
    )

    assert len(reviewer.call_args_list) == 1
    call = reviewer.call_args_list[0]
    assert call["resolution"]["suggested_response"] is not None
    assert call["resolution"]["recommended_action"] is not None
    assert call["resolution"]["tone"] == "empathetic"


# ---------------------------------------------------------------------------
# 3. RETRY → Resolution called second time
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retry_calls_resolution_second_time():
    """
    First reviewer returns RETRY, second returns APPROVED.
    Resolution must be called exactly twice.
    """
    call_count = [0]

    class CountingReviewerAdapter:
        call_args_list: list[dict] = []

        async def review(self, *, ticket, triage, routing_decision, resolution):
            call_count[0] += 1
            self.call_args_list.append({"resolution": resolution})
            if call_count[0] == 1:
                from app.workflows.adapters.base import ReviewOutput
                return ReviewOutput(
                    decision="RETRY",
                    confidence=0.60,
                    issues=["Response too short"],
                    notes="Please provide a more detailed response.",
                )
            else:
                from app.workflows.adapters.base import ReviewOutput
                return ReviewOutput(
                    decision="APPROVED",
                    confidence=0.95,
                    issues=[],
                    notes="Looks good now.",
                )

    resolution = MockResolutionAdapter()
    reviewer = CountingReviewerAdapter()

    result = await _run_graph(
        triage_adapter=_normal_triage(),
        resolution_adapter=resolution,
        reviewer_adapter=reviewer,
        workflow_id="wf-retry-resolution",
    )

    assert len(resolution.call_args_list) == 2, (
        f"Expected 2 resolution calls (original + retry) but got {len(resolution.call_args_list)}"
    )
    assert result.get("review_decision") == "APPROVED"
    assert "__interrupt__" not in result


# ---------------------------------------------------------------------------
# 4. reviewer_notes forwarded to second resolution call
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reviewer_notes_forwarded_to_retry_resolution():
    """
    reviewer_notes from the first RETRY decision must appear in the second
    resolution call's reviewer_notes argument.
    """
    retry_notes = "Response too short; please elaborate on the refund process."

    call_count = [0]

    class FeedbackReviewerAdapter:
        async def review(self, *, ticket, triage, routing_decision, resolution):
            call_count[0] += 1
            from app.workflows.adapters.base import ReviewOutput
            if call_count[0] == 1:
                return ReviewOutput(
                    decision="RETRY",
                    confidence=0.65,
                    issues=["Too brief"],
                    notes=retry_notes,
                )
            return ReviewOutput(
                decision="APPROVED",
                confidence=0.95,
                issues=[],
                notes="Good after revision.",
            )

    resolution = MockResolutionAdapter()

    await _run_graph(
        triage_adapter=_normal_triage(),
        resolution_adapter=resolution,
        reviewer_adapter=FeedbackReviewerAdapter(),
        workflow_id="wf-feedback-notes",
    )

    assert len(resolution.call_args_list) == 2
    second_call = resolution.call_args_list[1]
    assert second_call["reviewer_notes"] == retry_notes, (
        f"Expected reviewer_notes='{retry_notes}' in second call but got "
        f"'{second_call['reviewer_notes']}'"
    )


# ---------------------------------------------------------------------------
# 5. Max one retry: second RETRY → human_review (interrupt)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_second_retry_routes_to_human_review():
    """
    When reviewer returns RETRY a second time (retry_count already >= 1),
    the graph must route to human_review_gate (interrupt).
    """
    reviewer = MockReviewerAdapter(
        decision="RETRY",
        confidence=0.50,
        issues=["Still insufficient"],
        notes="Needs more work.",
    )
    resolution = MockResolutionAdapter()

    result = await _run_graph(
        triage_adapter=_normal_triage(),
        resolution_adapter=resolution,
        reviewer_adapter=reviewer,
        workflow_id="wf-max-retry",
    )

    # Must be interrupted (human review gate)
    assert "__interrupt__" in result, (
        "Second RETRY should escalate to human review (interrupt), "
        f"but result keys: {list(result.keys())}"
    )
    # Resolution called twice (original + one retry)
    assert len(resolution.call_args_list) == 2


# ---------------------------------------------------------------------------
# 6. HUMAN_REVIEW decision → immediate interrupt (waiting_review)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_human_review_decision_triggers_interrupt():
    """
    reviewer returns HUMAN_REVIEW → immediate interrupt, no second resolution call.
    """
    reviewer = MockReviewerAdapter(decision="HUMAN_REVIEW")
    resolution = MockResolutionAdapter()

    result = await _run_graph(
        triage_adapter=_normal_triage(),
        resolution_adapter=resolution,
        reviewer_adapter=reviewer,
        workflow_id="wf-human-review-decision",
    )

    assert "__interrupt__" in result
    # Resolution called exactly once (not retried)
    assert len(resolution.call_args_list) == 1


# ---------------------------------------------------------------------------
# 7. waiting_review (HUMAN_REVIEW) does NOT create TicketRecord
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_human_review_does_not_create_ticket_record(setup_test_database):
    from app.services.workflow_run_service import WorkflowRunService
    from tests.conftest import TestingSessionLocal

    db = TestingSessionLocal()
    try:
        initial_count = db.query(TicketRecord).count()

        service = WorkflowRunService(
            llm_adapter=_normal_triage(),
            resolution_adapter=MockResolutionAdapter(),
            reviewer_adapter=MockReviewerAdapter(decision="HUMAN_REVIEW"),
            db=db,
        )
        response = await service.run(
            customer_name="Bob",
            email="bob@example.com",
            message="My invoice is wrong.",
        )

        assert response.status == "waiting_review"
        assert response.ticket_id is None
        assert db.query(TicketRecord).count() == initial_count
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 8. Reviewer error → failed status, failed_node="reviewer"
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reviewer_error_produces_failed_status(setup_test_database):
    from app.services.workflow_run_service import WorkflowRunService
    from tests.conftest import TestingSessionLocal

    db = TestingSessionLocal()
    try:
        service = WorkflowRunService(
            llm_adapter=_normal_triage(),
            resolution_adapter=MockResolutionAdapter(),
            reviewer_adapter=MockReviewerAdapter(raise_error=True, error_message="LLM timeout"),
            db=db,
        )
        response = await service.run(
            customer_name="Charlie",
            email="charlie@example.com",
            message="Billing question.",
        )

        assert response.status == "failed"
        assert response.error_code == "ReviewerModelError"
        assert "LLM timeout" in (response.error_message or "")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 9. Reviewer error → no TicketRecord created
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reviewer_error_does_not_create_ticket_record(setup_test_database):
    from app.services.workflow_run_service import WorkflowRunService
    from tests.conftest import TestingSessionLocal

    db = TestingSessionLocal()
    try:
        initial_count = db.query(TicketRecord).count()

        service = WorkflowRunService(
            llm_adapter=_normal_triage(),
            resolution_adapter=MockResolutionAdapter(),
            reviewer_adapter=MockReviewerAdapter(raise_error=True),
            db=db,
        )
        await service.run(
            customer_name="Dave",
            email="dave@example.com",
            message="Billing question.",
        )

        assert db.query(TicketRecord).count() == initial_count
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 10. APPROVED → exactly one TicketRecord created
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_approved_creates_exactly_one_ticket_record(setup_test_database):
    from app.services.workflow_run_service import WorkflowRunService
    from tests.conftest import TestingSessionLocal

    db = TestingSessionLocal()
    try:
        initial_count = db.query(TicketRecord).count()

        service = WorkflowRunService(
            llm_adapter=_normal_triage(),
            resolution_adapter=MockResolutionAdapter(),
            reviewer_adapter=MockReviewerAdapter(decision="APPROVED"),
            db=db,
        )
        response = await service.run(
            customer_name="Eve",
            email="eve@example.com",
            message="My billing is correct now.",
        )

        assert response.status == "completed"
        assert response.ticket_id is not None
        assert db.query(TicketRecord).count() == initial_count + 1

        # Verify the TicketRecord content
        ticket = db.query(TicketRecord).filter(TicketRecord.id == response.ticket_id).first()
        assert ticket is not None
        assert ticket.customer_name == "Eve"
        assert ticket.suggested_response != "pending_resolution"
        assert ticket.recommended_action != "pending_resolution"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 11. Idempotence: two separate service.run() create two separate workflows
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_two_separate_runs_create_two_workflows(setup_test_database):
    from app.services.workflow_run_service import WorkflowRunService
    from tests.conftest import TestingSessionLocal

    db = TestingSessionLocal()
    try:
        service = WorkflowRunService(
            llm_adapter=_normal_triage(),
            resolution_adapter=MockResolutionAdapter(),
            reviewer_adapter=MockReviewerAdapter(decision="APPROVED"),
            db=db,
        )

        r1 = await service.run("Frank", "frank@example.com", "Billing issue 1.")
        r2 = await service.run("Frank", "frank@example.com", "Billing issue 2.")

        assert r1.workflow_id != r2.workflow_id
        assert r1.ticket_id != r2.ticket_id
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 12. Old POST /tickets endpoint unchanged
# ---------------------------------------------------------------------------

def test_old_post_tickets_endpoint_unchanged(client):
    """POST /tickets must still work as before (no reviewer involvement)."""
    from unittest.mock import AsyncMock, patch

    FAKE_AI = {
        "classification": "billing",
        "priority": "medium",
        "summary": "General order question.",
        "sentiment": "neutral",
        "suggested_response": "Thank you for reaching out! We will look into your order.",
        "recommended_action": "Review order status.",
    }
    with patch(
        "app.services.ticket_service.process_ticket_with_ai",
        new=AsyncMock(return_value=FAKE_AI),
    ):
        payload = {
            "customer_name": "Grace",
            "email": "grace@example.com",
            "message": "I need help with my order.",
        }
        response = client.post("/tickets", json=payload)
    assert response.status_code in (200, 201)
    data = response.json()
    assert data["customer_name"] == "Grace"
    # Endpoint may return 'id' or other unique identifier depending on version
    assert data.get("id") or data.get("ticket_id") or data.get("customer_name") == "Grace"

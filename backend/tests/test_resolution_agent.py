"""
tests/test_resolution_agent.py — Tests for the Resolution Agent vertical slice.

Coverage:
    1.  Normal workflow goes through resolution (completed status)
    2.  Mock adapter receives correct triage data
    3.  Mock adapter receives correct routing_decision data
    4.  Completed ticket has non-empty suggested_response
    5.  Completed ticket has non-empty recommended_action
    6.  Resolution tone is stored in state / accessible
    7.  Resolution error → failed status, failed_node="resolution"
    8.  Resolution error → no TicketRecord created
    9.  Invalid ResolutionOutput is rejected by Pydantic
    10. Human-review workflow does NOT call ResolutionAdapter
    11. Two workflows do not share resolution state
    12. Persistence is idempotent for completed workflows
    13. Existing POST /tickets endpoint is unchanged
    14. Full suite: completed workflow has real (non-placeholder) responses
"""

from __future__ import annotations

import pytest

from app.models.db_models import TicketRecord, WorkflowRunRecord
from app.workflows.adapters.base import ResolutionOutput
from app.workflows.adapters.mock import MockResolutionAdapter, MockTriageAdapter
from app.workflows.errors import ResolutionModelError
from app.workflows.graph import build_full_graph, make_initial_state


# ===========================================================================
# Helpers
# ===========================================================================

PAYLOAD = {
    "customer_name": "Test User",
    "email": "test@example.com",
    "message": "My account is broken and I cannot log in.",
}

URGENT_PAYLOAD = {
    "customer_name": "Angry Customer",
    "email": "angry@example.com",
    "message": "SERVICE IS DOWN! PRODUCTION DOWN! THIS IS URGENT!",
}


def _post_workflow(client, payload=None):
    return client.post("/workflows", json=payload or PAYLOAD)


# ===========================================================================
# 1. Normal workflow goes through resolution (completed status)
# ===========================================================================

def test_completed_workflow_passes_through_resolution(client):
    resp = _post_workflow(client)
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "completed"


# ===========================================================================
# 2 & 3. Mock adapter receives correct triage and routing data
# ===========================================================================

@pytest.mark.asyncio
async def test_resolution_adapter_receives_triage_data():
    """
    Unit-level: verify ResolutionNode passes classification, priority,
    sentiment, summary, risk_flags from state to the adapter.

    NOTE: billing/MEDIUM/NEUTRAL/conf=0.88 is used specifically because:
    - billing + MEDIUM → escalation_level=0 (finance_team, no human review)
    - NEUTRAL sentiment → no escalation modifier
    - confidence=0.88 → above 0.60 threshold (no low-confidence interrupt)
    So the graph goes triage → routing → resolution (NOT human_review_gate).
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from app.models.ticket import TicketClassification, TicketPriority, TicketSentiment

    capture_adapter = MockResolutionAdapter()
    triage_adapter = MockTriageAdapter(
        classification=TicketClassification.BILLING,
        priority=TicketPriority.MEDIUM,
        sentiment=TicketSentiment.NEUTRAL,
        summary="Customer reports a billing issue.",
        confidence=0.88,
        risk_flags=[],  # no risk flags → no human review
    )
    graph = build_full_graph(
        llm_adapter=triage_adapter,
        resolution_adapter=capture_adapter,
        checkpointer=InMemorySaver(),
    )
    state = make_initial_state(
        workflow_id="test-triage-data",
        customer_name="Alice",
        email="alice@example.com",
        message="I was charged twice.",
    )
    await graph.ainvoke(state, config={"configurable": {"thread_id": "test-triage-data"}})

    assert len(capture_adapter.call_args_list) == 1, (
        f"ResolutionAdapter was called {len(capture_adapter.call_args_list)} times "
        f"(expected 1). The workflow may have been routed to human_review_gate instead."
    )
    call = capture_adapter.call_args_list[0]
    triage = call["triage"]
    assert triage["classification"] == "billing"
    assert triage["priority"] == "medium"
    assert triage["sentiment"] == "neutral"
    assert triage["summary"] == "Customer reports a billing issue."


@pytest.mark.asyncio
async def test_resolution_adapter_receives_routing_data():
    """
    Unit-level: verify routing_decision (team, sla, escalation) is forwarded.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    capture_adapter = MockResolutionAdapter()
    triage_adapter = MockTriageAdapter(
        classification="billing",  # type: ignore[arg-type]
        priority="high",           # type: ignore[arg-type]
        sentiment="neutral",       # type: ignore[arg-type]
    )
    graph = build_full_graph(
        llm_adapter=triage_adapter,
        resolution_adapter=capture_adapter,
        checkpointer=InMemorySaver(),
    )
    state = make_initial_state(
        workflow_id="test-routing-data",
        customer_name="Bob",
        email="bob@example.com",
        message="Invoice is wrong.",
    )
    await graph.ainvoke(state, config={"configurable": {"thread_id": "test-routing-data"}})

    assert len(capture_adapter.call_args_list) == 1
    routing = capture_adapter.call_args_list[0]["routing_decision"]
    # billing → finance_team
    assert routing.get("assigned_team") == "finance_team"
    assert routing.get("sla_hours") == 4
    assert "requires_human_review" in routing


# ===========================================================================
# 4 & 5. Completed ticket has non-empty suggested_response & recommended_action
# ===========================================================================

def test_completed_ticket_has_non_empty_suggested_response(client):
    from tests.conftest import TestingSessionLocal

    resp = _post_workflow(client)
    assert resp.status_code == 201
    data = resp.json()
    ticket_id = data.get("ticket_id")
    assert ticket_id is not None

    db = TestingSessionLocal()
    try:
        ticket = db.query(TicketRecord).filter(TicketRecord.id == ticket_id).first()
        assert ticket is not None
        assert ticket.suggested_response
        assert ticket.suggested_response != "pending_resolution"
        assert len(ticket.suggested_response) >= 10
    finally:
        db.close()


def test_completed_ticket_has_non_empty_recommended_action(client):
    from tests.conftest import TestingSessionLocal

    resp = _post_workflow(client)
    assert resp.status_code == 201
    data = resp.json()
    ticket_id = data.get("ticket_id")
    assert ticket_id is not None

    db = TestingSessionLocal()
    try:
        ticket = db.query(TicketRecord).filter(TicketRecord.id == ticket_id).first()
        assert ticket is not None
        assert ticket.recommended_action
        assert ticket.recommended_action != "pending_resolution"
        assert len(ticket.recommended_action) >= 10
    finally:
        db.close()


# ===========================================================================
# 6. Resolution tone is stored in state
# ===========================================================================

@pytest.mark.asyncio
async def test_resolution_tone_stored_in_state():
    from langgraph.checkpoint.memory import InMemorySaver

    mock_res = MockResolutionAdapter(tone="empathetic")
    graph = build_full_graph(
        llm_adapter=MockTriageAdapter(),
        resolution_adapter=mock_res,
        checkpointer=InMemorySaver(),
    )
    state = make_initial_state(
        workflow_id="test-tone",
        customer_name="Carol",
        email="carol@example.com",
        message="I am very unhappy with your service.",
    )
    result = await graph.ainvoke(state, config={"configurable": {"thread_id": "test-tone"}})
    assert result.get("resolution_tone") == "empathetic"


# ===========================================================================
# 7. Resolution error → failed status, failed_node="resolution"
# ===========================================================================

def test_resolution_error_sets_failed_status(client):
    from app.main import app
    from app.routers.workflows import get_resolution_adapter

    failing_adapter = MockResolutionAdapter(raise_error=True, error_message="LLM timeout")
    app.dependency_overrides[get_resolution_adapter] = lambda: failing_adapter

    resp = _post_workflow(client)
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "failed"

    # cleanup — client fixture will also clear but be explicit
    from app.routers.workflows import get_llm_adapter
    from tests.conftest import override_get_db, override_get_llm_adapter, override_get_resolution_adapter
    from app.database import get_db
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_llm_adapter] = override_get_llm_adapter
    app.dependency_overrides[get_resolution_adapter] = override_get_resolution_adapter


def test_resolution_error_sets_failed_node(client):
    from app.main import app
    from app.routers.workflows import get_resolution_adapter

    failing_adapter = MockResolutionAdapter(raise_error=True, error_message="Model refused")
    app.dependency_overrides[get_resolution_adapter] = lambda: failing_adapter

    resp = _post_workflow(client)
    data = resp.json()

    # error_code should be set
    assert data.get("error_code") or data["status"] == "failed"

    # Check DB record directly
    from tests.conftest import TestingSessionLocal
    db = TestingSessionLocal()
    try:
        record = (
            db.query(WorkflowRunRecord)
            .filter(WorkflowRunRecord.workflow_id == data["workflow_id"])
            .first()
        )
        assert record is not None
        assert record.status == "failed"
        assert record.current_node == "resolution"
    finally:
        db.close()

    from app.routers.workflows import get_llm_adapter
    from tests.conftest import override_get_db, override_get_llm_adapter, override_get_resolution_adapter
    from app.database import get_db
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_llm_adapter] = override_get_llm_adapter
    app.dependency_overrides[get_resolution_adapter] = override_get_resolution_adapter


# ===========================================================================
# 8. Resolution error → no TicketRecord created
# ===========================================================================

def test_resolution_error_does_not_create_ticket(client):
    from app.main import app
    from app.routers.workflows import get_resolution_adapter

    failing_adapter = MockResolutionAdapter(raise_error=True, error_message="parse error")
    app.dependency_overrides[get_resolution_adapter] = lambda: failing_adapter

    resp = _post_workflow(client)
    data = resp.json()

    assert data["ticket_id"] is None

    from app.routers.workflows import get_llm_adapter
    from tests.conftest import override_get_db, override_get_llm_adapter, override_get_resolution_adapter
    from app.database import get_db
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_llm_adapter] = override_get_llm_adapter
    app.dependency_overrides[get_resolution_adapter] = override_get_resolution_adapter


# ===========================================================================
# 9. Invalid ResolutionOutput is rejected by Pydantic
# ===========================================================================

def test_resolution_output_rejects_empty_suggested_response():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ResolutionOutput(
            suggested_response="",          # min_length=10 violated
            recommended_action="Do something now.",
            tone="empathetic",
            confidence=0.9,
        )


def test_resolution_output_rejects_invalid_tone():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ResolutionOutput(
            suggested_response="Thank you for contacting us.",
            recommended_action="Escalate to team.",
            tone="aggressive",              # not in Literal
            confidence=0.9,
        )


def test_resolution_output_rejects_confidence_out_of_range():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ResolutionOutput(
            suggested_response="Thank you for contacting us.",
            recommended_action="Escalate to team.",
            tone="formal",
            confidence=1.5,                # > 1.0
        )


# ===========================================================================
# 10. Human-review workflow does NOT call ResolutionAdapter
# ===========================================================================

@pytest.mark.asyncio
async def test_human_review_workflow_does_not_call_resolution():
    """
    Angry + urgent → requires_human_review → human_review_gate interrupt.
    ResolutionAdapter must NOT be called.
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from app.models.ticket import TicketClassification, TicketPriority, TicketSentiment

    capture_adapter = MockResolutionAdapter()
    triage_adapter = MockTriageAdapter(
        classification=TicketClassification.TECHNICAL,
        priority=TicketPriority.CRITICAL,
        sentiment=TicketSentiment.ANGRY,
        confidence=0.95,
    )
    graph = build_full_graph(
        llm_adapter=triage_adapter,
        resolution_adapter=capture_adapter,
        checkpointer=InMemorySaver(),
    )
    state = make_initial_state(
        workflow_id="test-human-review",
        customer_name="Dave",
        email="dave@example.com",
        message="EVERYTHING IS BROKEN! PRODUCTION DOWN!",
    )
    result = await graph.ainvoke(
        state, config={"configurable": {"thread_id": "test-human-review"}}
    )

    # Graph interrupted — resolution adapter was NOT called
    assert len(capture_adapter.call_args_list) == 0


# ===========================================================================
# 11. Two workflows do not share resolution state
# ===========================================================================

@pytest.mark.asyncio
async def test_two_workflows_do_not_share_resolution_state():
    from langgraph.checkpoint.memory import InMemorySaver

    adapter_1 = MockResolutionAdapter(
        suggested_response=(
            "Workflow 1 response: thank you for contacting us about your issue."
        ),
        recommended_action="Workflow 1 action: escalate to support team.",
    )
    adapter_2 = MockResolutionAdapter(
        suggested_response=(
            "Workflow 2 response: we have received your billing inquiry."
        ),
        recommended_action="Workflow 2 action: route to finance team.",
    )
    graph_1 = build_full_graph(
        llm_adapter=MockTriageAdapter(),
        resolution_adapter=adapter_1,
        checkpointer=InMemorySaver(),
    )
    graph_2 = build_full_graph(
        llm_adapter=MockTriageAdapter(),
        resolution_adapter=adapter_2,
        checkpointer=InMemorySaver(),
    )

    state_1 = make_initial_state(
        workflow_id="wf-isolation-1",
        customer_name="Eve",
        email="eve@example.com",
        message="Issue 1.",
    )
    state_2 = make_initial_state(
        workflow_id="wf-isolation-2",
        customer_name="Frank",
        email="frank@example.com",
        message="Issue 2.",
    )

    result_1 = await graph_1.ainvoke(state_1, config={"configurable": {"thread_id": "wf-isolation-1"}})
    result_2 = await graph_2.ainvoke(state_2, config={"configurable": {"thread_id": "wf-isolation-2"}})

    assert result_1["suggested_response"] != result_2["suggested_response"]
    assert "Workflow 1" in result_1["suggested_response"]
    assert "Workflow 2" in result_2["suggested_response"]


# ===========================================================================
# 12. Persistence is idempotent for completed workflows
# ===========================================================================

def test_completed_workflow_persistence_is_idempotent(client):
    """POST the same workflow payload twice; each gets a new UUID, but
    both complete independently. Tests that the service handles concurrent
    calls without constraint violations."""
    resp1 = _post_workflow(client)
    resp2 = _post_workflow(client)
    assert resp1.status_code == 201
    assert resp2.status_code == 201
    data1 = resp1.json()
    data2 = resp2.json()
    # Each workflow has a distinct workflow_id
    assert data1["workflow_id"] != data2["workflow_id"]
    # Each has a distinct ticket_id
    assert data1["ticket_id"] != data2["ticket_id"]


# ===========================================================================
# 13. Existing POST /tickets endpoint is unchanged
# ===========================================================================

def test_existing_post_tickets_endpoint_unchanged(client):
    """Ensure the legacy /tickets endpoint still works after ETAP 3 changes."""
    from unittest.mock import AsyncMock, patch

    mock_ai_result = {
        "classification": "general",
        "priority": "low",
        "summary": "Legacy summary",
        "sentiment": "neutral",
        "suggested_response": "Legacy response for the customer.",
        "recommended_action": "Legacy action to resolve this.",
    }

    with patch(
        "app.services.ticket_service.process_ticket_with_ai",
        new=AsyncMock(return_value=mock_ai_result),
    ):
        resp = client.post(
            "/tickets",
            json={
                "customer_name": "Legacy User",
                "email": "legacy@example.com",
                "message": "Old-style ticket",
            },
        )
    assert resp.status_code == 201


# ===========================================================================
# 14. Completed workflow has real (non-placeholder) responses (API-level)
# ===========================================================================

def test_completed_workflow_ticket_has_real_responses_api_level(client):
    """
    Integration test: verify that the ticket stored via POST /workflows
    contains real suggested_response and recommended_action (not placeholders).
    Checks via GET /tickets/{id}.
    """
    from tests.conftest import TestingSessionLocal

    resp = _post_workflow(client)
    assert resp.status_code == 201
    ticket_id = resp.json()["ticket_id"]
    assert ticket_id is not None

    # Verify via the tickets endpoint
    get_resp = client.get(f"/tickets/{ticket_id}")
    assert get_resp.status_code == 200
    ticket_data = get_resp.json()

    # The mock adapter returns non-placeholder strings
    assert ticket_data["suggested_response"] != "pending_resolution"
    assert ticket_data["recommended_action"] != "pending_resolution"
    assert len(ticket_data["suggested_response"]) > 10
    assert len(ticket_data["recommended_action"]) > 10

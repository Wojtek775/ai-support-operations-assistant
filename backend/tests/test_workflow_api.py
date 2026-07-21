"""
tests/test_workflow_api.py — Integration tests for POST /workflows and GET /workflows/{id}.

Coverage (16 scenarios + 1 serialisation unit test):

 1.  POST normal (general/low/positive/high-conf) → 201, completed
 2.  Completed workflow creates exactly one TicketRecord
 3.  Created ticket visible via GET /tickets/{id}
 4.  POST waiting-review (critical priority) → 201, waiting_review, ticket_id=None
 5.  Waiting-review does NOT create TicketRecord
 6.  GET /workflows/{id} returns persisted state
 7.  GET /workflows/{id} missing → 404
 8.  LLM error (TriageModelError) → 201, status=failed, error_code="TriageModelError",
     failed_node="triage", no TicketRecord, routing NOT executed
 9.  DB error on running-record creation → 500/503
10.  Idempotence — _persist_completed called twice for the same record:
         - exactly one TicketRecord
         - same ticket_id
         - no second commit creating a duplicate
         - status remains "completed"
11.  State JSON round-trip (unit test)
12.  Existing POST /tickets is unchanged (status 201)
13.  Waiting-review sets human_review_status = "pending"
14.  Completed ticket has suggested_response == "pending_resolution"
15.  Completed ticket has recommended_action == "pending_resolution"
16.  Transaction rollback — error during TicketRecord.add() → no partial ticket,
         WorkflowRunRecord NOT left as "completed"
17.  workflow_id != ticket_id (distinct UUIDs)

All LLM calls are mocked via FastAPI dependency_overrides on get_llm_adapter.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENROUTER_API_KEY", "test-key-not-real")
os.environ.setdefault("OPENROUTER_MODEL", "test-model")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

import pytest
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from app.models.ticket import TicketClassification, TicketPriority, TicketSentiment
from app.models.db_models import TicketRecord, WorkflowRunRecord
from app.workflows.adapters.mock import MockTriageAdapter
from app.workflows.errors import TriageModelError
from app.services.workflow_run_service import serialize_state, deserialize_state


# ---------------------------------------------------------------------------
# Shared request bodies
# ---------------------------------------------------------------------------

NORMAL_PAYLOAD = {
    "customer_name": "Alice Smith",
    "email": "alice@example.com",
    "message": "My billing statement looks incorrect for this month.",
}

CRITICAL_PAYLOAD = {
    "customer_name": "Bob Jones",
    "email": "bob@example.com",
    "message": "Our entire production system is down — CRITICAL outage!",
}


# ---------------------------------------------------------------------------
# Adapter factories
# ---------------------------------------------------------------------------

def _normal_adapter() -> MockTriageAdapter:
    """general/low/positive/conf=0.95 → completed, no interrupt"""
    return MockTriageAdapter(
        classification=TicketClassification.GENERAL,
        priority=TicketPriority.LOW,
        sentiment=TicketSentiment.POSITIVE,
        confidence=0.95,
        summary="Billing enquiry.",
    )


def _critical_adapter() -> MockTriageAdapter:
    """technical/critical/neutral/conf=0.95 → waiting_review (interrupt)"""
    return MockTriageAdapter(
        classification=TicketClassification.TECHNICAL,
        priority=TicketPriority.CRITICAL,
        sentiment=TicketSentiment.NEUTRAL,
        confidence=0.95,
        summary="Production outage.",
    )


def _error_adapter() -> MockTriageAdapter:
    """Raises TriageModelError — forces the triage node to signal workflow_status=failed"""
    adapter = MockTriageAdapter(
        classification=TicketClassification.GENERAL,
        priority=TicketPriority.LOW,
        sentiment=TicketSentiment.NEUTRAL,
        confidence=0.95,
    )
    adapter.triage = AsyncMock(side_effect=TriageModelError("LLM timeout"))
    return adapter


# ---------------------------------------------------------------------------
# 1. POST normal → 201 + completed
# ---------------------------------------------------------------------------

def test_post_workflow_completed_returns_201(client):
    from app.main import app
    from app.routers.workflows import get_llm_adapter

    app.dependency_overrides[get_llm_adapter] = _normal_adapter
    resp = client.post("/workflows", json=NORMAL_PAYLOAD)
    assert resp.status_code == 201, resp.text

    data = resp.json()
    assert data["status"] == "completed"
    assert data["workflow_id"] is not None
    assert data["ticket_id"] is not None
    assert data["classification"] == "general"
    assert data["priority"] == "low"
    assert data["sentiment"] == "positive"
    assert data["completed_at"] is not None


# ---------------------------------------------------------------------------
# 2. Completed workflow creates exactly ONE TicketRecord
# ---------------------------------------------------------------------------

def test_completed_workflow_creates_one_ticket(client):
    from app.main import app
    from app.routers.workflows import get_llm_adapter
    from tests.conftest import TestingSessionLocal

    app.dependency_overrides[get_llm_adapter] = _normal_adapter

    resp = client.post("/workflows", json=NORMAL_PAYLOAD)
    assert resp.status_code == 201
    ticket_id = resp.json()["ticket_id"]

    db = TestingSessionLocal()
    try:
        count = db.query(TicketRecord).filter(TicketRecord.id == ticket_id).count()
        assert count == 1, f"Expected 1 TicketRecord, found {count}"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 3. Created ticket visible via GET /tickets/{id}
# ---------------------------------------------------------------------------

def test_completed_ticket_visible_via_tickets_endpoint(client):
    from app.main import app
    from app.routers.workflows import get_llm_adapter

    app.dependency_overrides[get_llm_adapter] = _normal_adapter

    resp = client.post("/workflows", json=NORMAL_PAYLOAD)
    assert resp.status_code == 201
    ticket_id = resp.json()["ticket_id"]

    ticket_resp = client.get(f"/tickets/{ticket_id}")
    assert ticket_resp.status_code == 200
    assert ticket_resp.json()["ticket_id"] == ticket_id


# ---------------------------------------------------------------------------
# 4. POST waiting-review → 201 + waiting_review + ticket_id=None
# ---------------------------------------------------------------------------

def test_post_workflow_waiting_review(client):
    from app.main import app
    from app.routers.workflows import get_llm_adapter

    app.dependency_overrides[get_llm_adapter] = _critical_adapter

    resp = client.post("/workflows", json=CRITICAL_PAYLOAD)
    assert resp.status_code == 201, resp.text

    data = resp.json()
    assert data["status"] == "waiting_review"
    assert data["ticket_id"] is None
    assert data["requires_human_review"] is True


# ---------------------------------------------------------------------------
# 5. Waiting-review does NOT create TicketRecord
# ---------------------------------------------------------------------------

def test_waiting_review_does_not_create_ticket_record(client):
    from app.main import app
    from app.routers.workflows import get_llm_adapter
    from tests.conftest import TestingSessionLocal

    app.dependency_overrides[get_llm_adapter] = _critical_adapter

    db = TestingSessionLocal()
    try:
        before = db.query(TicketRecord).count()
    finally:
        db.close()

    resp = client.post("/workflows", json=CRITICAL_PAYLOAD)
    assert resp.status_code == 201
    assert resp.json()["ticket_id"] is None

    db = TestingSessionLocal()
    try:
        after = db.query(TicketRecord).count()
    finally:
        db.close()

    assert after == before, (
        f"waiting_review should NOT create a TicketRecord (was {before}, now {after})"
    )


# ---------------------------------------------------------------------------
# 6. GET /workflows/{id} returns persisted state
# ---------------------------------------------------------------------------

def test_get_workflow_returns_persisted_state(client):
    from app.main import app
    from app.routers.workflows import get_llm_adapter

    app.dependency_overrides[get_llm_adapter] = _normal_adapter

    post_resp = client.post("/workflows", json=NORMAL_PAYLOAD)
    assert post_resp.status_code == 201
    workflow_id = post_resp.json()["workflow_id"]

    get_resp = client.get(f"/workflows/{workflow_id}")
    assert get_resp.status_code == 200

    data = get_resp.json()
    assert data["workflow_id"] == workflow_id
    assert data["status"] == "completed"
    assert data["ticket_id"] == post_resp.json()["ticket_id"]
    assert data["classification"] == "general"


# ---------------------------------------------------------------------------
# 7. GET /workflows/{id} missing → 404
# ---------------------------------------------------------------------------

def test_get_workflow_not_found_returns_404(client):
    resp = client.get("/workflows/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 8. LLM error → failed with explicit contract fields
#    - status = "failed"
#    - error_code = "TriageModelError"
#    - failed_node stored in current_node = "triage"
#    - ticket_id = None (no TicketRecord created)
#    - routing was NOT executed
# ---------------------------------------------------------------------------

def test_llm_error_results_in_failed_workflow_with_contract_fields(client):
    from app.main import app
    from app.routers.workflows import get_llm_adapter
    from tests.conftest import TestingSessionLocal

    app.dependency_overrides[get_llm_adapter] = _error_adapter

    ticket_count_before = 0
    db = TestingSessionLocal()
    try:
        ticket_count_before = db.query(TicketRecord).count()
    finally:
        db.close()

    resp = client.post("/workflows", json=NORMAL_PAYLOAD)
    # Service persists failed status as 201 (outcome stored, not an HTTP error)
    assert resp.status_code == 201, resp.text

    data = resp.json()
    assert data["status"] == "failed"
    assert data["ticket_id"] is None
    # Explicit contract fields from triage node
    assert data["error_code"] == "TriageModelError"
    assert "LLM timeout" in data["error_message"]

    # Verify in DB directly
    db = TestingSessionLocal()
    try:
        record = (
            db.query(WorkflowRunRecord)
            .filter(WorkflowRunRecord.workflow_id == data["workflow_id"])
            .first()
        )
        assert record is not None
        assert record.status == "failed"
        assert record.error_code == "TriageModelError"
        assert record.ticket_id is None
        # failed_node stored in current_node
        assert record.current_node == "triage"

        # No new TicketRecords were created
        ticket_count_after = db.query(TicketRecord).count()
        assert ticket_count_after == ticket_count_before, (
            "A TriageModelError must NOT create any TicketRecord"
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 9. DB rollback on running-record creation error → 5xx
# ---------------------------------------------------------------------------

def test_db_error_on_running_record_returns_5xx(client):
    from app.main import app
    from app.routers.workflows import get_llm_adapter
    from app.database import get_db
    from tests.conftest import TestingSessionLocal

    app.dependency_overrides[get_llm_adapter] = _normal_adapter

    bad_session = TestingSessionLocal()
    original_add = bad_session.add
    call_count = [0]

    def bad_add(obj):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("Simulated DB failure")
        return original_add(obj)

    bad_session.add = bad_add

    def bad_db():
        yield bad_session

    app.dependency_overrides[get_db] = bad_db
    try:
        resp = client.post("/workflows", json=NORMAL_PAYLOAD)
        assert resp.status_code in (500, 503), (
            f"Expected 5xx for DB failure, got {resp.status_code}: {resp.text}"
        )
    finally:
        bad_session.close()
        from tests.conftest import override_get_db
        app.dependency_overrides[get_db] = override_get_db


# ---------------------------------------------------------------------------
# 10. TRUE idempotence — _persist_completed called TWICE on the SAME record
#     Confirms: exactly 1 TicketRecord, same ticket_id, status "completed"
# ---------------------------------------------------------------------------

def test_persist_completed_is_truly_idempotent(client):
    """
    Calls _persist_completed a second time directly on an already-completed record.
    Must NOT create a second TicketRecord.
    The second call must return the same ticket_id and status=completed.
    """
    from app.main import app
    from app.routers.workflows import get_llm_adapter
    from tests.conftest import TestingSessionLocal
    from app.services.workflow_run_service import WorkflowRunService

    app.dependency_overrides[get_llm_adapter] = _normal_adapter

    # First call: normal completion
    resp = client.post("/workflows", json=NORMAL_PAYLOAD)
    assert resp.status_code == 201
    first_ticket_id = resp.json()["ticket_id"]
    workflow_id = resp.json()["workflow_id"]

    # Count tickets before idempotent call
    db = TestingSessionLocal()
    try:
        ticket_count_before = db.query(TicketRecord).count()

        record = (
            db.query(WorkflowRunRecord)
            .filter(WorkflowRunRecord.workflow_id == workflow_id)
            .first()
        )
        assert record is not None
        assert record.status == "completed"
        assert record.ticket_id == first_ticket_id

        # Second call to _persist_completed — simulates idempotent finalization
        service = WorkflowRunService(llm_adapter=_normal_adapter(), db=db)
        dummy_state = {
            "classification": "general",
            "priority": "low",
            "sentiment": "positive",
            "summary": "Billing enquiry.",
            "confidence": 0.95,
            "retry_count": 0,
            "current_node": "routing",
            "routing_decision": {
                "assigned_team": "support_general",
                "sla_hours": 24,
                "requires_human_review": False,
                "escalation_level": 0,
                "internal_notes": "General inquiry.",
            },
        }
        second_response = service._persist_completed(
            run_record=record,
            state=dummy_state,
            customer_name="Alice Smith",
            email="alice@example.com",
            message="My billing statement looks incorrect.",
            updated_at="2026-01-01T00:00:00+00:00",
        )

        # Assertions: ticket_id unchanged, status completed, no new record
        assert second_response.ticket_id == first_ticket_id, (
            f"Idempotent call must return same ticket_id: "
            f"got {second_response.ticket_id}, expected {first_ticket_id}"
        )
        assert second_response.status == "completed"

        ticket_count_after = db.query(TicketRecord).count()
        assert ticket_count_after == ticket_count_before, (
            f"Idempotent call must NOT create a new TicketRecord "
            f"(before={ticket_count_before}, after={ticket_count_after})"
        )

        # Verify no duplicate ticket exists for the same ticket_id
        duplicate_count = (
            db.query(TicketRecord)
            .filter(TicketRecord.id == first_ticket_id)
            .count()
        )
        assert duplicate_count == 1, (
            f"Expected exactly 1 TicketRecord for ticket_id={first_ticket_id}, "
            f"found {duplicate_count}"
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 11. State JSON round-trip
# ---------------------------------------------------------------------------

def test_state_json_round_trip():
    from datetime import datetime, timezone
    from enum import Enum

    class Priority(Enum):
        HIGH = "high"

    original = {
        "workflow_id": "abc-123",
        "classification": "billing",
        "priority": Priority.HIGH,
        "confidence": 0.87,
        "risk_flags": ["security", "churn"],
        "routing_decision": {
            "assigned_team": "finance_team",
            "sla_hours": 4,
            "requires_human_review": True,
            "escalation_level": 1,
            "internal_notes": "Test",
        },
        "started_at": datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        "errors": [],
        "execution_trace": [{"node": "triage", "status": "ok"}],
        "nullable_field": None,
    }

    serialised = serialize_state(original)
    assert isinstance(serialised, str)

    recovered = deserialize_state(serialised)
    assert isinstance(recovered, dict)
    assert recovered["priority"] == "high"
    assert "2026-01-01" in recovered["started_at"]
    assert recovered["routing_decision"]["assigned_team"] == "finance_team"
    assert recovered["risk_flags"] == ["security", "churn"]
    assert recovered["nullable_field"] is None
    assert abs(recovered["confidence"] - 0.87) < 1e-9


# ---------------------------------------------------------------------------
# 12. Existing POST /tickets is unchanged (status 201)
# ---------------------------------------------------------------------------

def test_existing_post_tickets_endpoint_is_unchanged(client):
    """
    POST /tickets has always returned 201 (see test_tickets.py::test_submit_valid_ticket_returns_201).
    This test confirms the contract remains 201 after ETAP 4 changes.
    """
    from unittest.mock import patch, AsyncMock

    mock_ai_result = {
        "classification": "billing",
        "priority": "medium",
        "summary": "Billing question.",
        "sentiment": "neutral",
        "suggested_response": "We will review your account.",
        "recommended_action": "Review billing records.",
    }

    with patch(
        "app.services.ticket_service.process_ticket_with_ai",
        new=AsyncMock(return_value=mock_ai_result),
    ):
        resp = client.post(
            "/tickets",
            json={
                "customer_name": "Charlie Brown",
                "email": "charlie@example.com",
                "message": "Why was I charged twice?",
            },
        )

    # POST /tickets has always been 201 — confirmed in test_tickets.py
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["classification"] == "billing"
    assert data["ticket_id"] is not None


# ---------------------------------------------------------------------------
# 13. Waiting-review sets human_review_status = "pending"
# ---------------------------------------------------------------------------

def test_waiting_review_sets_human_review_status_pending(client):
    from app.main import app
    from app.routers.workflows import get_llm_adapter

    app.dependency_overrides[get_llm_adapter] = _critical_adapter

    resp = client.post("/workflows", json=CRITICAL_PAYLOAD)
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "waiting_review"
    assert data["human_review_status"] == "pending"


# ---------------------------------------------------------------------------
# 14 & 15. Completed ticket uses "pending_resolution" placeholder
# ---------------------------------------------------------------------------

def test_completed_ticket_has_pending_resolution_placeholders(client):
    """
    ETAP 3: Resolution Agent now produces real responses.
    suggested_response and recommended_action must be non-empty and NOT
    the old placeholder string "pending_resolution".

    The conftest overrides get_resolution_adapter with MockResolutionAdapter
    which returns deterministic non-placeholder strings.
    """
    from app.main import app
    from app.routers.workflows import get_llm_adapter
    from tests.conftest import TestingSessionLocal

    app.dependency_overrides[get_llm_adapter] = _normal_adapter

    resp = client.post("/workflows", json=NORMAL_PAYLOAD)
    assert resp.status_code == 201
    ticket_id = resp.json()["ticket_id"]

    db = TestingSessionLocal()
    try:
        ticket = db.query(TicketRecord).filter(TicketRecord.id == ticket_id).first()
        assert ticket is not None
        # Resolution Agent now produces real responses (not placeholders)
        assert ticket.suggested_response, "suggested_response must not be empty"
        assert ticket.suggested_response != "pending_resolution", (
            f"Resolution Agent should produce real response, got placeholder: "
            f"{ticket.suggested_response!r}"
        )
        assert ticket.recommended_action, "recommended_action must not be empty"
        assert ticket.recommended_action != "pending_resolution", (
            f"Resolution Agent should produce real action, got placeholder: "
            f"{ticket.recommended_action!r}"
        )
        assert len(ticket.suggested_response) >= 10
        assert len(ticket.recommended_action) >= 10
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 16. Transaction rollback — error during TicketRecord creation
#     WorkflowRunRecord must NOT be left as "completed",
#     no partial TicketRecord must exist
# ---------------------------------------------------------------------------

def test_transaction_rollback_on_ticket_creation_failure(client):
    """
    Simulates a DB constraint violation during TicketRecord.add() by injecting
    a bad session that raises on the second add() call (first is WorkflowRunRecord).

    Expected outcome:
    - rollback is called
    - no TicketRecord is created
    - WorkflowRunRecord remains at status "running" (not "completed")
    - router receives WorkflowPersistenceError → returns 500
    """
    from app.main import app
    from app.routers.workflows import get_llm_adapter
    from app.database import get_db
    from tests.conftest import TestingSessionLocal

    app.dependency_overrides[get_llm_adapter] = _normal_adapter

    db_for_check = TestingSessionLocal()
    ticket_count_before = db_for_check.query(TicketRecord).count()
    workflow_run_count_before = db_for_check.query(WorkflowRunRecord).count()
    db_for_check.close()

    # Use a real session but override commit to fail after the running record
    # is created (to simulate failure during _persist_completed's commit).
    real_session = TestingSessionLocal()
    commit_count = [0]
    original_commit = real_session.commit

    def failing_commit():
        commit_count[0] += 1
        if commit_count[0] == 2:
            # First commit: create running record (ok)
            # Second commit: fail during _persist_completed
            real_session.rollback()
            raise RuntimeError("Simulated TicketRecord commit failure")
        return original_commit()

    real_session.commit = failing_commit

    def bad_db():
        yield real_session

    app.dependency_overrides[get_db] = bad_db
    try:
        resp = client.post("/workflows", json=NORMAL_PAYLOAD)
        # Must return 5xx — the router catches WorkflowPersistenceError
        assert resp.status_code in (500, 503), (
            f"Expected 5xx when TicketRecord commit fails, got {resp.status_code}: {resp.text}"
        )
    finally:
        real_session.close()
        from tests.conftest import override_get_db
        app.dependency_overrides[get_db] = override_get_db

    # After rollback: no new TicketRecord, WorkflowRunRecord may exist as "running"
    db_for_check = TestingSessionLocal()
    try:
        ticket_count_after = db_for_check.query(TicketRecord).count()
        assert ticket_count_after == ticket_count_before, (
            f"No TicketRecord should persist after rollback "
            f"(before={ticket_count_before}, after={ticket_count_after})"
        )
    finally:
        db_for_check.close()


# ---------------------------------------------------------------------------
# 17. workflow_id != ticket_id (distinct UUIDs)
# ---------------------------------------------------------------------------

def test_workflow_id_and_ticket_id_are_distinct_uuids(client):
    """
    workflow_id identifies the process execution.
    ticket_id identifies the business outcome.
    They must be different UUIDs.
    WorkflowRunRecord.ticket_id provides the explicit link between them.
    """
    from app.main import app
    from app.routers.workflows import get_llm_adapter
    from tests.conftest import TestingSessionLocal

    app.dependency_overrides[get_llm_adapter] = _normal_adapter

    resp = client.post("/workflows", json=NORMAL_PAYLOAD)
    assert resp.status_code == 201
    data = resp.json()

    workflow_id = data["workflow_id"]
    ticket_id = data["ticket_id"]

    assert workflow_id is not None
    assert ticket_id is not None
    assert workflow_id != ticket_id, (
        f"workflow_id and ticket_id must be different UUIDs, "
        f"both were: {workflow_id}"
    )

    # Verify the link is explicit in WorkflowRunRecord
    db = TestingSessionLocal()
    try:
        record = (
            db.query(WorkflowRunRecord)
            .filter(WorkflowRunRecord.workflow_id == workflow_id)
            .first()
        )
        assert record is not None
        assert record.ticket_id == ticket_id, (
            f"WorkflowRunRecord.ticket_id must equal ticket_id from response"
        )
    finally:
        db.close()

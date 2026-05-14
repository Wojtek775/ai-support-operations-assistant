"""
tests/test_tickets.py — API endpoint tests.

All fixtures (client, database, env vars) come from conftest.py.
Tests never call the real OpenRouter API.

Mock patch target explanation:
    ticket_service.py does:
        from app.services.ai_service import process_ticket_with_ai
    This binds the name 'process_ticket_with_ai' into the ticket_service module.
    To intercept it at runtime, we must patch the name WHERE IT IS USED:
        app.services.ticket_service.process_ticket_with_ai
    NOT where it was originally defined (app.services.ai_service).

Run with:
    cd backend
    python -m pytest tests/ -v --tb=short
"""

import pytest
from unittest.mock import AsyncMock, patch


# ---------------------------------------------------------------------------
# Fake AI result — replaces the real OpenRouter API response in all tests
# ---------------------------------------------------------------------------

FAKE_AI_RESULT = {
    "classification": "billing",
    "priority": "high",
    "summary": "Customer reports a duplicate charge.",
    "sentiment": "negative",          # must be a SentimentEnum value: positive/neutral/negative/angry
    "suggested_response": (
        "I'm sorry about the duplicate charge. "
        "Please share your invoice number so we can investigate it."
    ),
    "recommended_action": "Escalate to billing support.",
}

# Correct patch target: the name as it exists in the module that CALLS it.
AI_PATCH_TARGET = "app.services.ticket_service.process_ticket_with_ai"


# ---------------------------------------------------------------------------
# Tests: GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        """Health endpoint should always return HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """Health response must include status, app, version, timestamp."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"
        assert "app" in data
        assert "version" in data
        assert "timestamp" in data


# ---------------------------------------------------------------------------
# Tests: POST /tickets
# ---------------------------------------------------------------------------

class TestSubmitTicket:

    @patch(AI_PATCH_TARGET, new_callable=AsyncMock, return_value=FAKE_AI_RESULT)
    def test_submit_valid_ticket_returns_201(self, mock_ai, client):
        """A valid ticket submission should return HTTP 201."""
        payload = {
            "customer_name": "Jan Kowalski",
            "email": "jan@example.com",
            "message": "My invoice is wrong, I was charged twice this month.",
        }
        response = client.post("/tickets", json=payload)
        assert response.status_code == 201

    @patch(AI_PATCH_TARGET, new_callable=AsyncMock, return_value=FAKE_AI_RESULT)
    def test_submit_ticket_response_has_all_ai_fields(self, mock_ai, client):
        """Response must include all 6 AI-generated fields plus metadata."""
        payload = {
            "customer_name": "Anna Nowak",
            "email": "anna@example.com",
            "message": "I cannot login to my account since yesterday, please help.",
        }
        response = client.post("/tickets", json=payload)
        assert response.status_code == 201
        data = response.json()

        assert "ticket_id" in data
        assert "classification" in data
        assert "priority" in data
        assert "summary" in data
        assert "sentiment" in data
        assert "suggested_response" in data
        assert "recommended_action" in data
        assert "processed_at" in data
        assert "ai_model_used" in data

    @patch(AI_PATCH_TARGET, new_callable=AsyncMock, return_value=FAKE_AI_RESULT)
    def test_submit_ticket_ai_values_match_fake_result(self, mock_ai, client):
        """AI fields in response must match the mocked return value."""
        payload = {
            "customer_name": "Piotr Zając",
            "email": "piotr@example.com",
            "message": "My invoice is wrong and I demand a full refund immediately.",
        }
        response = client.post("/tickets", json=payload)
        assert response.status_code == 201
        data = response.json()

        assert data["classification"] == "billing"
        assert data["priority"] == "high"
        assert data["summary"] == "Customer reports a duplicate charge."
        assert data["sentiment"] == "negative"

    @patch(AI_PATCH_TARGET, new_callable=AsyncMock, return_value=FAKE_AI_RESULT)
    def test_submit_ticket_preserves_original_message(self, mock_ai, client):
        """The original_message field must be the exact submitted message."""
        message = "This is a test message that is long enough to pass validation."
        payload = {
            "customer_name": "Test User",
            "email": "test@example.com",
            "message": message,
        }
        response = client.post("/tickets", json=payload)
        assert response.status_code == 201
        assert response.json()["original_message"] == message

    def test_submit_ticket_missing_message_returns_422(self, client):
        """Request body missing 'message' must be rejected with 422."""
        payload = {
            "customer_name": "Jan Kowalski",
            "email": "jan@example.com",
        }
        response = client.post("/tickets", json=payload)
        assert response.status_code == 422

    def test_submit_ticket_message_too_short_returns_422(self, client):
        """Message shorter than 10 characters must be rejected with 422."""
        payload = {
            "customer_name": "Jan Kowalski",
            "email": "jan@example.com",
            "message": "Help",  # 4 chars — below min_length=10
        }
        response = client.post("/tickets", json=payload)
        assert response.status_code == 422

    def test_submit_ticket_invalid_email_returns_422(self, client):
        """Invalid email format must be rejected with 422."""
        payload = {
            "customer_name": "Jan Kowalski",
            "email": "not-a-valid-email",
            "message": "My invoice is wrong, I was charged twice this month.",
        }
        response = client.post("/tickets", json=payload)
        assert response.status_code == 422

    def test_submit_ticket_empty_body_returns_422(self, client):
        """Empty JSON body must be rejected with 422."""
        response = client.post("/tickets", json={})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Tests: GET /tickets
# ---------------------------------------------------------------------------

class TestListTickets:

    def test_list_tickets_returns_200(self, client):
        """GET /tickets should return HTTP 200."""
        response = client.get("/tickets")
        assert response.status_code == 200

    def test_list_tickets_returns_list(self, client):
        """GET /tickets should return a JSON array."""
        response = client.get("/tickets")
        assert isinstance(response.json(), list)

    @patch(AI_PATCH_TARGET, new_callable=AsyncMock, return_value=FAKE_AI_RESULT)
    def test_list_tickets_contains_submitted_ticket(self, mock_ai, client):
        """After submitting a ticket, GET /tickets must include it."""
        payload = {
            "customer_name": "List Test User",
            "email": "listtest@example.com",
            "message": "Testing that my ticket appears in the list endpoint.",
        }
        post_response = client.post("/tickets", json=payload)
        assert post_response.status_code == 201

        list_response = client.get("/tickets")
        assert list_response.status_code == 200
        emails = [t["email"] for t in list_response.json()]
        assert "listtest@example.com" in emails


# ---------------------------------------------------------------------------
# Tests: GET /tickets/{ticket_id}
# ---------------------------------------------------------------------------

class TestGetTicketById:

    @patch(AI_PATCH_TARGET, new_callable=AsyncMock, return_value=FAKE_AI_RESULT)
    def test_get_ticket_by_id_returns_200_and_full_record(self, mock_ai, client):
        """Submit a ticket, then retrieve it by ID — should return full data."""
        payload = {
            "customer_name": "Maria Wiśniewska",
            "email": "maria@example.com",
            "message": "My package has not arrived after 10 days, please investigate.",
        }
        create_response = client.post("/tickets", json=payload)
        assert create_response.status_code == 201
        ticket_id = create_response.json()["ticket_id"]

        get_response = client.get(f"/tickets/{ticket_id}")
        assert get_response.status_code == 200

        data = get_response.json()
        assert data["ticket_id"] == ticket_id
        assert data["customer_name"] == "Maria Wiśniewska"
        assert data["email"] == "maria@example.com"
        assert "summary" in data
        assert "suggested_response" in data

    def test_get_nonexistent_ticket_returns_404(self, client):
        """GET /tickets/{id} for an unknown UUID must return 404."""
        response = client.get("/tickets/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404

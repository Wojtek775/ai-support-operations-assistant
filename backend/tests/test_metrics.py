"""
tests/test_metrics.py — Tests for GET /metrics endpoint.

Uses the shared `client` fixture from conftest.py (in-memory SQLite, no real AI calls).
Any test that submits a ticket via POST /tickets must patch the AI service —
the same pattern used in test_tickets.py.
"""

from unittest.mock import AsyncMock, patch

# Patch target: name as it exists in the module that CALLS process_ticket_with_ai.
AI_PATCH_TARGET = "app.services.ticket_service.process_ticket_with_ai"

FAKE_AI_RESULT = {
    "classification": "billing",
    "priority": "high",
    "summary": "Customer reports a billing issue.",
    "sentiment": "negative",
    "suggested_response": "We will investigate your billing issue.",
    "recommended_action": "Escalate to billing support.",
}


class TestMetricsEndpoint:
    """Tests for GET /metrics — system aggregation endpoint."""

    def test_metrics_endpoint_returns_200(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_response_contains_all_required_fields(self, client):
        response = client.get("/metrics")
        data = response.json()

        assert "total_tickets" in data
        assert "urgent_tickets" in data
        assert "high_priority_tickets" in data
        assert "negative_sentiment_tickets" in data
        assert "top_category" in data

    def test_metrics_counts_are_integers(self, client):
        response = client.get("/metrics")
        data = response.json()

        assert isinstance(data["total_tickets"], int)
        assert isinstance(data["urgent_tickets"], int)
        assert isinstance(data["high_priority_tickets"], int)
        assert isinstance(data["negative_sentiment_tickets"], int)

    def test_metrics_top_category_is_string_or_none(self, client):
        response = client.get("/metrics")
        data = response.json()

        assert data["top_category"] is None or isinstance(data["top_category"], str)

    def test_metrics_empty_db_returns_zeros(self, client):
        """Fresh in-memory DB from conftest should have zero tickets."""
        response = client.get("/metrics")
        data = response.json()

        assert data["total_tickets"] == 0
        assert data["urgent_tickets"] == 0
        assert data["high_priority_tickets"] == 0
        assert data["negative_sentiment_tickets"] == 0
        assert data["top_category"] is None

    @patch(AI_PATCH_TARGET, new_callable=AsyncMock, return_value=FAKE_AI_RESULT)
    def test_metrics_reflects_submitted_ticket(self, mock_ai, client):
        """After submitting one ticket via the mocked AI, total_tickets should be 1."""
        post_response = client.post(
            "/tickets",
            json={
                "customer_name": "Test User",
                "email": "test@example.com",
                "message": "My account is locked and I cannot access the system.",
            },
        )
        assert post_response.status_code == 201

        response = client.get("/metrics")
        data = response.json()

        assert data["total_tickets"] == 1

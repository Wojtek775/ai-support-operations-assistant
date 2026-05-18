"""
tests/test_evals.py — Unit tests for the AI Evaluation Runner

Tests cover pure logic functions only.
No calls to OpenRouter, no HTTP requests, no database access.

All AI outputs are hardcoded mock dicts.
"""

import json
import tempfile
from pathlib import Path

import pytest

from app.evals.run_evals import (
    calculate_metrics,
    evaluate_prediction,
    load_dataset,
    normalize_sentiment,
    save_results,
)


# ---------------------------------------------------------------------------
# Fixtures — reusable ticket and AI output templates
# ---------------------------------------------------------------------------

def make_ticket(
    ticket_id: str = "SYN-0001",
    category: str = "billing",
    priority: str = "high",
    sentiment: str = "frustrated",
    expected_action: str = "escalate_billing",
    difficulty: str = "easy",
    edge_case=None,
) -> dict:
    """Build a minimal synthetic ticket dict for testing."""
    return {
        "ticket_id": ticket_id,
        "customer_name": "Test User",
        "email": "test@example.com",
        "message": "Test message content.",
        "category": category,
        "priority": priority,
        "sentiment": sentiment,
        "expected_action": expected_action,
        "difficulty": difficulty,
        "language": "en",
        "edge_case": edge_case,
    }


def make_ai_output(
    classification: str = "billing",
    priority: str = "high",
    sentiment: str = "negative",
    recommended_action: str = "escalate_billing",
) -> dict:
    """Build a minimal AI pipeline output dict for testing."""
    return {
        "classification": classification,
        "priority": priority,
        "sentiment": sentiment,
        "recommended_action": recommended_action,
        "summary": "Mock summary.",
        "suggested_response": "Mock response.",
    }


# ---------------------------------------------------------------------------
# normalize_sentiment tests
# ---------------------------------------------------------------------------

class TestNormalizeSentiment:
    """Sentiment normalization maps dataset values to AI output vocabulary."""

    def test_frustrated_maps_to_negative(self):
        assert normalize_sentiment("frustrated") == "negative"

    def test_confused_maps_to_neutral(self):
        assert normalize_sentiment("confused") == "neutral"

    def test_angry_passes_through(self):
        assert normalize_sentiment("angry") == "angry"

    def test_positive_passes_through(self):
        assert normalize_sentiment("positive") == "positive"

    def test_neutral_passes_through(self):
        assert normalize_sentiment("neutral") == "neutral"

    def test_case_insensitive(self):
        assert normalize_sentiment("Frustrated") == "negative"
        assert normalize_sentiment("CONFUSED") == "neutral"

    def test_unknown_value_passes_through_lowercased(self):
        # Unknown values are returned as-is (lowercased)
        assert normalize_sentiment("UNKNOWN") == "unknown"


# ---------------------------------------------------------------------------
# evaluate_prediction tests
# ---------------------------------------------------------------------------

class TestEvaluatePrediction:
    """evaluate_prediction compares AI output against dataset ground truth."""

    def test_perfect_match_all_fields_correct(self):
        ticket = make_ticket(
            category="billing",
            priority="high",
            sentiment="frustrated",  # will be normalized to "negative"
            expected_action="escalate_billing",
        )
        ai_output = make_ai_output(
            classification="billing",
            priority="high",
            sentiment="negative",  # matches normalized "frustrated"
            recommended_action="escalate_billing",
        )
        result = evaluate_prediction(ticket, ai_output)

        assert result["category_match"] is True
        assert result["priority_match"] is True
        assert result["sentiment_match"] is True
        assert result["action_match"] is True
        assert result["all_correct"] is True

    def test_partial_match_two_fields_correct(self):
        ticket = make_ticket(
            category="billing",
            priority="high",
            sentiment="angry",
            expected_action="escalate_billing",
        )
        ai_output = make_ai_output(
            classification="billing",      # correct
            priority="medium",             # wrong
            sentiment="angry",             # correct
            recommended_action="refund_review",  # wrong
        )
        result = evaluate_prediction(ticket, ai_output)

        assert result["category_match"] is True
        assert result["priority_match"] is False
        assert result["sentiment_match"] is True
        assert result["action_match"] is False
        assert result["all_correct"] is False

    def test_zero_match_all_fields_wrong(self):
        ticket = make_ticket(
            category="billing",
            priority="high",
            sentiment="angry",
            expected_action="escalate_billing",
        )
        ai_output = make_ai_output(
            classification="technical_issue",
            priority="low",
            sentiment="positive",
            recommended_action="log_to_product_backlog",
        )
        result = evaluate_prediction(ticket, ai_output)

        assert result["category_match"] is False
        assert result["priority_match"] is False
        assert result["sentiment_match"] is False
        assert result["action_match"] is False
        assert result["all_correct"] is False

    def test_result_contains_expected_metadata_fields(self):
        ticket = make_ticket(ticket_id="SYN-0042", difficulty="hard", edge_case="vague_message")
        ai_output = make_ai_output()
        result = evaluate_prediction(ticket, ai_output)

        assert result["ticket_id"] == "SYN-0042"
        assert result["difficulty"] == "hard"
        assert result["edge_case"] == "vague_message"
        assert "expected_category" in result
        assert "predicted_category" in result

    def test_sentiment_normalization_applied_in_evaluate(self):
        """frustrated in dataset → negative expected → negative AI output = match"""
        ticket = make_ticket(sentiment="frustrated")
        ai_output = make_ai_output(sentiment="negative")
        result = evaluate_prediction(ticket, ai_output)

        assert result["expected_sentiment"] == "negative"
        assert result["sentiment_match"] is True

    def test_confused_normalized_to_neutral_in_evaluate(self):
        """confused in dataset → neutral expected → neutral AI output = match"""
        ticket = make_ticket(sentiment="confused")
        ai_output = make_ai_output(sentiment="neutral")
        result = evaluate_prediction(ticket, ai_output)

        assert result["expected_sentiment"] == "neutral"
        assert result["sentiment_match"] is True

    def test_ai_output_values_are_lowercased_before_comparison(self):
        ticket = make_ticket(category="billing", priority="high")
        ai_output = make_ai_output(classification="BILLING", priority="HIGH")
        result = evaluate_prediction(ticket, ai_output)

        assert result["category_match"] is True
        assert result["priority_match"] is True


# ---------------------------------------------------------------------------
# calculate_metrics tests
# ---------------------------------------------------------------------------

class TestCalculateMetrics:
    """calculate_metrics aggregates per-ticket results into accuracy scores."""

    def _make_prediction(self, all_correct: bool, category: str = "billing",
                         priority: str = "high", difficulty: str = "easy",
                         edge_case=None) -> dict:
        """Build a minimal prediction dict as returned by evaluate_prediction."""
        return {
            "ticket_id": "SYN-0001",
            "difficulty": difficulty,
            "edge_case": edge_case,
            "expected_category": category,
            "expected_priority": priority,
            "expected_sentiment": "negative",
            "expected_action": "escalate_billing",
            "predicted_category": category if all_correct else "wrong",
            "predicted_priority": priority if all_correct else "wrong",
            "predicted_sentiment": "negative" if all_correct else "wrong",
            "predicted_action": "escalate_billing" if all_correct else "wrong",
            "category_match": all_correct,
            "priority_match": all_correct,
            "sentiment_match": all_correct,
            "action_match": all_correct,
            "all_correct": all_correct,
        }

    def test_perfect_accuracy_all_correct(self):
        predictions = [self._make_prediction(True) for _ in range(10)]
        metrics = calculate_metrics(predictions)

        assert metrics["overall_accuracy"] == 1.0
        assert metrics["category_accuracy"] == 1.0
        assert metrics["priority_accuracy"] == 1.0
        assert metrics["sentiment_accuracy"] == 1.0
        assert metrics["expected_action_accuracy"] == 1.0
        assert metrics["total_tickets"] == 10

    def test_zero_accuracy_all_wrong(self):
        predictions = [self._make_prediction(False) for _ in range(5)]
        metrics = calculate_metrics(predictions)

        assert metrics["overall_accuracy"] == 0.0
        assert metrics["category_accuracy"] == 0.0
        assert metrics["total_tickets"] == 5

    def test_fifty_percent_accuracy(self):
        predictions = (
            [self._make_prediction(True) for _ in range(5)]
            + [self._make_prediction(False) for _ in range(5)]
        )
        metrics = calculate_metrics(predictions)

        assert metrics["overall_accuracy"] == 0.5

    def test_accuracy_by_difficulty_computed(self):
        predictions = [
            self._make_prediction(True, difficulty="easy"),
            self._make_prediction(True, difficulty="easy"),
            self._make_prediction(False, difficulty="hard"),
        ]
        metrics = calculate_metrics(predictions)

        assert metrics["accuracy_by_difficulty"]["easy"] == 1.0
        assert metrics["accuracy_by_difficulty"]["hard"] == 0.0

    def test_edge_case_accuracy_computed(self):
        predictions = [
            self._make_prediction(True, edge_case="vague_message"),
            self._make_prediction(False, edge_case="vague_message"),
            self._make_prediction(True),  # non-edge case
        ]
        metrics = calculate_metrics(predictions)

        # 1 correct out of 2 edge cases = 0.5
        assert metrics["edge_case_accuracy"] == 0.5
        assert metrics["accuracy_by_edge_case"]["vague_message"] == 0.5

    def test_empty_predictions_returns_safe_defaults(self):
        metrics = calculate_metrics([])
        assert metrics["total_tickets"] == 0
        assert metrics["overall_accuracy"] == 0.0

    def test_accuracy_by_category_computed(self):
        predictions = [
            self._make_prediction(True, category="billing"),
            self._make_prediction(False, category="billing"),
            self._make_prediction(True, category="technical_issue"),
        ]
        metrics = calculate_metrics(predictions)

        assert metrics["accuracy_by_category"]["billing"] == 0.5
        assert metrics["accuracy_by_category"]["technical_issue"] == 1.0


# ---------------------------------------------------------------------------
# save_results / load_dataset tests
# ---------------------------------------------------------------------------

class TestSaveResults:
    """save_results writes a valid JSON file with the correct structure."""

    def test_save_results_creates_file(self):
        metrics = {
            "total_tickets": 2,
            "overall_accuracy": 0.5,
            "category_accuracy": 1.0,
            "priority_accuracy": 0.5,
            "sentiment_accuracy": 0.5,
            "expected_action_accuracy": 0.5,
            "edge_case_accuracy": 0.0,
            "accuracy_by_category": {"billing": 1.0},
            "accuracy_by_priority": {"high": 0.5},
            "accuracy_by_difficulty": {"easy": 0.5},
            "accuracy_by_edge_case": {},
        }
        per_ticket = [{"ticket_id": "SYN-0001", "all_correct": True}]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_results.json"
            save_results(metrics, per_ticket, output_path)

            assert output_path.exists()
            with open(output_path, encoding="utf-8") as f:
                data = json.load(f)

            assert data["total_tickets"] == 2
            assert data["overall_accuracy"] == 0.5
            assert "meta" in data
            assert "per_ticket_results" in data
            assert len(data["per_ticket_results"]) == 1


class TestLoadDataset:
    """load_dataset reads the actual tickets_dataset.json correctly."""

    def test_load_dataset_returns_list(self):
        dataset = load_dataset()
        assert isinstance(dataset, list)

    def test_load_dataset_has_100_tickets(self):
        dataset = load_dataset()
        assert len(dataset) == 100

    def test_load_dataset_ticket_has_required_fields(self):
        required_fields = {
            "ticket_id", "customer_name", "email", "message",
            "category", "priority", "sentiment", "expected_action",
            "difficulty", "language", "edge_case",
        }
        dataset = load_dataset()
        for ticket in dataset:
            assert required_fields.issubset(set(ticket.keys())), (
                f"Ticket {ticket.get('ticket_id')} missing fields: "
                f"{required_fields - set(ticket.keys())}"
            )

    def test_load_dataset_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_dataset(Path("/nonexistent/path/tickets.json"))

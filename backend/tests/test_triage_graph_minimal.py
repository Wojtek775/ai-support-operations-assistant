"""
tests/test_triage_graph_minimal.py — Minimal LangGraph smoke test.

Verifies that:
1. The graph START -> triage -> END runs without error.
2. MockTriageAdapter output is correctly propagated to state.
3. LLM adapter error leaves classification=None and workflow_status=failed.
4. Two runs with different thread_ids do not share state.

No network calls. No DB. Pure in-memory.
"""

from __future__ import annotations

import os

# Set required env vars before any app import
os.environ.setdefault("OPENROUTER_API_KEY", "test-key-not-real")
os.environ.setdefault("OPENROUTER_MODEL", "test-model")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

import pytest

from app.models.ticket import TicketClassification, TicketPriority, TicketSentiment
from app.workflows.adapters.mock import MockTriageAdapter
from app.workflows.graph import build_triage_only_graph, make_initial_state


# -- helpers -------------------------------------------------------------------

def _config(workflow_id: str) -> dict:
    return {"configurable": {"thread_id": workflow_id}}


async def _run(adapter, workflow_id: str, message: str = "My API keeps returning 500 errors."):
    graph = build_triage_only_graph(llm_adapter=adapter)
    initial = make_initial_state(
        workflow_id=workflow_id,
        customer_name="Test User",
        email="test@example.com",
        message=message,
    )
    result = await graph.ainvoke(initial, config=_config(workflow_id))
    return result


# -- tests ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_minimal_graph_runs_successfully():
    """Happy path: graph completes and triage output is in state."""
    adapter = MockTriageAdapter(
        classification=TicketClassification.TECHNICAL,
        priority=TicketPriority.HIGH,
        sentiment=TicketSentiment.NEGATIVE,
        summary="API returning 500 errors.",
        confidence=0.95,
    )
    result = await _run(adapter, workflow_id="wf-001")

    assert result["classification"] == "technical"
    assert result["priority"] == "high"
    assert result["sentiment"] == "negative"
    assert result["confidence"] == 0.95
    assert result["summary"] == "API returning 500 errors."
    assert result["current_node"] == "triage"
    # execution_trace should have one entry from triage_node
    assert len(result["execution_trace"]) == 1
    assert result["execution_trace"][0]["node"] == "triage"
    assert result["execution_trace"][0]["status"] == "ok"


@pytest.mark.asyncio
async def test_triage_error_sets_failed_status():
    """When LLM adapter raises TriageModelError, workflow_status becomes failed."""
    adapter = MockTriageAdapter(raise_error=True, error_message="OpenRouter timeout")
    result = await _run(adapter, workflow_id="wf-002")

    assert result["workflow_status"] == "failed"
    assert result["classification"] is None
    assert len(result["errors"]) == 1
    assert "triage_node failed" in result["errors"][0]


@pytest.mark.asyncio
async def test_two_runs_have_independent_state():
    """Two workflows with different thread_ids do not share state."""
    adapter_a = MockTriageAdapter(
        classification=TicketClassification.BILLING,
        priority=TicketPriority.HIGH,
        sentiment=TicketSentiment.ANGRY,
        summary="Billing issue.",
        confidence=0.88,
    )
    adapter_b = MockTriageAdapter(
        classification=TicketClassification.GENERAL,
        priority=TicketPriority.LOW,
        sentiment=TicketSentiment.POSITIVE,
        summary="General question.",
        confidence=0.99,
    )

    result_a = await _run(adapter_a, workflow_id="wf-003-a")
    result_b = await _run(adapter_b, workflow_id="wf-003-b")

    assert result_a["classification"] == "billing"
    assert result_b["classification"] == "general"
    assert result_a["workflow_id"] == "wf-003-a"
    assert result_b["workflow_id"] == "wf-003-b"
    # States are independent -- no bleed-through
    assert result_a["priority"] != result_b["priority"]

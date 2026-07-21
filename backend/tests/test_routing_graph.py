"""
tests/test_routing_graph.py — Integration tests for full graph:
    START -> triage -> routing -> conditional -> [END | human_review_gate(interrupt)]

Test scenarios required:
1.  Correct enum values passed to route_ticket()
2.  routing_decision fully populated in state
3.  Normal ticket (low priority, positive) reaches END without interrupt
4.  CRITICAL priority triggers human review
5.  Low confidence triggers human review
6.  Angry sentiment (urgent) triggers human review (escalation=2)
7.  security risk_flag triggers human review
8.  escalation_level >= 2 triggers human review
9.  confidence=None is treated as low confidence (triggers human review)
10. Two different thread_ids do not share state

No network calls. No DB. Pure InMemorySaver.

NOTE on interrupt() behavior:
When using graph.ainvoke() with InMemorySaver, interrupt() does NOT raise
GraphInterrupt to the caller. Instead the result dict contains '__interrupt__'
key. We assert on that key for scenarios 4-9.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENROUTER_API_KEY", "test-key-not-real")
os.environ.setdefault("OPENROUTER_MODEL", "test-model")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

import pytest

from app.models.ticket import TicketClassification, TicketPriority, TicketSentiment
from app.workflows.adapters.mock import MockResolutionAdapter, MockTriageAdapter
from app.workflows.adapters.base import TriageOutput
from app.workflows.graph import build_full_graph, make_initial_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(workflow_id: str) -> dict:
    return {"configurable": {"thread_id": workflow_id}}


def _adapter(
    classification: TicketClassification = TicketClassification.GENERAL,
    priority: TicketPriority = TicketPriority.LOW,
    sentiment: TicketSentiment = TicketSentiment.POSITIVE,
    confidence: float = 0.95,
    risk_flags: list[str] | None = None,
    summary: str = "Standard ticket.",
) -> MockTriageAdapter:
    return MockTriageAdapter(
        classification=classification,
        priority=priority,
        sentiment=sentiment,
        confidence=confidence,
        risk_flags=risk_flags or [],
        summary=summary,
    )


async def _run(adapter, workflow_id: str) -> dict:
    """Run full graph; return the result dict."""
    graph = build_full_graph(
        llm_adapter=adapter,
        resolution_adapter=MockResolutionAdapter(),
    )
    initial = make_initial_state(
        workflow_id=workflow_id,
        customer_name="Test User",
        email="test@example.com",
        message="Help me please.",
    )
    return await graph.ainvoke(initial, config=_config(workflow_id))


def _was_interrupted(result: dict) -> bool:
    """
    Check if LangGraph paused via interrupt().
    With ainvoke + InMemorySaver, interrupt() does NOT raise an exception —
    instead the result contains the '__interrupt__' key.
    """
    return "__interrupt__" in result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_routing_receives_correct_enum_values():
    """
    Scenario 1: route_ticket() is called with .value strings from enums.
    We use billing/medium/neutral (no escalation) so no interrupt fires.
    Verify the state contains the correctly mapped string values.
    """
    adapter = _adapter(
        classification=TicketClassification.BILLING,
        priority=TicketPriority.MEDIUM,
        sentiment=TicketSentiment.NEUTRAL,
        confidence=0.95,
    )
    result = await _run(adapter, "wf-r-001")

    # triage node must have written .value strings (not Enum objects)
    assert result["classification"] == "billing"
    assert result["priority"] == "medium"
    assert result["sentiment"] == "neutral"

    rd = result["routing_decision"]
    assert rd is not None
    # billing -> finance_team per workflow_service rules
    assert rd["assigned_team"] == "finance_team"
    assert rd["sla_hours"] == 4
    # medium/neutral -> no escalation, no interrupt
    assert rd["requires_human_review"] is False
    assert rd["escalation_level"] == 0
    assert not _was_interrupted(result)


@pytest.mark.asyncio
async def test_routing_decision_fully_populated():
    """
    Scenario 2: routing_decision dict contains all expected keys.
    Uses a safe ticket (general/low/positive) that does NOT trigger interrupt.
    """
    adapter = _adapter()  # general, low, positive, confidence=0.95
    result = await _run(adapter, "wf-r-002")

    rd = result["routing_decision"]
    assert rd is not None
    assert "assigned_team" in rd
    assert "sla_hours" in rd
    assert "requires_human_review" in rd
    assert "escalation_level" in rd
    assert "internal_notes" in rd
    assert rd["requires_human_review"] is False
    assert rd["escalation_level"] == 0
    assert rd["assigned_team"] == "support_general"
    assert not _was_interrupted(result)


@pytest.mark.asyncio
async def test_normal_ticket_reaches_end_without_interrupt():
    """
    Scenario 3: general/low/positive/confidence=0.95 ticket reaches END.
    No interrupt. Both triage and routing nodes appear in execution_trace.
    human_review_gate must NOT appear.
    """
    adapter = _adapter()  # general, low, positive, high confidence
    result = await _run(adapter, "wf-r-003")

    assert result["routing_decision"] is not None
    assert result["routing_decision"]["requires_human_review"] is False
    assert not _was_interrupted(result)

    # execution_trace has triage + routing (human_review_gate NOT reached)
    nodes_visited = [e["node"] for e in result["execution_trace"]]
    assert "triage" in nodes_visited
    assert "routing" in nodes_visited
    assert "human_review_gate" not in nodes_visited


@pytest.mark.asyncio
async def test_critical_priority_triggers_human_review():
    """
    Scenario 4: CRITICAL priority -> human_review_gate -> interrupt().
    _needs_human_review returns True for priority=='critical'.
    Graph result contains '__interrupt__' key.
    """
    adapter = _adapter(
        classification=TicketClassification.TECHNICAL,
        priority=TicketPriority.CRITICAL,
        sentiment=TicketSentiment.NEUTRAL,
        confidence=0.95,  # above threshold — only critical triggers
    )
    result = await _run(adapter, "wf-r-004")
    assert _was_interrupted(result), (
        f"Expected interrupt for CRITICAL priority. Result keys: {list(result.keys())}"
    )


@pytest.mark.asyncio
async def test_low_confidence_triggers_human_review():
    """
    Scenario 5: confidence=0.40 (below 0.60 threshold) triggers interrupt.
    """
    adapter = _adapter(
        classification=TicketClassification.GENERAL,
        priority=TicketPriority.LOW,
        sentiment=TicketSentiment.NEUTRAL,
        confidence=0.40,  # below threshold
    )
    result = await _run(adapter, "wf-r-005")
    assert _was_interrupted(result), (
        f"Expected interrupt for low confidence=0.40. Result keys: {list(result.keys())}"
    )


@pytest.mark.asyncio
async def test_angry_sentiment_triggers_human_review():
    """
    Scenario 6: ANGRY sentiment with URGENT priority triggers interrupt.
    (angry + urgent -> escalation_level=2, routing requires_human_review=True)
    """
    adapter = _adapter(
        classification=TicketClassification.GENERAL,
        priority=TicketPriority.URGENT,
        sentiment=TicketSentiment.ANGRY,
        confidence=0.95,
    )
    result = await _run(adapter, "wf-r-006")
    assert _was_interrupted(result), (
        f"Expected interrupt for ANGRY+URGENT. Result keys: {list(result.keys())}"
    )


@pytest.mark.asyncio
async def test_security_risk_flag_triggers_human_review():
    """
    Scenario 7: 'security' in risk_flags -> _needs_human_review returns True.
    """
    adapter = _adapter(
        classification=TicketClassification.TECHNICAL,
        priority=TicketPriority.MEDIUM,
        sentiment=TicketSentiment.NEUTRAL,
        confidence=0.95,
        risk_flags=["security"],
    )
    result = await _run(adapter, "wf-r-007")
    assert _was_interrupted(result), (
        f"Expected interrupt for security risk_flag. Result keys: {list(result.keys())}"
    )


@pytest.mark.asyncio
async def test_escalation_level_2_triggers_human_review():
    """
    Scenario 8: angry + urgent -> escalation_level=2 -> interrupt.
    routing_decision.requires_human_review=True, escalation_level=2.
    """
    adapter = _adapter(
        classification=TicketClassification.GENERAL,
        priority=TicketPriority.URGENT,
        sentiment=TicketSentiment.ANGRY,
        confidence=0.95,
    )
    result = await _run(adapter, "wf-r-008")
    assert _was_interrupted(result), (
        f"Expected interrupt for escalation_level=2. Result keys: {list(result.keys())}"
    )
    # Also verify routing_decision reflects escalation_level=2
    rd = result.get("routing_decision") or {}
    assert rd.get("escalation_level", 0) >= 2


@pytest.mark.asyncio
async def test_none_confidence_triggers_human_review():
    """
    Scenario 9: confidence=None in state is treated as low confidence -> interrupt.
    We use model_construct() to bypass Pydantic validation and inject None,
    simulating the real-world case where the LLM omits confidence entirely.
    """
    # model_construct skips Pydantic validation — lets us set confidence=None
    output_no_conf = TriageOutput.model_construct(
        classification=TicketClassification.GENERAL,
        priority=TicketPriority.LOW,
        sentiment=TicketSentiment.NEUTRAL,
        summary="No confidence ticket.",
        confidence=None,
        risk_flags=[],
    )

    adapter_none_conf = MockTriageAdapter(
        classification=TicketClassification.GENERAL,
        priority=TicketPriority.LOW,
        sentiment=TicketSentiment.NEUTRAL,
        confidence=0.95,
        summary="placeholder",
    )
    # Override the pre-built output with the None-confidence version
    adapter_none_conf._output = output_no_conf

    result = await _run(adapter_none_conf, "wf-r-009")
    assert _was_interrupted(result), (
        f"Expected interrupt for confidence=None. Result keys: {list(result.keys())}"
    )


@pytest.mark.asyncio
async def test_two_thread_ids_do_not_share_state():
    """
    Scenario 10: Two workflows with different thread_ids have independent state.
    """
    adapter_a = _adapter(
        classification=TicketClassification.BILLING,
        priority=TicketPriority.LOW,
        sentiment=TicketSentiment.POSITIVE,
        confidence=0.99,
        summary="Billing question.",
    )
    adapter_b = _adapter(
        classification=TicketClassification.GENERAL,
        priority=TicketPriority.LOW,
        sentiment=TicketSentiment.POSITIVE,
        confidence=0.99,
        summary="General question.",
    )

    result_a = await _run(adapter_a, "wf-r-010-a")
    result_b = await _run(adapter_b, "wf-r-010-b")

    assert result_a["workflow_id"] == "wf-r-010-a"
    assert result_b["workflow_id"] == "wf-r-010-b"
    assert result_a["classification"] == "billing"
    assert result_b["classification"] == "general"
    # Both should reach END without interrupt (low/positive/high confidence)
    assert not _was_interrupted(result_a)
    assert not _was_interrupted(result_b)
    # routing decisions are independent
    assert result_a["routing_decision"]["assigned_team"] == "finance_team"
    assert result_b["routing_decision"]["assigned_team"] == "support_general"

"""
tests/test_supervisor.py — Tests for the Supervisor Agent (ETAP 5).

Coverage:
  1. SupervisorNode unit tests (with MockSupervisorAdapter)
     - happy path: first call returns "triage" when no agents completed
     - routing step: returns "routing" after triage done
     - FINISH: returns FINISH after all agents done
     - max iterations guard: supervisor_iterations >= 10 → human_review
  2. MockSupervisorAdapter unit tests
     - smart mode decision logic
     - fixed sequence mode
     - call_args_list recording
  3. build_supervisor_graph integration test
     - full happy path via supervisor graph (all mocks, no LLM)
     - graph completes without interrupts
     - execution_trace contains supervisor entries
  4. SupervisorOutput / base.py model tests
     - valid output accepted
     - invalid next_agent rejected by Pydantic
     - reasoning length validation
"""

from __future__ import annotations

import uuid

import pytest

from app.workflows.adapters.base import SupervisorOutput
from app.workflows.adapters.mock import (
    MockResolutionAdapter,
    MockReviewerAdapter,
    MockSupervisorAdapter,
    MockTriageAdapter,
)
from app.workflows.graph import build_supervisor_graph, make_initial_state
from app.workflows.nodes.supervisor import MAX_SUPERVISOR_ITERATIONS, SupervisorNode


# ---------------------------------------------------------------------------
# Helper: minimal WorkflowState dict for unit tests
# ---------------------------------------------------------------------------

def _minimal_state(
    workflow_id: str | None = None,
    completed_nodes: list[str] | None = None,
    supervisor_iterations: int = 0,
    review_decision: str | None = None,
    routing_decision: dict | None = None,
) -> dict:
    """Build a minimal state dict for SupervisorNode unit tests."""
    trace = [
        {"node": node, "status": "ok"}
        for node in (completed_nodes or [])
    ]
    return {
        "workflow_id": workflow_id or str(uuid.uuid4()),
        "customer_name": "Test User",
        "email": "test@example.com",
        "message": "My account is broken.",
        "classification": "technical" if "triage" in (completed_nodes or []) else None,
        "priority": "medium" if "triage" in (completed_nodes or []) else None,
        "sentiment": "neutral" if "triage" in (completed_nodes or []) else None,
        "confidence": 0.92 if "triage" in (completed_nodes or []) else None,
        "routing_decision": routing_decision,
        "suggested_response": "We'll help you." if "resolution" in (completed_nodes or []) else None,
        "recommended_action": "Assign to tech team." if "resolution" in (completed_nodes or []) else None,
        "review_decision": review_decision,
        "review_issues": [],
        "workflow_status": "running",
        "execution_trace": trace,
        "supervisor_iterations": supervisor_iterations,
        "supervisor_next": None,
        "supervisor_reasoning": None,
    }


# ---------------------------------------------------------------------------
# 1. SupervisorNode unit tests
# ---------------------------------------------------------------------------

class TestSupervisorNode:
    """Unit tests for SupervisorNode.__call__()."""

    @pytest.mark.asyncio
    async def test_first_call_returns_triage(self):
        """On first call (no agents completed), mock returns 'triage'."""
        adapter = MockSupervisorAdapter()
        node = SupervisorNode(adapter)
        state = _minimal_state(completed_nodes=[])

        result = await node(state)

        assert result["supervisor_next"] == "triage"
        assert result["supervisor_iterations"] == 1
        assert result["current_node"] == "supervisor"
        assert len(result["execution_trace"]) == 1

    @pytest.mark.asyncio
    async def test_after_triage_returns_routing(self):
        """After triage completes, supervisor should pick routing."""
        adapter = MockSupervisorAdapter()
        node = SupervisorNode(adapter)
        state = _minimal_state(completed_nodes=["triage"])

        result = await node(state)

        assert result["supervisor_next"] == "routing"

    @pytest.mark.asyncio
    async def test_after_all_agents_returns_finish(self):
        """After all agents complete with APPROVED, supervisor returns FINISH."""
        adapter = MockSupervisorAdapter()
        node = SupervisorNode(adapter)
        state = _minimal_state(
            completed_nodes=["triage", "routing", "resolution", "reviewer"],
            review_decision="APPROVED",
        )

        result = await node(state)

        assert result["supervisor_next"] == "FINISH"

    @pytest.mark.asyncio
    async def test_max_iterations_guard(self):
        """
        When supervisor_iterations >= MAX_SUPERVISOR_ITERATIONS,
        the node bypasses the adapter and routes to human_review.
        """
        adapter = MockSupervisorAdapter()
        node = SupervisorNode(adapter)
        state = _minimal_state(
            supervisor_iterations=MAX_SUPERVISOR_ITERATIONS,
            completed_nodes=["triage"],
        )

        result = await node(state)

        assert result["supervisor_next"] == "human_review"
        assert result["supervisor_iterations"] == MAX_SUPERVISOR_ITERATIONS + 1
        # Adapter should NOT have been called (check call_args_list is empty)
        assert len(adapter.call_args_list) == 0

    @pytest.mark.asyncio
    async def test_execution_trace_entry_structure(self):
        """Trace entry should contain required keys."""
        adapter = MockSupervisorAdapter()
        node = SupervisorNode(adapter)
        state = _minimal_state(completed_nodes=[])

        result = await node(state)

        trace = result["execution_trace"]
        assert len(trace) == 1
        entry = trace[0]
        assert entry["node"] == "supervisor"
        assert "next_agent" in entry
        assert "reasoning" in entry
        assert "iteration" in entry
        assert entry["status"] == "ok"

    @pytest.mark.asyncio
    async def test_requires_human_review_flag(self):
        """If routing_decision.requires_human_review is True, supervisor routes to human_review."""
        adapter = MockSupervisorAdapter()
        node = SupervisorNode(adapter)
        routing = {"requires_human_review": True, "assigned_team": "human", "sla_hours": 2}
        state = _minimal_state(
            completed_nodes=["triage", "routing"],
            routing_decision=routing,
        )

        result = await node(state)

        assert result["supervisor_next"] == "human_review"


# ---------------------------------------------------------------------------
# 2. MockSupervisorAdapter unit tests
# ---------------------------------------------------------------------------

class TestMockSupervisorAdapter:
    """Unit tests for MockSupervisorAdapter (smart mode + sequence mode)."""

    @pytest.mark.asyncio
    async def test_smart_mode_happy_path_sequence(self):
        """
        Smart mode should follow: triage → routing → resolution → reviewer → FINISH.
        """
        adapter = MockSupervisorAdapter()

        # Simulate the state at each step
        states_and_expected = [
            ([], None, None, "triage"),
            (["triage"], None, None, "routing"),
            (["triage", "routing"], None, None, "resolution"),
            (["triage", "routing", "resolution"], None, None, "reviewer"),
            (["triage", "routing", "resolution", "reviewer"], "APPROVED", None, "FINISH"),
        ]

        for completed, review_decision, routing, expected in states_and_expected:
            result = await adapter.supervise(
                message="test",
                customer_name="User",
                completed_agents=completed,
                state_summary={
                    "review_decision": review_decision,
                    "routing_decision": routing,
                },
            )
            assert result.next_agent == expected, (
                f"Expected {expected} after completing {completed}, got {result.next_agent}"
            )

    @pytest.mark.asyncio
    async def test_fixed_sequence_mode(self):
        """Fixed sequence mode replays the provided list exactly."""
        seq = ["triage", "routing", "resolution", "FINISH"]
        adapter = MockSupervisorAdapter(sequence=seq)

        for expected in seq:
            result = await adapter.supervise(
                message="x",
                customer_name="U",
                completed_agents=[],
                state_summary={},
            )
            assert result.next_agent == expected

        # After exhausting sequence, returns FINISH
        result = await adapter.supervise(
            message="x",
            customer_name="U",
            completed_agents=[],
            state_summary={},
        )
        assert result.next_agent == "FINISH"

    @pytest.mark.asyncio
    async def test_call_args_recording(self):
        """call_args_list should record every call."""
        adapter = MockSupervisorAdapter()

        await adapter.supervise(
            message="broken login",
            customer_name="Alice",
            completed_agents=["triage"],
            state_summary={"classification": "technical"},
        )

        assert len(adapter.call_args_list) == 1
        call = adapter.call_args_list[0]
        assert call["message"] == "broken login"
        assert call["customer_name"] == "Alice"
        assert call["completed_agents"] == ["triage"]
        assert call["state_summary"]["classification"] == "technical"

    @pytest.mark.asyncio
    async def test_retry_routes_back_to_resolution(self):
        """If review_decision is RETRY, adapter returns resolution."""
        adapter = MockSupervisorAdapter()
        result = await adapter.supervise(
            message="x",
            customer_name="U",
            completed_agents=["triage", "routing", "resolution"],
            state_summary={"review_decision": "RETRY"},
        )
        assert result.next_agent == "resolution"

    @pytest.mark.asyncio
    async def test_human_review_escalation(self):
        """If review_decision is HUMAN_REVIEW, adapter escalates."""
        adapter = MockSupervisorAdapter()
        result = await adapter.supervise(
            message="x",
            customer_name="U",
            completed_agents=["triage", "routing", "resolution"],
            state_summary={"review_decision": "HUMAN_REVIEW"},
        )
        assert result.next_agent == "human_review"


# ---------------------------------------------------------------------------
# 3. SupervisorOutput / Pydantic model tests
# ---------------------------------------------------------------------------

class TestSupervisorOutputModel:
    """Tests for SupervisorOutput Pydantic validation."""

    def test_valid_output_triage(self):
        out = SupervisorOutput(next_agent="triage", reasoning="Starting with triage.")
        assert out.next_agent == "triage"

    def test_valid_output_finish(self):
        out = SupervisorOutput(next_agent="FINISH", reasoning="All done.")
        assert out.next_agent == "FINISH"

    def test_invalid_next_agent_raises(self):
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            SupervisorOutput(next_agent="unknown_agent", reasoning="Bad agent.")

    def test_reasoning_too_short_raises(self):
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            SupervisorOutput(next_agent="triage", reasoning="Hi")  # < 5 chars

    def test_reasoning_too_long_raises(self):
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            SupervisorOutput(next_agent="triage", reasoning="x" * 501)  # > 500 chars


# ---------------------------------------------------------------------------
# 4. build_supervisor_graph integration tests
# ---------------------------------------------------------------------------

class TestSupervisorGraphIntegration:
    """End-to-end integration tests using build_supervisor_graph with all mocks."""

    @pytest.mark.asyncio
    async def test_happy_path_completes(self):
        """
        Full supervisor graph happy path.
        MockSupervisorAdapter (smart mode) → triage → routing → resolution → reviewer → FINISH.
        MockReviewerAdapter returns APPROVED → workflow completes cleanly.
        """
        supervisor_adapter = MockSupervisorAdapter()
        triage_adapter = MockTriageAdapter()
        resolution_adapter = MockResolutionAdapter()
        reviewer_adapter = MockReviewerAdapter(decision="APPROVED")

        graph = build_supervisor_graph(
            supervisor_adapter=supervisor_adapter,
            llm_adapter=triage_adapter,
            resolution_adapter=resolution_adapter,
            reviewer_adapter=reviewer_adapter,
        )

        initial_state = make_initial_state(
            workflow_id=str(uuid.uuid4()),
            customer_name="Bob",
            email="bob@example.com",
            message="My invoice is wrong.",
        )

        config = {"configurable": {"thread_id": initial_state["workflow_id"]}}
        final_state = await graph.ainvoke(initial_state, config=config)

        # Supervisor ran multiple times
        assert final_state["supervisor_iterations"] >= 4  # at least 4 cycles
        assert final_state["supervisor_next"] == "FINISH"
        # All key agents ran
        node_names = [e["node"] for e in final_state.get("execution_trace", [])]
        assert "triage" in node_names
        assert "routing" in node_names
        assert "resolution" in node_names
        assert "reviewer" in node_names
        assert "supervisor" in node_names
        # Triage results populated
        assert final_state["classification"] is not None
        assert final_state["suggested_response"] is not None
        assert final_state["review_decision"] == "APPROVED"

    @pytest.mark.asyncio
    async def test_human_review_path(self):
        """
        When the supervisor explicitly routes to human_review,
        the human_review_gate node fires interrupt() and the graph pauses.

        LangGraph's ainvoke() does NOT raise GraphInterrupt — instead it returns
        a dict with "__interrupt__" key (same pattern as other tests in
        test_routing_boundary.py and test_reviewer.py).

        We use a fixed sequence adapter that explicitly includes 'human_review'
        to guarantee the human_review_gate node is reached.
        """
        # Fixed sequence: go to triage, routing, then human_review (skip resolution)
        supervisor_adapter = MockSupervisorAdapter(
            sequence=["triage", "routing", "human_review"]
        )
        triage_adapter = MockTriageAdapter(
            priority="urgent",
            confidence=0.45,
        )
        resolution_adapter = MockResolutionAdapter()
        reviewer_adapter = MockReviewerAdapter(decision="APPROVED")

        graph = build_supervisor_graph(
            supervisor_adapter=supervisor_adapter,
            llm_adapter=triage_adapter,
            resolution_adapter=resolution_adapter,
            reviewer_adapter=reviewer_adapter,
        )

        initial_state = make_initial_state(
            workflow_id=str(uuid.uuid4()),
            customer_name="Carol",
            email="carol@example.com",
            message="URGENT: System is completely down!",
        )

        config = {"configurable": {"thread_id": initial_state["workflow_id"]}}
        result = await graph.ainvoke(initial_state, config=config)

        # LangGraph stores interrupt signal in "__interrupt__" key
        assert "__interrupt__" in result, (
            f"Expected '__interrupt__' in result but got keys: {list(result.keys())}"
        )
        # Interrupt payload is non-empty
        assert len(result["__interrupt__"]) > 0
        # Resolution was NOT called (supervisor routed directly to human_review)
        assert len(resolution_adapter.call_args_list) == 0
        # Reviewer was NOT called
        assert len(reviewer_adapter.call_args_list) == 0

    @pytest.mark.asyncio
    async def test_supervisor_adapter_called_multiple_times(self):
        """Supervisor adapter is called once per cycle (triage/routing/resolution/reviewer/FINISH)."""
        supervisor_adapter = MockSupervisorAdapter()
        triage_adapter = MockTriageAdapter()
        resolution_adapter = MockResolutionAdapter()
        reviewer_adapter = MockReviewerAdapter(decision="APPROVED")

        graph = build_supervisor_graph(
            supervisor_adapter=supervisor_adapter,
            llm_adapter=triage_adapter,
            resolution_adapter=resolution_adapter,
            reviewer_adapter=reviewer_adapter,
        )

        initial_state = make_initial_state(
            workflow_id=str(uuid.uuid4()),
            customer_name="Dave",
            email="dave@example.com",
            message="Password reset not working.",
        )

        config = {"configurable": {"thread_id": initial_state["workflow_id"]}}
        await graph.ainvoke(initial_state, config=config)

        # Supervisor called at least 5 times: initial + after each of 4 agents
        assert len(supervisor_adapter.call_args_list) >= 5

    @pytest.mark.asyncio
    async def test_fixed_sequence_graph(self):
        """
        Using a fixed sequence adapter ensures predictable node ordering.
        Sequence: triage → routing → resolution → reviewer → FINISH
        """
        sequence = ["triage", "routing", "resolution", "reviewer", "FINISH"]
        supervisor_adapter = MockSupervisorAdapter(sequence=sequence)
        triage_adapter = MockTriageAdapter()
        resolution_adapter = MockResolutionAdapter()
        reviewer_adapter = MockReviewerAdapter(decision="APPROVED")

        graph = build_supervisor_graph(
            supervisor_adapter=supervisor_adapter,
            llm_adapter=triage_adapter,
            resolution_adapter=resolution_adapter,
            reviewer_adapter=reviewer_adapter,
        )

        initial_state = make_initial_state(
            workflow_id=str(uuid.uuid4()),
            customer_name="Eve",
            email="eve@example.com",
            message="My order is missing.",
        )

        config = {"configurable": {"thread_id": initial_state["workflow_id"]}}
        final_state = await graph.ainvoke(initial_state, config=config)

        assert final_state["classification"] is not None
        assert final_state["suggested_response"] is not None
        assert final_state["review_decision"] == "APPROVED"

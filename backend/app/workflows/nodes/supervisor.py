"""
nodes/supervisor.py — Supervisor Node for LangGraph multi-agent workflow.

The Supervisor is the central coordinator of the multi-agent graph.
It runs in a loop: after each specialist agent completes, control
returns to the Supervisor, which decides who should run next or
signals FINISH to end the workflow.

Pattern: Supervisor → Specialist → Supervisor → ... → FINISH

Advantages over a fixed pipeline:
  - Dynamic agent selection based on ticket state
  - LLM can skip or repeat agents based on context
  - Reasoning stored in execution_trace for full observability
  - Graceful fallback to human_review when confidence is low

The Supervisor does NOT modify ticket classification, response, or
routing — it only controls the flow between specialist agents.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.utils.logger import logger
from app.workflows.adapters.base import SupervisorAdapter
from app.workflows.state import WorkflowState

# Maximum iterations to prevent infinite loops
MAX_SUPERVISOR_ITERATIONS = 10


class SupervisorNode:
    """
    Supervisor Agent — orchestrates specialist agents in a loop.

    On each invocation:
      1. Extracts current state summary (completed agents, results so far)
      2. Calls SupervisorAdapter.supervise() to get the next agent decision
      3. Returns partial state update: supervisor_next + supervisor_reasoning

    The conditional edge in the graph reads supervisor_next and routes to
    the appropriate specialist node or END.

    Adapter injected via constructor (dependency injection / testability).
    """

    def __init__(self, supervisor_adapter: SupervisorAdapter) -> None:
        self._adapter = supervisor_adapter

    async def __call__(self, state: WorkflowState) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        iterations = state.get("supervisor_iterations", 0)

        # Safety: prevent infinite loops
        if iterations >= MAX_SUPERVISOR_ITERATIONS:
            logger.warning(
                f"[supervisor_node] workflow_id={state['workflow_id']} "
                f"max iterations ({MAX_SUPERVISOR_ITERATIONS}) reached — routing to human_review"
            )
            trace_entry = {
                "node": "supervisor",
                "started_at": now,
                "finished_at": now,
                "status": "max_iterations",
                "iteration": iterations,
                "next_agent": "human_review",
                "reasoning": f"Max supervisor iterations ({MAX_SUPERVISOR_ITERATIONS}) reached. Escalating to human review.",
            }
            return {
                "supervisor_next": "human_review",
                "supervisor_reasoning": trace_entry["reasoning"],
                "supervisor_iterations": iterations + 1,
                "current_node": "supervisor",
                "execution_trace": state.get("execution_trace", []) + [trace_entry],
                "updated_at": now,
            }

        # Build list of agents that have already run (from execution_trace)
        completed_agents: list[str] = [
            entry["node"]
            for entry in state.get("execution_trace", [])
            if entry.get("node") not in ("supervisor",) and entry.get("status") != "max_iterations"
        ]

        # Build a concise state summary for the supervisor LLM
        state_summary = {
            "classification": state.get("classification"),
            "priority": state.get("priority"),
            "sentiment": state.get("sentiment"),
            "confidence": state.get("confidence"),
            "routing_decision": state.get("routing_decision"),
            "suggested_response": bool(state.get("suggested_response")),  # don't send full text
            "recommended_action": bool(state.get("recommended_action")),
            "review_decision": state.get("review_decision"),
            "review_issues": state.get("review_issues", []),
            "workflow_status": state.get("workflow_status"),
            "requires_human_review": (state.get("routing_decision") or {}).get(
                "requires_human_review", False
            ),
        }

        logger.info(
            f"[supervisor_node] workflow_id={state['workflow_id']} "
            f"iteration={iterations} completed={completed_agents}"
        )

        result = await self._adapter.supervise(
            message=state["message"],
            customer_name=state["customer_name"],
            completed_agents=completed_agents,
            state_summary=state_summary,
        )

        finished_at = datetime.now(timezone.utc).isoformat()
        trace_entry = {
            "node": "supervisor",
            "started_at": now,
            "finished_at": finished_at,
            "status": "ok",
            "iteration": iterations,
            "next_agent": result.next_agent,
            "reasoning": result.reasoning,
            "completed_agents": completed_agents,
        }

        logger.info(
            f"[supervisor_node] iteration={iterations} "
            f"next={result.next_agent} reasoning='{result.reasoning[:60]}...'"
            if len(result.reasoning) > 60
            else f"[supervisor_node] iteration={iterations} "
            f"next={result.next_agent} reasoning='{result.reasoning}'"
        )

        return {
            "supervisor_next": result.next_agent,
            "supervisor_reasoning": result.reasoning,
            "supervisor_iterations": iterations + 1,
            "current_node": "supervisor",
            "execution_trace": state.get("execution_trace", []) + [trace_entry],
            "updated_at": finished_at,
        }

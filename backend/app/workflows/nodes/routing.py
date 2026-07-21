"""
nodes/routing.py — Deterministic Routing Node.

Wraps the existing workflow_service.route_ticket() function.
No LLM calls. Pure Python business logic stays in Python.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.services.workflow_service import route_ticket
from app.utils.logger import logger
from app.workflows.state import WorkflowState


async def routing_node(state: WorkflowState) -> dict:
    """
    Apply deterministic routing rules based on triage output.
    Returns partial state with routing_decision populated.

    If triage failed (workflow_status == "failed"), this node is a no-op.
    The graph's conditional edge will skip it, but defensive guard is here too.
    """
    if state.get("workflow_status") == "failed":
        return {}

    now = datetime.now(timezone.utc).isoformat()
    logger.info(
        f"[routing_node] workflow_id={state['workflow_id']} "
        f"cls={state.get('classification')} pri={state.get('priority')}"
    )

    decision = route_ticket(
        classification=state["classification"] or "general",
        priority=state["priority"] or "medium",
        sentiment=state["sentiment"] or "neutral",
    )

    trace_entry = {
        "node": "routing",
        "started_at": now,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "assigned_team": decision.assigned_team,
        "escalation_level": decision.escalation_level,
    }

    routing_dict = {
        "assigned_team": decision.assigned_team,
        "sla_hours": decision.sla_hours,
        "requires_human_review": decision.requires_human_review,
        "escalation_level": decision.escalation_level,
        "internal_notes": decision.internal_notes,
    }

    logger.info(
        f"[routing_node] team={decision.assigned_team} sla={decision.sla_hours}h "
        f"escalation={decision.escalation_level}"
    )

    return {
        "routing_decision": routing_dict,
        "current_node": "routing",
        "execution_trace": state.get("execution_trace", []) + [trace_entry],
        "updated_at": trace_entry["finished_at"],
    }

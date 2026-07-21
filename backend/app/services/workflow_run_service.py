"""
services/workflow_run_service.py — Application service for workflow persistence.

Orchestrates the full lifecycle:

    1. create_running()         — persist WorkflowRunRecord(status=running)
    2. run_graph()              — invoke LangGraph full graph
    3. finalize()               — interpret __interrupt__ / errors → update record
       - completed              — also creates TicketRecord in same transaction
                                  using REAL suggested_response / recommended_action
                                  from ResolutionNode (not placeholders)
       - waiting_review         — no TicketRecord yet; Resolution not called
       - failed                 — records error_code + error_message + failed_node

Design rules:
    - NO HTTPException here — that belongs in the router
    - Node functions are independent of SQLAlchemy (injected via adapters)
    - Single commit boundary: TicketRecord + WorkflowRunRecord update share one
      transaction for completed workflows
    - State serialisation is explicit and typed (no json.dumps(..., default=str))
    - completed workflow MUST have non-empty suggested_response / recommended_action
      sourced from ResolutionNode — "pending_resolution" is only a defensive fallback
      for historical or incomplete states loaded from DB
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import settings
from app.models.db_models import TicketRecord, WorkflowRunRecord
from app.models.workflow import WorkflowResponse
from app.utils.logger import logger
from app.workflows.adapters.base import LLMAdapter, ResolutionAdapter, ReviewerAdapter
from app.workflows.graph import build_full_graph, make_initial_state


# ---------------------------------------------------------------------------
# Custom exceptions — raised by service, mapped to HTTP codes by router
# ---------------------------------------------------------------------------

class WorkflowPersistenceError(Exception):
    """Raised when a DB write fails (create or update of WorkflowRunRecord)."""


class WorkflowNotFoundError(Exception):
    """Raised when GET /workflows/{id} finds no record."""


# ---------------------------------------------------------------------------
# State serialisation helpers
# ---------------------------------------------------------------------------

def _serialize_value(v: Any) -> Any:
    """
    Recursively convert a single value to a JSON-serialisable form.

    Conversion rules:
        Enum        → .value  (string)
        datetime    → .isoformat()
        dataclass   → dict (then recurse)
        Pydantic    → model_dump(mode="json")
        dict        → recurse values
        list/tuple  → recurse items
        None/scalar → as-is
    """
    if v is None:
        return None
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, datetime):
        return v.isoformat()
    if is_dataclass(v) and not isinstance(v, type):
        return _serialize_value(asdict(v))
    if isinstance(v, BaseModel):
        return v.model_dump(mode="json")
    if isinstance(v, dict):
        return {k: _serialize_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_serialize_value(item) for item in v]
    # str, int, float, bool → pass through
    return v


def serialize_state(state: dict) -> str:
    """
    Serialise a WorkflowState dict to a JSON string.

    Raises ValueError if the result is not round-trip JSON-decodable
    (i.e. json.loads(result) must succeed without error).
    """
    cleaned = _serialize_value(state)
    result = json.dumps(cleaned, ensure_ascii=False)
    # Verify round-trip is possible
    json.loads(result)
    return result


def deserialize_state(state_json: str) -> dict:
    """Deserialise the stored JSON string back to a plain dict."""
    return json.loads(state_json)


# ---------------------------------------------------------------------------
# Helpers to build WorkflowResponse from ORM record
# ---------------------------------------------------------------------------

def _record_to_response(record: WorkflowRunRecord) -> WorkflowResponse:
    """Convert a WorkflowRunRecord ORM object to a WorkflowResponse."""
    state: dict = {}
    if record.state_json:
        try:
            state = deserialize_state(record.state_json)
        except Exception:
            pass

    rd = state.get("routing_decision") or {}

    return WorkflowResponse(
        workflow_id=record.workflow_id,
        ticket_id=record.ticket_id,
        status=record.status,
        classification=state.get("classification"),
        priority=state.get("priority"),
        sentiment=state.get("sentiment"),
        summary=state.get("summary"),
        confidence=state.get("confidence"),
        assigned_team=rd.get("assigned_team"),
        sla_hours=rd.get("sla_hours"),
        escalation_level=rd.get("escalation_level"),
        requires_human_review=record.requires_human_review,
        human_review_status=record.human_review_status,
        started_at=record.started_at,
        updated_at=record.updated_at,
        completed_at=record.completed_at,
        error_code=record.error_code,
        error_message=record.error_message,
    )


# ---------------------------------------------------------------------------
# Main application service
# ---------------------------------------------------------------------------

class WorkflowRunService:
    """
    Orchestrates the graph invocation and persistence lifecycle.

    Constructor arguments:
        llm_adapter         — injected Triage LLM adapter
        resolution_adapter  — injected Resolution LLM adapter
        db                  — SQLAlchemy Session

    Usage:
        service = WorkflowRunService(
            llm_adapter=triage_adapter,
            resolution_adapter=resolution_adapter,
            db=db,
        )
        response = await service.run(...)
    """

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        db: Session,
        resolution_adapter: ResolutionAdapter | None = None,
        reviewer_adapter: ReviewerAdapter | None = None,
    ) -> None:
        self._llm = llm_adapter
        self._resolution = resolution_adapter
        self._reviewer = reviewer_adapter
        self._db = db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        customer_name: str,
        email: str,
        message: str,
    ) -> WorkflowResponse:
        """
        Full lifecycle:
            1. generate workflow_id
            2. persist running record
            3. invoke graph (triage → routing → [resolution | human_review_gate])
            4. interpret result → persist final state
            5. return WorkflowResponse

        Raises:
            WorkflowPersistenceError  — on any DB error
            Any exception from the graph is caught and stored as failed status
        """
        workflow_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # --- Step 1: persist running record ---
        run_record = self._create_running_record(
            workflow_id=workflow_id,
            started_at=now,
        )

        # --- Step 2: invoke graph ---
        graph_result: dict | None = None
        graph_error: Exception | None = None

        try:
            # resolution_adapter is required; use a safe fallback only if None
            # (e.g. GET /workflows/{id} path which passes llm_adapter=None)
            resolution_adapter = self._resolution
            if resolution_adapter is None:
                from app.workflows.adapters.openrouter_resolution import OpenRouterResolutionAdapter
                resolution_adapter = OpenRouterResolutionAdapter()

            reviewer_adapter = self._reviewer
            if reviewer_adapter is None:
                from app.workflows.adapters.mock import MockReviewerAdapter
                reviewer_adapter = MockReviewerAdapter(decision="APPROVED")

            graph = build_full_graph(
                llm_adapter=self._llm,
                resolution_adapter=resolution_adapter,
                reviewer_adapter=reviewer_adapter,
            )
            initial_state = make_initial_state(
                workflow_id=workflow_id,
                customer_name=customer_name,
                email=email,
                message=message,
            )
            graph_result = await graph.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": workflow_id}},
            )
        except Exception as exc:
            graph_error = exc
            logger.error(f"[workflow_run_service] graph error wf={workflow_id}: {exc}")

        # --- Step 3: interpret result and persist ---
        updated_at = datetime.now(timezone.utc).isoformat()

        if graph_error is not None:
            return self._persist_failed(
                run_record=run_record,
                error=graph_error,
                updated_at=updated_at,
            )

        assert graph_result is not None

        interrupted = "__interrupt__" in graph_result

        # Triage/Resolution node signals controlled failure via
        # workflow_status="failed" in the returned state.
        # The node sets explicit fields in execution_trace[-1]:
        # error_code, error_message, failed_node.
        graph_status = graph_result.get("workflow_status")
        node_failed = graph_status == "failed"

        if node_failed:
            return self._persist_failed_from_state(
                run_record=run_record,
                state=graph_result,
                updated_at=updated_at,
            )
        elif interrupted:
            return self._persist_waiting_review(
                run_record=run_record,
                state=graph_result,
                updated_at=updated_at,
            )
        else:
            return self._persist_completed(
                run_record=run_record,
                state=graph_result,
                customer_name=customer_name,
                email=email,
                message=message,
                updated_at=updated_at,
            )

    async def get(self, workflow_id: str) -> WorkflowResponse:
        """
        Retrieve a workflow by ID from the business read model.

        Does NOT consult LangGraph checkpoints — reads only WorkflowRunRecord.

        Raises:
            WorkflowNotFoundError  — if no record exists
        """
        record = (
            self._db.query(WorkflowRunRecord)
            .filter(WorkflowRunRecord.workflow_id == workflow_id)
            .first()
        )
        if record is None:
            raise WorkflowNotFoundError(f"Workflow '{workflow_id}' not found.")
        return _record_to_response(record)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_running_record(
        self,
        workflow_id: str,
        started_at: str,
    ) -> WorkflowRunRecord:
        """
        Create and persist a WorkflowRunRecord with status=running.

        Raises WorkflowPersistenceError on DB failure.
        """
        record = WorkflowRunRecord(
            workflow_id=workflow_id,
            ticket_id=None,
            status="running",
            current_node=None,
            state_json=None,
            retry_count=0,
            requires_human_review=False,
            human_review_status=None,
            started_at=started_at,
            updated_at=started_at,
            completed_at=None,
            error_code=None,
            error_message=None,
        )
        try:
            self._db.add(record)
            self._db.commit()
            self._db.refresh(record)
        except Exception as exc:
            self._db.rollback()
            logger.error(f"[workflow_run_service] failed to create running record: {exc}")
            raise WorkflowPersistenceError(f"Failed to create workflow record: {exc}") from exc
        return record

    def _persist_failed(
        self,
        run_record: WorkflowRunRecord,
        error: Exception,
        updated_at: str,
    ) -> WorkflowResponse:
        """
        Update record to status=failed for an unexpected infrastructure exception
        (the graph itself raised an exception — not a controlled node failure).
        No TicketRecord created.
        """
        run_record.status = "failed"
        run_record.updated_at = updated_at
        run_record.error_code = type(error).__name__
        run_record.error_message = str(error)[:2000]

        try:
            self._db.commit()
            self._db.refresh(run_record)
        except Exception as exc:
            self._db.rollback()
            logger.error(f"[workflow_run_service] failed to persist failed record: {exc}")
            raise WorkflowPersistenceError(f"Failed to update workflow record: {exc}") from exc

        return _record_to_response(run_record)

    def _persist_failed_from_state(
        self,
        run_record: WorkflowRunRecord,
        state: dict,
        updated_at: str,
    ) -> WorkflowResponse:
        """
        Update record to status=failed using the explicit failure contract fields
        set by a node (error_code, error_message, failed_node in execution_trace[-1]).

        Used when a node catches a domain error (TriageModelError, ResolutionModelError)
        and signals failure via workflow_status="failed" in its returned state dict.

        No TicketRecord created.
        """
        run_record.status = "failed"
        run_record.updated_at = updated_at
        # error_code, error_message and failed_node are stored in the last
        # execution_trace entry because LangGraph's state merging only updates
        # keys declared in WorkflowState TypedDict.
        trace = state.get("execution_trace") or []
        last_trace = trace[-1] if trace else {}
        run_record.error_code = last_trace.get("error_code") or "NodeFailure"
        run_record.error_message = (last_trace.get("error_message") or "")[:2000]
        # failed_node stored in current_node for observability
        run_record.current_node = last_trace.get("failed_node") or state.get("current_node")

        logger.error(
            f"[workflow_run_service] wf={run_record.workflow_id} → failed "
            f"node={run_record.current_node} "
            f"code={run_record.error_code}: {run_record.error_message}"
        )

        try:
            self._db.commit()
            self._db.refresh(run_record)
        except Exception as exc:
            self._db.rollback()
            logger.error(f"[workflow_run_service] failed to persist node-failed record: {exc}")
            raise WorkflowPersistenceError(f"Failed to update workflow record: {exc}") from exc

        return _record_to_response(run_record)

    def _persist_waiting_review(
        self,
        run_record: WorkflowRunRecord,
        state: dict,
        updated_at: str,
    ) -> WorkflowResponse:
        """
        Update record to status=waiting_review.
        ticket_id remains None. No TicketRecord created.
        Resolution Agent was NOT called (human_review_gate interrupted before it).
        """
        run_record.status = "waiting_review"
        run_record.current_node = state.get("current_node")
        run_record.state_json = serialize_state(state)
        run_record.retry_count = state.get("retry_count", 0)
        run_record.requires_human_review = True
        run_record.human_review_status = "pending"
        run_record.updated_at = updated_at
        run_record.ticket_id = None

        try:
            self._db.commit()
            self._db.refresh(run_record)
        except Exception as exc:
            self._db.rollback()
            logger.error(f"[workflow_run_service] failed to persist waiting_review record: {exc}")
            raise WorkflowPersistenceError(f"Failed to update workflow record: {exc}") from exc

        logger.info(
            f"[workflow_run_service] wf={run_record.workflow_id} → waiting_review"
        )
        return _record_to_response(run_record)

    def _persist_completed(
        self,
        run_record: WorkflowRunRecord,
        state: dict,
        customer_name: str,
        email: str,
        message: str,
        updated_at: str,
    ) -> WorkflowResponse:
        """
        Create TicketRecord + update WorkflowRunRecord in a SINGLE transaction.

        Uses REAL suggested_response and recommended_action from ResolutionNode
        (stored in state). "pending_resolution" is a defensive fallback only —
        a completed workflow that went through ResolutionNode will always have
        non-empty values.

        Idempotence: if ticket_id is already set on run_record, skip TicketRecord
        creation (second call is a no-op for the ticket).
        """
        completed_at = updated_at
        rd = (state.get("routing_decision") or {})

        # Pull resolution output from state (set by ResolutionNode)
        suggested_response = state.get("suggested_response") or "pending_resolution"
        recommended_action = state.get("recommended_action") or "pending_resolution"

        ticket_id: str
        if run_record.ticket_id:
            # Already completed (idempotent re-call)
            ticket_id = run_record.ticket_id
        else:
            ticket_id = str(uuid.uuid4())

            ticket_record = TicketRecord(
                id=ticket_id,
                customer_name=customer_name,
                email=email,
                original_message=message,
                classification=state.get("classification", "general"),
                priority=state.get("priority", "low"),
                summary=state.get("summary") or "",
                sentiment=state.get("sentiment", "neutral"),
                suggested_response=suggested_response,
                recommended_action=recommended_action,
                assigned_team=rd.get("assigned_team"),
                sla_hours=rd.get("sla_hours"),
                requires_human_review=rd.get("requires_human_review", False),
                escalation_level=rd.get("escalation_level", 0),
                internal_notes=rd.get("internal_notes"),
                processed_at=datetime.now(timezone.utc),
                ai_model_used=settings.openrouter_model,
            )
            self._db.add(ticket_record)

        # Update run record fields
        run_record.status = "completed"
        run_record.ticket_id = ticket_id
        run_record.current_node = state.get("current_node")
        run_record.state_json = serialize_state(state)
        run_record.retry_count = state.get("retry_count", 0)
        run_record.requires_human_review = rd.get("requires_human_review", False)
        run_record.updated_at = updated_at
        run_record.completed_at = completed_at

        # Single commit — both objects in same transaction
        try:
            self._db.commit()
            self._db.refresh(run_record)
        except Exception as exc:
            self._db.rollback()
            logger.error(f"[workflow_run_service] failed to persist completed record: {exc}")
            raise WorkflowPersistenceError(f"Failed to update workflow record: {exc}") from exc

        logger.info(
            f"[workflow_run_service] wf={run_record.workflow_id} → completed "
            f"ticket={ticket_id} "
            f"suggested_len={len(suggested_response)} "
            f"action_len={len(recommended_action)}"
        )
        return _record_to_response(run_record)

"""
routers/workflows.py — FastAPI router for the /workflows API.

Endpoints:
    POST /workflows           — create and run a new workflow (201)
    GET  /workflows/{id}      — retrieve a persisted workflow (200 | 404)

HTTP status mapping:
    201  — workflow created (completed OR waiting_review)
    404  — workflow_id not found
    422  — validation error (Pydantic)
    503  — LLM adapter / graph runtime error
    500  — unexpected persistence error
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.workflow import WorkflowCreateRequest, WorkflowResponse
from app.services.workflow_run_service import (
    WorkflowNotFoundError,
    WorkflowPersistenceError,
    WorkflowRunService,
)
from app.utils.logger import logger
from app.workflows.adapters.openrouter import OpenRouterTriageAdapter
from app.workflows.adapters.openrouter_resolution import OpenRouterResolutionAdapter
from app.workflows.adapters.openrouter_reviewer import OpenRouterReviewerAdapter

router = APIRouter(prefix="/workflows", tags=["Workflows"])


# ---------------------------------------------------------------------------
# Dependencies: LLM adapters
# ---------------------------------------------------------------------------

def get_llm_adapter():
    """
    Provides the production Triage LLM adapter.
    Override in tests via app.dependency_overrides.
    """
    return OpenRouterTriageAdapter()


def get_resolution_adapter():
    """
    Provides the production Resolution LLM adapter.
    Override in tests via app.dependency_overrides.
    """
    return OpenRouterResolutionAdapter()


def get_reviewer_adapter():
    """
    Provides the production Quality Reviewer adapter.
    Override in tests via app.dependency_overrides.
    """
    return OpenRouterReviewerAdapter()


# ---------------------------------------------------------------------------
# POST /workflows
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=WorkflowResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create and run a new AI workflow",
    description=(
        "Submits a support message through the full LangGraph pipeline: "
        "triage → routing → (conditional) resolution OR human review gate. "
        "Returns 201 for both 'completed' and 'waiting_review' outcomes."
    ),
)
async def create_workflow(
    request: WorkflowCreateRequest,
    db: Session = Depends(get_db),
    llm_adapter=Depends(get_llm_adapter),
    resolution_adapter=Depends(get_resolution_adapter),
    reviewer_adapter=Depends(get_reviewer_adapter),
) -> WorkflowResponse:
    service = WorkflowRunService(
        llm_adapter=llm_adapter,
        resolution_adapter=resolution_adapter,
        reviewer_adapter=reviewer_adapter,
        db=db,
    )
    try:
        return await service.run(
            customer_name=request.customer_name,
            email=str(request.email),
            message=request.message,
        )
    except WorkflowPersistenceError as exc:
        logger.error(f"[workflows router] persistence error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow persistence failed: {exc}",
        )
    except Exception as exc:
        logger.error(f"[workflows router] unexpected error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Workflow service unavailable: {exc}",
        )


# ---------------------------------------------------------------------------
# GET /workflows/{workflow_id}
# ---------------------------------------------------------------------------

@router.get(
    "/{workflow_id}",
    response_model=WorkflowResponse,
    status_code=status.HTTP_200_OK,
    summary="Get workflow run by ID",
    description=(
        "Returns the business read model for the given workflow_id. "
        "Does NOT consult LangGraph checkpoints — only the DB record."
    ),
)
async def get_workflow(
    workflow_id: str,
    db: Session = Depends(get_db),
) -> WorkflowResponse:
    service = WorkflowRunService(llm_adapter=None, db=db)  # type: ignore[arg-type]
    try:
        return await service.get(workflow_id)
    except WorkflowNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow '{workflow_id}' not found.",
        )

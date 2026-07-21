"""
workflows/errors.py — Domain exceptions for the LangGraph workflow layer.

These are raised by nodes, adapters, and persistence.
The FastAPI router maps them to HTTP status codes.
No HTTPException here — this layer is HTTP-unaware.
"""


class WorkflowError(Exception):
    """Base class for all workflow domain errors."""

    def __init__(self, message: str, workflow_id: str | None = None) -> None:
        super().__init__(message)
        self.workflow_id = workflow_id


class TriageModelError(WorkflowError):
    """
    Raised when the LLM adapter fails to produce a valid TriageOutput.
    Covers: network errors, malformed JSON, failed Pydantic validation.
    """


class ResolutionModelError(WorkflowError):
    """
    Raised when the Resolution adapter fails to produce a valid ResolutionOutput.
    Covers: network errors, malformed JSON, failed Pydantic validation,
    empty suggested_response or recommended_action.
    """


class ReviewerModelError(WorkflowError):
    """
    Raised when the Reviewer adapter fails to produce a valid ReviewOutput.
    Covers: network errors, malformed JSON, failed Pydantic validation.
    """


class WorkflowPersistenceError(WorkflowError):
    """Raised when saving/reading WorkflowRunRecord fails."""


class WorkflowNotFoundError(WorkflowError):
    """Raised when GET /workflows/{id} finds no matching record."""

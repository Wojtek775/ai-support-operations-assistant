"""
routers/tickets.py — REST endpoints for ticket management.

Endpoints:
    POST /tickets       — Submit a new ticket for AI processing
    GET  /tickets       — List all processed tickets (condensed)
    GET  /tickets/{id}  — Get a single ticket by UUID
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.models.ticket import TicketRequest, ProcessedTicket, TicketListItem
from app.services import ticket_service
from app.database import get_db

router = APIRouter(prefix="/tickets", tags=["Tickets"])


@router.post(
    "",
    response_model=ProcessedTicket,
    status_code=201,
    summary="Submit a new support ticket",
    description=(
        "Accepts a customer support message, runs it through the full AI pipeline "
        "(classify → prioritize → summarize → sentiment → suggest response → recommend action), "
        "saves the result, and returns the complete processed ticket."
    ),
)
async def submit_ticket(
    request: TicketRequest,
    db: Session = Depends(get_db),
) -> ProcessedTicket:
    """
    Full AI pipeline for a single ticket.

    - Validates input (Pydantic)
    - Calls OpenRouter AI API
    - Persists result to database
    - Returns full ProcessedTicket
    """
    return await ticket_service.create_ticket(request, db)


@router.get(
    "",
    response_model=list[TicketListItem],
    summary="List all processed tickets",
    description="Returns all tickets ordered by newest first. Each item is a condensed view (no long text fields).",
)
def list_tickets(
    db: Session = Depends(get_db),
) -> list[TicketListItem]:
    """
    Returns a list of all tickets with key metadata fields.
    Long text fields (summary, suggested_response, etc.) are excluded for brevity.
    Use GET /tickets/{ticket_id} to get the full record.
    """
    return ticket_service.get_all_tickets(db)


@router.get(
    "/{ticket_id}",
    response_model=ProcessedTicket,
    summary="Get a single ticket by ID",
    description="Fetches the full processed ticket record including all AI-generated fields.",
)
def get_ticket(
    ticket_id: str,
    db: Session = Depends(get_db),
) -> ProcessedTicket:
    """
    Returns the full ticket record for the given UUID.
    Returns 404 if the ticket does not exist.
    """
    return ticket_service.get_ticket_by_id(ticket_id, db)

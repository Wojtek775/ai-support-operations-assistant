"""
services/ticket_service.py — Business logic for ticket processing.

This is the orchestration layer:
- Receives a TicketRequest from the router
- Calls the AI service to process the message
- Builds a TicketRecord and persists it to the database
- Returns a ProcessedTicket to the router

This module knows about the DB and AI service, but not about HTTP.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from fastapi import HTTPException

from app.models.ticket import TicketRequest, ProcessedTicket, TicketListItem
from app.models.db_models import TicketRecord
from app.services.ai_service import process_ticket_with_ai
from app.config import settings
from app.utils.logger import logger


async def create_ticket(request: TicketRequest, db: Session) -> ProcessedTicket:
    """
    Full pipeline: receive request → AI processing → persist → return result.

    Args:
        request: Validated TicketRequest from the router.
        db: SQLAlchemy database session (injected by FastAPI dependency).

    Returns:
        ProcessedTicket with all fields populated.

    Raises:
        HTTPException 503: If the AI API call fails.
        HTTPException 500: For unexpected errors.
    """
    ticket_id = str(uuid.uuid4())
    logger.info(f"Processing new ticket | id={ticket_id} | customer={request.customer_name}")

    # Step 1: Call AI service for analysis
    try:
        ai_result = await process_ticket_with_ai(request.message)
    except Exception as e:
        logger.error(f"AI service failed for ticket {ticket_id}: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"AI processing service is unavailable. Please try again. Details: {str(e)}",
        )

    # Step 2: Build timestamp
    processed_at = datetime.now(timezone.utc)

    # Step 3: Persist to database
    try:
        db_record = TicketRecord(
            id=ticket_id,
            customer_name=request.customer_name,
            email=str(request.email),
            original_message=request.message,
            classification=ai_result["classification"],
            priority=ai_result["priority"],
            summary=ai_result["summary"],
            sentiment=ai_result["sentiment"],
            suggested_response=ai_result["suggested_response"],
            recommended_action=ai_result["recommended_action"],
            processed_at=processed_at,
            ai_model_used=settings.openrouter_model,
        )
        db.add(db_record)
        db.commit()
        db.refresh(db_record)
        logger.info(f"Ticket saved to database | id={ticket_id}")
    except Exception as e:
        db.rollback()
        logger.error(f"Database error for ticket {ticket_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save ticket to database.")

    # Step 4: Return the full processed ticket
    return ProcessedTicket(
        ticket_id=ticket_id,
        customer_name=request.customer_name,
        email=str(request.email),
        original_message=request.message,
        classification=ai_result["classification"],
        priority=ai_result["priority"],
        summary=ai_result["summary"],
        sentiment=ai_result["sentiment"],
        suggested_response=ai_result["suggested_response"],
        recommended_action=ai_result["recommended_action"],
        processed_at=processed_at,
        ai_model_used=settings.openrouter_model,
    )


def get_all_tickets(db: Session) -> list[TicketListItem]:
    """
    Fetch all tickets from the database, ordered by newest first.

    Args:
        db: SQLAlchemy database session.

    Returns:
        List of TicketListItem (condensed, no long text fields).
    """
    records = (
        db.query(TicketRecord)
        .order_by(TicketRecord.processed_at.desc())
        .all()
    )

    return [
        TicketListItem(
            ticket_id=record.id,
            customer_name=record.customer_name,
            email=record.email,
            classification=record.classification,
            priority=record.priority,
            sentiment=record.sentiment,
            processed_at=record.processed_at,
        )
        for record in records
    ]


def get_ticket_by_id(ticket_id: str, db: Session) -> ProcessedTicket:
    """
    Fetch a single ticket by its UUID.

    Args:
        ticket_id: The UUID string of the ticket.
        db: SQLAlchemy database session.

    Returns:
        ProcessedTicket with all fields.

    Raises:
        HTTPException 404: If no ticket with the given ID exists.
    """
    record = db.query(TicketRecord).filter(TicketRecord.id == ticket_id).first()

    if not record:
        logger.warning(f"Ticket not found | id={ticket_id}")
        raise HTTPException(status_code=404, detail=f"Ticket with id '{ticket_id}' not found.")

    return ProcessedTicket(
        ticket_id=record.id,
        customer_name=record.customer_name,
        email=record.email,
        original_message=record.original_message,
        classification=record.classification,
        priority=record.priority,
        summary=record.summary,
        sentiment=record.sentiment,
        suggested_response=record.suggested_response,
        recommended_action=record.recommended_action,
        processed_at=record.processed_at,
        ai_model_used=record.ai_model_used,
    )

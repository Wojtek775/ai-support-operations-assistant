"""
models/ticket.py — Pydantic schemas for ticket request and response.

These are the data shapes used at the API boundary:
- TicketRequest: what the client sends
- ProcessedTicket: what the API returns after AI processing
- TicketListItem: lightweight version used in list responses
"""

from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from enum import Enum


# ---------------------------------------------------------------------------
# Enums — constrain AI output values for type safety
# ---------------------------------------------------------------------------

class TicketClassification(str, Enum):
    """Categories a support ticket can be assigned to."""
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    SHIPPING = "shipping"
    GENERAL = "general"


class TicketPriority(str, Enum):
    """Urgency levels for a support ticket."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketSentiment(str, Enum):
    """Customer emotional tone detected in the message."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    ANGRY = "angry"


# ---------------------------------------------------------------------------
# Request schema — incoming POST /tickets body
# ---------------------------------------------------------------------------

class TicketRequest(BaseModel):
    """
    Data the client submits when creating a new support ticket.
    All fields are required to ensure meaningful AI processing.
    """
    customer_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Full name of the customer submitting the ticket.",
        examples=["Jan Kowalski"],
    )
    email: EmailStr = Field(
        ...,
        description="Customer email address.",
        examples=["jan@example.com"],
    )
    message: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="The support message or issue description.",
        examples=["My invoice is wrong, I was charged twice this month."],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_name": "Jan Kowalski",
                "email": "jan@example.com",
                "message": "My invoice is wrong, I was charged twice this month. Please fix this urgently.",
            }
        }
    }


# ---------------------------------------------------------------------------
# Response schema — full processed ticket returned by POST /tickets
# ---------------------------------------------------------------------------

class ProcessedTicket(BaseModel):
    """
    Full ticket record after AI pipeline processing.
    Contains original request data plus all AI-generated fields.
    """
    ticket_id: str = Field(..., description="Unique ticket identifier (UUID).")
    customer_name: str
    email: str
    original_message: str = Field(..., description="The raw message submitted by the customer.")

    # AI-generated fields
    classification: TicketClassification = Field(..., description="Ticket category assigned by AI.")
    priority: TicketPriority = Field(..., description="Urgency level assigned by AI.")
    summary: str = Field(..., description="1-2 sentence summary of the issue.")
    sentiment: TicketSentiment = Field(..., description="Customer sentiment detected by AI.")
    suggested_response: str = Field(..., description="Draft reply for the customer support agent.")
    recommended_action: str = Field(..., description="Internal operational step recommended by AI.")

    # Metadata
    processed_at: datetime = Field(..., description="Server-side timestamp when processing completed.")
    ai_model_used: str = Field(..., description="The AI model identifier used for processing.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "ticket_id": "550e8400-e29b-41d4-a716-446655440000",
                "customer_name": "Jan Kowalski",
                "email": "jan@example.com",
                "original_message": "My invoice is wrong, I was charged twice this month.",
                "classification": "billing",
                "priority": "high",
                "summary": "Customer reports a duplicate billing charge for the current month.",
                "sentiment": "negative",
                "suggested_response": "Dear Jan, we sincerely apologize for the inconvenience. Our billing team will investigate the duplicate charge and issue a refund within 2 business days.",
                "recommended_action": "ESCALATE_TO_BILLING — Verify payment records for duplicate charge, initiate refund if confirmed.",
                "processed_at": "2024-01-15T10:30:00Z",
                "ai_model_used": "openai/gpt-4o-mini",
            }
        }
    }


# ---------------------------------------------------------------------------
# List item schema — lightweight version for GET /tickets list
# ---------------------------------------------------------------------------

class TicketListItem(BaseModel):
    """
    Condensed ticket representation for list views.
    Omits long text fields (summary, suggested_response, etc.) for brevity.
    """
    ticket_id: str
    customer_name: str
    email: str
    classification: TicketClassification
    priority: TicketPriority
    sentiment: TicketSentiment
    processed_at: datetime

# Technical Architecture — AI Support & Operations Assistant

## Overview

This system is an **AI-powered operational workflow engine** built on FastAPI and OpenRouter. It processes incoming support messages through a structured multi-step AI pipeline and stores results in a database for downstream consumption (dashboards, reporting, automation).

---

## System Design Principles

- **Single Responsibility** — each module does one thing well
- **Separation of Concerns** — routing, business logic, and AI calls are isolated
- **Configuration-First** — all secrets and settings in `.env`, never hardcoded
- **DB-Agnostic** — SQLAlchemy ORM abstracts the database; swap SQLite → PostgreSQL with one config change
- **Fail-Safe AI** — AI errors are caught, logged, and returned as structured error responses
- **Extensible Pipeline** — add/remove pipeline steps without touching the router or models

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT / FRONTEND                        │
│              (curl / Postman / Next.js dashboard)               │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP REST
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FASTAPI APPLICATION                        │
│                                                                 │
│  ┌─────────────┐   ┌──────────────────────────────────────────┐ │
│  │  /health    │   │             /tickets                     │ │
│  │  router     │   │  POST → submit ticket                    │ │
│  └─────────────┘   │  GET  → list all tickets                 │ │
│                    │  GET  → get single ticket by ID           │ │
│                    └────────────────┬─────────────────────────┘ │
└────────────────────────────────────┼────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                     TICKET SERVICE (Pipeline)                   │
│                                                                 │
│  Input: raw customer message                                    │
│                                                                 │
│  Step 1: Build structured AI prompt                             │
│  Step 2: Call AI Service → OpenRouter API                       │
│  Step 3: Parse JSON response from LLM                           │
│  Step 4: Validate & enrich with metadata                        │
│  Step 5: Persist to database                                    │
│  Step 6: Return ProcessedTicket to caller                       │
└──────────────┬──────────────────────────────────────────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────────┐   ┌─────────────────────────────────┐
│      AI SERVICE          │   │         DATABASE LAYER          │
│                          │   │                                 │
│  - Builds system prompt  │   │  SQLAlchemy ORM                 │
│  - Calls OpenRouter API  │   │  SQLite (dev)                   │
│  - Returns structured    │   │  PostgreSQL (prod-ready)        │
│    JSON with:            │   │                                 │
│    • classification      │   │  Table: tickets                 │
│    • priority            │   │  - id (UUID)                    │
│    • summary             │   │  - customer_name                │
│    • sentiment           │   │  - email                        │
│    • suggested_response  │   │  - original_message             │
│    • recommended_action  │   │  - classification               │
└──────────────────────────┘   │  - priority                     │
                               │  - summary                      │
                               │  - sentiment                    │
                               │  - suggested_response           │
                               │  - recommended_action           │
                               │  - processed_at                 │
                               │  - ai_model_used                │
                               └─────────────────────────────────┘
```

---

## AI Pipeline — Prompt Design

The system uses a **single-call structured output** approach for efficiency:

```
SYSTEM PROMPT:
  You are an expert customer support analyst.
  Analyze the message and return ONLY valid JSON with these fields:
    - classification: one of [billing, technical, account, shipping, general]
    - priority: one of [low, medium, high, critical]
    - summary: 1-2 sentence summary
    - sentiment: one of [positive, neutral, negative, angry]
    - suggested_response: professional reply to the customer
    - recommended_action: internal operational next step

USER MESSAGE:
  {customer_message}
```

**Why single-call?**
- Reduces API latency and cost
- Keeps the pipeline atomic
- Easier to debug (one request, one response)
- Can be split into multi-step chain in future iterations

---

## Data Flow

```
POST /tickets
     │
     ├── Validate input (Pydantic)
     │
     ├── Generate ticket_id (UUID)
     │
     ├── Send to AI Service
     │      └── OpenRouter API call
     │             └── Returns JSON with 6 fields
     │
     ├── Parse & validate AI response
     │
     ├── Persist to SQLite/PostgreSQL
     │
     └── Return ProcessedTicket (200 OK)
```

---

## Module Responsibilities

### `app/main.py`
- FastAPI app initialization
- Router registration
- Database table creation on startup
- CORS middleware configuration

### `app/config.py`
- Pydantic `BaseSettings` for environment variables
- Singleton config object used across the app
- Database URL construction

### `app/models/ticket.py`
- `TicketRequest` — incoming POST body schema
- `ProcessedTicket` — full pipeline output schema
- `TicketListItem` — lightweight schema for listing

### `app/models/db_models.py`
- SQLAlchemy `TicketRecord` ORM model
- Maps directly to `tickets` table
- UUID primary key (string-stored for SQLite compatibility)

### `app/services/ai_service.py`
- `process_ticket_with_ai(message: str) -> dict`
- Builds prompt, calls OpenRouter, parses JSON response
- Handles API errors, malformed JSON, timeouts

### `app/services/ticket_service.py`
- `create_ticket(request, db)` — orchestrates the full pipeline
- `get_all_tickets(db)` — fetches all records
- `get_ticket_by_id(ticket_id, db)` — fetches single record

### `app/routers/tickets.py`
- POST `/tickets` → calls `ticket_service.create_ticket`
- GET `/tickets` → calls `ticket_service.get_all_tickets`
- GET `/tickets/{ticket_id}` → calls `ticket_service.get_ticket_by_id`

### `app/routers/health.py`
- GET `/health` → returns system status, version, timestamp

### `app/utils/logger.py`
- Configured Python logger with structured format
- Used across services for consistent log output

---

## Database Design

### Table: `tickets`

| Column              | Type         | Notes                        |
|---------------------|--------------|------------------------------|
| id                  | VARCHAR(36)  | UUID primary key             |
| customer_name       | VARCHAR(255) | From request                 |
| email               | VARCHAR(255) | From request                 |
| original_message    | TEXT         | Raw input                    |
| classification      | VARCHAR(50)  | AI output                    |
| priority            | VARCHAR(20)  | AI output                    |
| summary             | TEXT         | AI output                    |
| sentiment           | VARCHAR(20)  | AI output                    |
| suggested_response  | TEXT         | AI output                    |
| recommended_action  | TEXT         | AI output                    |
| processed_at        | DATETIME     | Server timestamp             |
| ai_model_used       | VARCHAR(100) | Model identifier             |

---

## Migration Path: SQLite → PostgreSQL

Only one change required in `.env`:

```env
# SQLite (default)
DATABASE_URL=sqlite:///./app/data/tickets.db

# PostgreSQL (production)
DATABASE_URL=postgresql://user:password@localhost:5432/support_db
```

SQLAlchemy handles the rest. No code changes needed.

---

## Error Handling Strategy

| Scenario                        | Response                              |
|---------------------------------|---------------------------------------|
| Invalid request body            | 422 Unprocessable Entity (Pydantic)   |
| OpenRouter API error            | 503 Service Unavailable + message     |
| AI returns malformed JSON       | 500 Internal Error + raw AI output    |
| Ticket not found                | 404 Not Found                         |
| Database error                  | 500 Internal Error + logged details   |

---

## Security Considerations (MVP level)

- API key stored in `.env`, never committed to git
- `.gitignore` excludes `.env` and database files
- Input validation via Pydantic on all endpoints
- No authentication on MVP (add API key auth in Phase 2)

---

## Future Extensions

- [ ] Add authentication (API key or JWT)
- [ ] Webhook support for incoming tickets
- [ ] Multi-step LLM chain (classify → then respond)
- [ ] Email notification on critical priority tickets
- [ ] PostgreSQL + Alembic migrations
- [ ] Next.js dashboard frontend
- [ ] Ticket assignment and status tracking
- [ ] Analytics endpoints (classification breakdown, avg priority)

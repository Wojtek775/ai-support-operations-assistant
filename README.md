# AI Support & Operations Assistant

> **Portfolio MVP** — AI-powered operational workflow system demonstrating intelligent ticket processing, classification, and response generation.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![OpenRouter](https://img.shields.io/badge/AI-OpenRouter-orange.svg)](https://openrouter.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 What This System Does

This is **not just a chatbot**. It is an AI-powered operational workflow engine that processes customer support messages through a structured pipeline:

1. **Receive** — Accept incoming support ticket via REST API
2. **Classify** — Categorize the issue (billing, technical, account, etc.)
3. **Detect Priority** — Assign urgency level (low / medium / high / critical)
4. **Summarize** — Generate a concise 1-2 sentence summary
5. **Detect Sentiment** — Identify emotional tone (positive / neutral / negative / angry)
6. **Suggest Response** — Draft a professional customer reply
7. **Recommend Action** — Propose an internal operational step
8. **Persist** — Store structured results in SQLite (PostgreSQL-ready)
9. **Expose** — Serve data via REST API for dashboard consumption

---

## 🏗️ Project Structure

```
ai-support-operations-assistant/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app entry point
│   │   ├── config.py            # Environment config (Pydantic Settings)
│   │   ├── models/
│   │   │   ├── ticket.py        # Pydantic request/response models
│   │   │   └── db_models.py     # SQLAlchemy ORM models
│   │   ├── services/
│   │   │   ├── ai_service.py    # OpenRouter API integration
│   │   │   └── ticket_service.py# Business logic: full pipeline
│   │   ├── routers/
│   │   │   ├── tickets.py       # POST /tickets, GET /tickets
│   │   │   └── health.py        # GET /health
│   │   ├── data/                # SQLite DB file lives here
│   │   └── utils/
│   │       └── logger.py        # Structured logging setup
│   ├── tests/
│   │   └── test_tickets.py      # Basic endpoint tests
│   ├── requirements.txt
│   └── .env.example
├── docs/
│   ├── TECHNICAL_ARCHITECTURE.md
│   └── ROADMAP.md
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/your-username/ai-support-operations-assistant.git
cd ai-support-operations-assistant/backend

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env — add your OpenRouter API key
```

### 3. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Submit a ticket
curl -X POST http://localhost:8000/tickets \
  -H "Content-Type: application/json" \
  -d '{"customer_name": "Jan Kowalski", "email": "jan@example.com", "message": "My invoice is wrong, I was charged twice this month."}'
```

Interactive docs: **http://localhost:8000/docs**

---

## 📡 API Endpoints

| Method | Endpoint         | Description                          |
|--------|-----------------|--------------------------------------|
| GET    | `/health`        | System health check                  |
| POST   | `/tickets`       | Submit a new support ticket          |
| GET    | `/tickets`       | List all processed tickets           |
| GET    | `/tickets/{id}`  | Get a single ticket by ID            |

---

## 🧠 AI Pipeline Output Example

```json
{
  "ticket_id": "tkt_20240115_001",
  "customer_name": "Jan Kowalski",
  "email": "jan@example.com",
  "original_message": "My invoice is wrong, I was charged twice this month.",
  "classification": "billing",
  "priority": "high",
  "summary": "Customer reports duplicate billing charge for the current month.",
  "sentiment": "negative",
  "suggested_response": "Dear Jan, we sincerely apologize for the billing error. Our finance team will review your account within 24 hours and issue a refund if a duplicate charge is confirmed.",
  "recommended_action": "ESCALATE_TO_BILLING — Investigate duplicate charge, verify payment records, initiate refund if confirmed.",
  "processed_at": "2024-01-15T10:30:00Z",
  "ai_model_used": "openai/gpt-4o-mini"
}
```

---

## 🧪 Running Tests

```bash
cd backend
pytest tests/ -v
```

---

## 🛠️ Tech Stack

| Layer       | Technology              |
|-------------|------------------------|
| Backend     | Python 3.11 + FastAPI  |
| AI Provider | OpenRouter API         |
| Database    | SQLite (→ PostgreSQL)  |
| ORM         | SQLAlchemy 2.x         |
| Validation  | Pydantic v2            |
| Testing     | pytest + httpx         |

---

## 📌 Use Cases

- **LinkedIn content** — Demonstrating real AI-powered system design
- **GitHub portfolio** — Production-ready code structure
- **Job applications** — AI Integration Developer / AI Automation Engineer
- **Consulting** — Small business AI workflow automation starter

---

## 📄 Docs

- [Technical Architecture](docs/TECHNICAL_ARCHITECTURE.md)
- [Roadmap](docs/ROADMAP.md)

---

## 📝 License

MIT — free to use, modify, and distribute.

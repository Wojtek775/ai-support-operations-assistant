# Project Roadmap — AI Support & Operations Assistant

## Vision

Build a production-quality AI-powered support operations system that can be demonstrated to potential employers and clients, progressively adding real-world features phase by phase.

---

## Phase 1 — Backend MVP ✅ (Current)

**Goal:** Working API that processes a support ticket end-to-end through an AI pipeline.

### Deliverables
- [x] Project structure and documentation
- [x] FastAPI application skeleton
- [x] `.env` configuration management
- [x] OpenRouter API integration
- [x] Full AI pipeline: classify → priority → summarize → sentiment → respond → action
- [x] SQLite database with SQLAlchemy ORM
- [x] REST endpoints: POST /tickets, GET /tickets, GET /tickets/{id}
- [x] Health check endpoint
- [x] Pydantic input validation
- [x] Structured error handling
- [x] Basic test suite
- [x] curl / Postman examples

### Success Criteria
- A `POST /tickets` call returns a fully processed ticket with all 6 AI fields populated
- Data persists between server restarts
- Docs available at `/docs`

---

## Phase 2 — Hardening & Auth

**Goal:** Make the API production-safe with authentication and better error handling.

### Deliverables
- [ ] API Key authentication middleware
- [ ] Rate limiting (slowapi)
- [ ] Request/response logging middleware
- [ ] Retry logic for OpenRouter API calls
- [ ] Improved AI prompt with few-shot examples
- [ ] Ticket status field (new / in_progress / resolved)
- [ ] PATCH `/tickets/{id}/status` endpoint
- [ ] Alembic database migrations
- [ ] Environment-specific configs (dev / prod)
- [ ] Docker + docker-compose setup

### Success Criteria
- Unauthenticated requests rejected with 401
- Server recovers gracefully from AI API downtime
- `docker-compose up` starts the full stack

---

## Phase 3 — Analytics & Reporting

**Goal:** Expose aggregated data for dashboard consumption.

### Deliverables
- [ ] GET `/analytics/summary` — ticket counts by classification, priority, sentiment
- [ ] GET `/analytics/trends` — tickets per day/week
- [ ] Filtering on GET `/tickets` (by classification, priority, date range)
- [ ] Pagination on GET `/tickets`
- [ ] CSV export endpoint
- [ ] PostgreSQL migration (Alembic)
- [ ] Background task for async ticket processing (FastAPI BackgroundTasks)

### Success Criteria
- Analytics endpoint returns meaningful aggregations
- GET /tickets supports `?classification=billing&priority=high` filters
- Runs on PostgreSQL without code changes

---

## Phase 4 — Frontend Dashboard

**Goal:** Visual interface for viewing and managing tickets.

### Deliverables
- [ ] Next.js 14 frontend (App Router)
- [ ] Ticket submission form
- [ ] Ticket list with filters and search
- [ ] Ticket detail view (full AI analysis displayed)
- [ ] Analytics charts (Recharts or Chart.js)
- [ ] Priority badge color-coding
- [ ] Sentiment icon display
- [ ] Dark mode support
- [ ] Responsive design (mobile-ready)

### Success Criteria
- Submit a ticket via form → see AI results within 5 seconds
- Dashboard shows classification and priority breakdown charts

---

## Phase 5 — Production Deployment

**Goal:** Deploy a live demo accessible via public URL.

### Deliverables
- [ ] Deploy backend to Railway / Render / Fly.io
- [ ] Deploy frontend to Vercel
- [ ] PostgreSQL hosted database (Railway / Supabase)
- [ ] Environment secrets management
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Custom domain (optional)
- [ ] Demo video for LinkedIn/portfolio

### Success Criteria
- Public URL works 24/7
- CI runs tests on every push to main
- README includes live demo link

---

## Phase 6 — Advanced AI Features

**Goal:** Demonstrate advanced AI engineering capabilities.

### Deliverables
- [ ] Multi-step LLM chain (separate classify → respond calls)
- [ ] Embeddings-based similar ticket detection
- [ ] Auto-escalation webhook (when priority = critical)
- [ ] Email notification integration (SendGrid/Resend)
- [ ] Slack notification for critical tickets
- [ ] LLM-powered ticket search (semantic search)
- [ ] A/B testing different AI models
- [ ] Cost tracking per ticket (token usage logging)

---

## Milestone Timeline (Suggested)

| Phase | Duration   | Focus                    |
|-------|------------|--------------------------|
| 1     | Week 1     | Backend MVP              |
| 2     | Week 2     | Auth + hardening         |
| 3     | Week 3     | Analytics + filtering    |
| 4     | Weeks 4-5  | Frontend dashboard       |
| 5     | Week 6     | Deployment               |
| 6     | Ongoing    | Advanced features        |

---

## Tech Debt Tracker

Things intentionally deferred for simplicity in MVP:

| Item                          | Deferred Until |
|-------------------------------|----------------|
| Authentication                | Phase 2        |
| Async AI processing           | Phase 3        |
| PostgreSQL                    | Phase 3        |
| Frontend                      | Phase 4        |
| CI/CD                         | Phase 5        |
| Semantic search               | Phase 6        |

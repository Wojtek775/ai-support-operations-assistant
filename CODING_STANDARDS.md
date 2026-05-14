# Coding Standards — AI Support & Operations Assistant

> These standards apply to all code in this repository.
> The goal is readable, maintainable, and demo-ready code.

---

## 1. Python Version

- **Target: Python 3.11+**
- Use modern Python features: `match/case`, `X | Y` union types, `list[str]` generics (no `List[str]` from `typing`)
- Do not use deprecated `typing` aliases — use built-in types directly

---

## 2. Code Formatting

- **Formatter: Black** (line length: 88)
- **Import sorter: isort** (compatible with Black)
- All files must pass `black --check .` with zero changes
- No trailing whitespace, no mixed tabs/spaces
- Two blank lines between top-level definitions, one blank line between class methods

```bash
# Format before committing:
black backend/
isort backend/
```

---

## 3. Naming Conventions

| Element            | Convention          | Example                      |
|--------------------|---------------------|------------------------------|
| Variables          | `snake_case`        | `ticket_id`, `customer_name` |
| Functions          | `snake_case`        | `create_ticket()`, `get_db()`|
| Classes            | `PascalCase`        | `TicketRequest`, `Settings`  |
| Constants          | `UPPER_SNAKE_CASE`  | `REQUIRED_AI_FIELDS`         |
| Files/modules      | `snake_case`        | `ai_service.py`, `main.py`   |
| Pydantic models    | `PascalCase`        | `ProcessedTicket`            |
| DB ORM models      | `PascalCase + Record` | `TicketRecord`             |
| Enums              | `PascalCase`        | `TicketPriority`             |
| Enum values        | `UPPER_SNAKE_CASE`  | `BILLING`, `HIGH`            |
| Router prefixes    | lowercase, no slash | `"/tickets"`, `"/health"`    |

---

## 4. Module Structure Rules

- Each module has a **single responsibility** — no mixing of routing, business logic, and DB access
- Import order: stdlib → third-party → local (enforced by isort)
- No circular imports — the dependency graph flows one way:
  ```
  routers → services → ai_service / db models → config / utils
  ```
- Every module starts with a docstring explaining its responsibility

---

## 5. API Design Conventions

- Use **Pydantic models** for all request/response schemas — never raw dicts at the boundary
- HTTP status codes must be accurate:
  - `201 Created` for successful POST
  - `200 OK` for GET
  - `404 Not Found` for missing resources
  - `422 Unprocessable Entity` for validation errors (automatic via Pydantic)
  - `503 Service Unavailable` for external API failures
  - `500 Internal Server Error` for unexpected server failures
- Route paths: lowercase, hyphenated, plural nouns (`/tickets`, `/analytics/summary`)
- Always include `summary` and `description` in route decorators for auto-docs
- Use `response_model=` on every endpoint to document the response schema

---

## 6. Error Handling Principles

- **Always catch specific exceptions** — never bare `except:` or `except Exception:` without logging
- Log errors with context (ticket ID, model name, etc.) before re-raising
- External API failures → `HTTPException(503)` with user-friendly message
- Database failures → rollback + `HTTPException(500)`
- AI response parse errors → `HTTPException(500)` with raw response in dev mode
- Validation errors → handled automatically by Pydantic / FastAPI (422)
- Never expose raw tracebacks or internal error details in production responses

---

## 7. Environment Variable Rules

- **All secrets and configurable values must be in `.env`** — never hardcoded
- `.env` is listed in `.gitignore` — never commit it
- `.env.example` is committed — it documents every required variable with safe placeholder values
- Access config only via the `settings` singleton from `app.config`
- No `os.getenv()` calls scattered in business logic — always go through `settings`
- Production secrets are injected via environment variables (never `.env` files in deployment)

---

## 8. Security Rules

- **Never hardcode API keys, passwords, tokens, or connection strings**
- `.env` must be in `.gitignore` — verified before every commit
- Never log raw API keys, passwords, or PII (emails should be masked in prod logs)
- No `DEBUG=true` in production deployments
- CORS: restrict `allow_origins` to known domains in production
- Input validation: rely on Pydantic for all API inputs — add field constraints (`min_length`, `max_length`)
- No SQL string interpolation — always use SQLAlchemy ORM or parameterized queries

---

## 9. Testing Expectations

- Tests live in `backend/tests/`
- Test file naming: `test_<module_name>.py`
- Use `pytest` as the test runner
- Use FastAPI's `TestClient` for endpoint tests
- **Mock external dependencies** (OpenRouter API, email services, etc.) — tests must never call real APIs
- Override FastAPI dependencies with `app.dependency_overrides` for DB isolation
- Use in-memory SQLite (`sqlite:///:memory:`) for test databases
- Each test class covers one endpoint or one unit of behavior
- Test naming: `test_<what_it_does>_<expected_result>` (e.g., `test_submit_ticket_returns_201`)
- Aim for: happy path + validation failure + not-found error for each endpoint

```bash
# Run tests:
cd backend
pytest tests/ -v
```

---

## 10. AI Integration Guidelines

- **Structured output**: always request JSON from the AI and validate the response schema
- **Temperature**: use low temperature (`0.1`–`0.3`) for deterministic structured output
- **Single-call preference**: prefer one AI call per pipeline run over multiple chained calls (for MVP)
- **Required fields validation**: always check that all expected JSON keys are present in the AI response
- **Graceful degradation**: if the AI call fails, return a 503 — do not silently ignore errors
- **Model traceability**: always log and store which AI model was used (`ai_model_used` field)
- **Prompt versioning**: keep prompts in constants (not inline strings) — makes them easy to iterate
- **No PII in logs**: strip customer names/emails from log messages in production

---

## 11. Documentation Rules

- Every module starts with a module-level docstring (purpose, what it provides)
- Every public function/class has a docstring with: what it does, args, returns, raises
- Use inline comments only for non-obvious logic — not to restate what the code already says
- README must include: what the project does, quick start, API endpoints, example request/response
- Architecture decisions go in `docs/TECHNICAL_ARCHITECTURE.md`

---

## 12. Git Conventions

- Branch naming: `feature/<name>`, `fix/<name>`, `docs/<name>`
- Commit messages: `type: short description`
  - Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`
  - Examples: `feat: add ticket classification endpoint`, `fix: handle AI JSON parse error`
- Never commit `.env`, `*.db`, `__pycache__`, or generated files
- Each commit should leave the codebase in a working state

---

## 13. Quick Reference Checklist

Before opening a PR or pushing to main:

- [ ] Code is formatted (`black`, `isort`)
- [ ] All tests pass (`pytest`)
- [ ] No hardcoded secrets
- [ ] `.env` is not committed
- [ ] New endpoints have `response_model`, `summary`, `description`
- [ ] New functions have docstrings
- [ ] Error handling covers: API failure, DB failure, not-found
- [ ] `.env.example` updated if new env vars were added

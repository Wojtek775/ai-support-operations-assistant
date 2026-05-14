"""
tests/conftest.py — Pytest configuration, shared fixtures, and test database setup.

IMPORTANT: This file is loaded by pytest BEFORE any test module is imported.
That is why env vars must be set here, at the top, before any app imports.

Two root causes of test failures this file fixes:

1. OPENROUTER_API_KEY is required by Settings (no default).
   → Set it here before app.config is imported.

2. SQLite :memory: creates a NEW empty database per connection.
   → Use StaticPool so all sessions share the same single connection
     and therefore the same tables.
"""

import os

# ---------------------------------------------------------------------------
# Set env vars FIRST — before any app module is imported.
# These are never sent to real APIs; the AI function is mocked in tests.
# Using os.environ[] (not setdefault) to guarantee override.
# ---------------------------------------------------------------------------
os.environ["OPENROUTER_API_KEY"] = "test-key-not-real"
os.environ["OPENROUTER_MODEL"] = "test-model"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["DEBUG"] = "false"          # guard against outer .env (e.g. c:\kurs\.env) having DEBUG=release

# ---------------------------------------------------------------------------
# Now it is safe to import app modules
# ---------------------------------------------------------------------------
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from app.main import app
from app.database import get_db

# Import Base AND TicketRecord to ensure the model is registered in
# Base.metadata BEFORE create_all() is called.
from app.models.db_models import Base, TicketRecord  # noqa: F401


# ---------------------------------------------------------------------------
# Test database engine
#
# StaticPool is the key: it forces SQLAlchemy to reuse the same underlying
# connection for every session. Without it, each session.connect() creates
# a fresh :memory: database and the tables created by create_all() are gone.
# ---------------------------------------------------------------------------
TEST_ENGINE = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=TEST_ENGINE,
)


def override_get_db():
    """
    FastAPI dependency override for tests.
    All requests during tests use this session (bound to TEST_ENGINE).
    """
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """
    Create all ORM tables in the in-memory test database once per session.
    Runs automatically before any test.
    Tears down after the entire test session completes.
    """
    Base.metadata.create_all(bind=TEST_ENGINE)
    yield
    Base.metadata.drop_all(bind=TEST_ENGINE)


@pytest.fixture()
def client(setup_test_database):
    """
    Provides a FastAPI TestClient for each test.

    - Overrides get_db so all endpoints use the in-memory test database.
    - Uses 'with TestClient(app)' to trigger lifespan (startup/shutdown).
    - Clears dependency_overrides after the test to avoid bleed-through.
    """
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

"""
database.py — SQLAlchemy engine setup and session management.

Provides:
  - engine: SQLAlchemy engine connected to the configured database
  - SessionLocal: session factory for creating DB sessions
  - Base: imported from db_models (single source of truth for ORM metadata)
  - get_db(): FastAPI dependency that yields a session and closes it safely

Migration path: change DATABASE_URL in .env to a PostgreSQL URL.
No other changes needed — SQLAlchemy handles the rest.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import Generator

from app.config import settings
from app.models.db_models import Base  # noqa: F401  — re-exported for main.py
from app.utils.logger import logger


# ---------------------------------------------------------------------------
# Engine setup
# ---------------------------------------------------------------------------

# For SQLite: disable same-thread check (required for FastAPI's async context)
# For PostgreSQL and others: no extra connect args needed
connect_args = {}
if settings.database_url.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

    # Ensure the data directory exists before SQLite tries to create the file
    db_path = settings.database_url.replace("sqlite:///", "").replace("sqlite://", "")
    db_dir = os.path.dirname(db_path)
    if db_dir and db_dir != ".":
        os.makedirs(db_dir, exist_ok=True)
        logger.debug(f"Ensured database directory exists: {db_dir}")

engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    # echo=settings.debug,  # Uncomment to log all SQL queries during development
)

logger.info(f"Database engine created | url={settings.database_url}")


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


# ---------------------------------------------------------------------------
# FastAPI dependency: get_db
# ---------------------------------------------------------------------------

def get_db() -> Generator:
    """
    FastAPI dependency that provides a database session per request.

    Usage in a router:
        from app.database import get_db

        @router.get("/example")
        def example(db: Session = Depends(get_db)):
            ...

    The session is always closed after the request, even if an exception occurs.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

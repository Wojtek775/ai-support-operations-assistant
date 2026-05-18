"""
main.py — FastAPI application entry point.

Creates the FastAPI app, registers all routers, configures middleware,
and initializes the database on startup.

Run with:
    cd backend
    uvicorn app.main:app --reload --port 8000

Interactive API docs available at:
    http://localhost:8000/docs
    http://localhost:8000/redoc
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import engine, Base
from app.routers import health, tickets, metrics
from app.utils.logger import logger


# ---------------------------------------------------------------------------
# Lifespan — runs on startup and shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    On startup:
      - Creates all database tables if they don't exist yet.
        (Uses SQLAlchemy metadata from all imported ORM models.)

    On shutdown:
      - Placeholder for cleanup (close connections, flush logs, etc.)
    """
    # STARTUP
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")

    try:
        # Create tables defined in ORM models (safe to call multiple times)
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database tables: {e}")
        raise

    yield  # Application is running

    # SHUTDOWN
    logger.info(f"Shutting down {settings.app_name}")


# ---------------------------------------------------------------------------
# FastAPI app instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "AI-powered operational workflow system that receives customer support messages "
        "and runs them through a full AI pipeline: classify → prioritize → summarize → "
        "sentiment → suggest response → recommend action."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

# CORS — open for development; restrict origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Tighten this in production: ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Router registration
# ---------------------------------------------------------------------------

app.include_router(health.router)
app.include_router(tickets.router)
app.include_router(metrics.router)


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------

@app.get("/", tags=["Root"], include_in_schema=False)
async def root():
    """Redirect hint — API docs are at /docs."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
    }

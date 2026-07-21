"""
config.py — Application configuration management.

Uses Pydantic BaseSettings to load values from .env file.
All settings are accessible via the `settings` singleton imported elsewhere.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """
    Central configuration for the application.
    Values are loaded from the .env file in the backend directory.
    """

    # Application metadata
    app_name: str = "AI Support & Operations Assistant"
    app_version: str = "0.1.0"
    debug: bool = False

    # OpenRouter AI settings
    openrouter_api_key: str
    openrouter_model: str = "openai/gpt-4o-mini"
    app_url: str = "http://localhost:8000"
    ai_request_timeout: int = 30

    # Database — swap to PostgreSQL URL for production
    database_url: str = "sqlite:///./app/data/tickets.db"

    # Workflow settings
    workflow_confidence_threshold: float = 0.60

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",          # silently ignore unknown env vars from the OS / other .env files
        env_file_override=False, # do NOT let .env file override values already in os.environ
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    Use this function throughout the app to avoid re-reading .env on every call.
    """
    return Settings()


# Convenience singleton — import this directly:
# from app.config import settings
settings = get_settings()

"""
logger.py — Centralized logging configuration.

Import `logger` from this module in any service or router:
    from app.utils.logger import logger
"""

import logging
import sys


def setup_logger(name: str = "ai_support_assistant") -> logging.Logger:
    """
    Creates and configures a logger with a consistent format.
    Logs go to stdout so they appear in docker logs / cloud log aggregators.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    # Format: timestamp | level | module | message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# Module-level logger — import and use directly
logger = setup_logger()

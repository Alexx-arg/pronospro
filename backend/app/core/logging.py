"""Logging utilities built on top of ``loguru``.

Minimal setup for Phase 2: configure a sink on stdout with JSON-friendly
serialized output and bind a request id context var in later phases. Other
services (providers, GLM, etc.) will enrich with additional context once
implemented.
"""

from __future__ import annotations

import sys

from loguru import logger

from app.config import get_settings

_CONFIGURED = False


def configure_logging() -> None:
    """Configure the global loguru logger from application settings.

    Idempotent: subsequent calls are no-ops so that importing the package
    multiple times (e.g. by tests) does not duplicate sinks.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    logger.remove()
    logger.configure(
        extra={"request_id": None, "service": "football-backend"},
    )
    logger.add(
        sys.stdout,
        level=settings.log_level,
        serialize=False,
        backtrace=True,
        diagnose=settings.env != "production",
        enqueue=False,
    )
    _CONFIGURED = True


def get_logger() -> "loguru.Logger":  # type: ignore[name-defined]
    """Return the configured loguru logger."""
    configure_logging()
    return logger

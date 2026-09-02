"""Async SQLAlchemy engine, sessionmaker and FastAPI dependency.

Only the persistence fundamentals live here. Services / repositories obtain
an :class:`AsyncSession` either through the FastAPI dependency
:func:`get_session` or by calling :func:`session_factory` directly in
background tasks / scheduled jobs.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings


def create_engine() -> AsyncEngine:
    """Build the async engine from the configured settings.

    The pool options are kept conservative (suitable for a small service):
    later phases can swap to ``AsyncAdaptedQueuePool`` configuration tweaks
    via ``DB_POOL_SIZE`` / ``DB_MAX_OVERFLOW`` env vars.
    """
    settings = get_settings()
    return create_async_engine(
        settings.async_database_url(),
        echo=settings.db_echo,
        pool_pre_ping=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
        future=True,
    )


# A module-level engine / factory are convenient for the FastAPI dependency,
# background tasks and tests. They are created lazily so that importing the
# package does not require a running database (important for tooling such as
# ruff/mypy where DATABASE_URL might be unset).
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Return the lazily-initialised singleton async engine."""
    global _engine
    if _engine is None:
        _engine = create_engine()
    return _engine


def session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the lazy singleton ``async_sessionmaker``."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )
    return _session_factory


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency that yields a transactional ``AsyncSession``.

    Commits on success, rolls back on exception, always closes.  Repositories
    never commit on their own to keep transaction ownership in the caller
    (or in this generator).
    """
    factory = session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Context manager equivalent of :func:`get_session`.

    Use from background tasks / scheduled jobs where FastAPI's dependency
    injection is not available.

    Usage::

        async with session_scope() as session:
            repo = PredictionRepository(session)
            await repo.insert(prediction)
    """
    factory = session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def dispose_engine() -> None:
    """Dispose the global engine (used in tests / app shutdown)."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None

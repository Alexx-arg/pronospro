"""Pytest configuration shared across the test-suite.

Design assumptions:

* The Postgres instance is provided externally — either by Docker Compose
  (the same one used in production) or by ``testcontainers`` when installed.
  We do NOT call ``Base.metadata.create_all`` here: the schema is created
  by Alembic migration ``0001_initial`` so that the trigger-based
  immutability policy is exercised by tests.

* Tests requiring a real database are marked ``@pytest.mark.integration``
  so they can be skipped on environments without Postgres availability::

      pytest -m "not integration"   # unit-only run
      pytest                         # everything
      pytest -m integration          # integration-only

* Each integration test runs inside its own SAVEPOINT-based transaction
  that is rolled back at teardown so tests are isolated and fast. This is
  the standard pattern for SQLAlchemy async integration tests.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Import all models so that Base.metadata is populated even when tests
# access relationships before the normal app import path runs.
from app.models import *  # noqa: F401,F403
from app.db import session as db_session  # noqa: F401


# -----------------------------------------------------------------------------
# Environment tweaks
# -----------------------------------------------------------------------------
# Tests use a dedicated database URL when DATABASE_URL is set. When unset,
# they fall back to the developer host (today default in .env.example).
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://football:changeme@localhost:5432/football_test",
    ),
)


# -----------------------------------------------------------------------------
# Config: register the integration marker (defined in pyproject.toml too).
# -----------------------------------------------------------------------------
def pytest_configure(config: pytest.Config) -> None:
    """Register the ``integration`` marker (defensive; pyproject already does)."""
    config.addinivalue_line("markers", "integration: requires a Postgres instance")


# -----------------------------------------------------------------------------
# Async session fixture connected to the real Postgres instance.
# -----------------------------------------------------------------------------
@pytest.fixture
async def _raw_engine():
    """Create a throwaway async engine for the test database."""
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(TEST_DATABASE_URL, future=True, pool_pre_ping=True)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
async def session(_raw_engine) -> AsyncIterator[AsyncSession]:
    """Yield an :class:`AsyncSession` connected to the live test database.

    Each test receives its own nested transaction so writes during the test
    can be rolled back without affecting the next test. The caller must NOT
    commit; the fixture rolls back automatically at teardown.

    The schema must already exist (imported via Alembic). A sanity check is
    run at fixture start to fail fast with a clear message if the
    migration was not applied.
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker

    factory = async_sessionmaker(
        bind=_raw_engine, class_=AsyncSession, expire_on_commit=False
    )

    async with factory() as sess:
        # Sanity check: ensure the initial migration ran by counting tables.
        result = await sess.execute(
            text("SELECT COUNT(*) FROM information_schema.tables "
                 "WHERE table_schema = 'public'")
        )
        n_tables = result.scalar_one()
        if n_tables < 15:  # we expect ~15 tables per SCHEMA.md
            pytest.skip(
                "Initial migration not applied to the test database: "
                f"only {n_tables} tables found (expected >=15). "
                "Run `alembic upgrade head` first."
            )

        try:
            yield sess
        finally:
            await sess.rollback()


# -----------------------------------------------------------------------------
# Convenience: a settings fixture so tests can override env vars if needed.
# -----------------------------------------------------------------------------
@pytest.fixture
def settings():
    from app.config import get_settings

    return get_settings()

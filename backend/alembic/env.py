"""Alembic environment.

Runs migrations **asynchronously** against the same database configured by
``app.config.Settings.async_database_url``. Uses
``run_sync`` to bridge between Alembic's sync API and the async engine.

Autogenerate is supported: ``target_metadata`` points at
``app.models`` (which imports every ORM file and populates ``Base.metadata``).
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.config import get_settings
from app.db.base import Base
from app.models import *  # noqa: F401,F403  # ensure all models are imported


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Inject the async DB URL into the config so Alembic off-line generators can
# see it as well.
config.set_main_option("sqlalchemy.url", get_settings().async_database_url())

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations without a DB connection (emit SQL to stdout).

    Uses a sync URL placeholder. In practice we always run online below.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Configure the context with a sync connection and run migrations."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        # Naming convention is already set on MetaData; still pass it so that
        # Alembic generates the same constraint names for new objects.
        render_as_batch=False,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations against the async engine."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """Entry point: dispatch to the async runner."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

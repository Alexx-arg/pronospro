"""Job: sync upcoming fixtures.

Scheduled to run every ``SYNC_UPCOMING_INTERVAL_MINUTES`` minutes. Pulls
fixtures whose kickoff falls within ``SYNC_UPCOMING_DAYS`` days for each
configured league and upserts them keyed by ``external_id``.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.providers.base import DataProvider
from app.services.sync import sync_upcoming_fixtures
from app.tasks._runner import JobResult, run_job

JOB_NAME = "sync_upcoming"


async def run_sync_upcoming(
    *,
    settings: Settings | None = None,
    provider: DataProvider | None = None,
    session: Any = None,
    force: bool = False,
    league_ids: list[int] | None = None,
    days: int | None = None,
) -> JobResult:
    """Run the upcoming-fixtures sync job.

    See :mod:`app.tasks._runner` for the meaning of the common kwargs.
    """
    return await run_job(
        job_name=JOB_NAME,
        enabled_flag_fn=lambda s: s.sync_upcoming_enabled,
        force=force,
        sync_fn=sync_upcoming_fixtures,
        settings=settings,
        provider=provider,
        session=session,
        sync_kwargs={"league_ids": league_ids, "days": days},
    )


__all__ = ["JOB_NAME", "run_sync_upcoming"]

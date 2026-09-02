"""Job: sync finished fixtures.

Runs every ``SYNC_FINISHED_INTERVAL_MINUTES`` minutes. Back-fills scores
and ``status='finished'`` for fixtures that finished within
``SYNC_FINISHED_WINDOW_HOURS`` hours ago. The job is intentionally more
frequent than ``sync_upcoming`` because the finish event is time-sensitive
for the prediction outcomes phase.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.providers.base import DataProvider
from app.services.sync import sync_finished_fixtures
from app.tasks._runner import JobResult, run_job

JOB_NAME = "sync_finished"


async def run_sync_finished(
    *,
    settings: Settings | None = None,
    provider: DataProvider | None = None,
    session: Any = None,
    force: bool = False,
    league_ids: list[int] | None = None,
    window_hours: int | None = None,
) -> JobResult:
    """Run the finished-fixtures sync job."""
    return await run_job(
        job_name=JOB_NAME,
        enabled_flag_fn=lambda s: s.sync_finished_enabled,
        force=force,
        sync_fn=sync_finished_fixtures,
        settings=settings,
        provider=provider,
        session=session,
        sync_kwargs={"league_ids": league_ids, "window_hours": window_hours},
    )


__all__ = ["JOB_NAME", "run_sync_finished"]

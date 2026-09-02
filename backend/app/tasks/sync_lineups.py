"""Job: sync lineups.

Looks up the upcoming fixtures (scheduled + in_play) for the configured
leagues and asks the provider for each one's lineup. The provider
returns an empty list when the lineup isn't published yet — that's
counted as ``skipped``, not a failure. Default cadence: 2h.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.providers.base import DataProvider
from app.services.sync import sync_lineups
from app.tasks._runner import JobResult, run_job

JOB_NAME = "sync_lineups"


async def run_sync_lineups(
    *,
    settings: Settings | None = None,
    provider: DataProvider | None = None,
    session: Any = None,
    force: bool = False,
    fixture_external_ids: list[int] | None = None,
) -> JobResult:
    """Run the lineups sync job.

    When ``fixture_external_ids`` is ``None`` the sync service looks up
    upcoming fixtures itself (see :func:`app.services.sync.sync_lineups`).
    """
    return await run_job(
        job_name=JOB_NAME,
        enabled_flag_fn=lambda s: s.sync_lineups_enabled,
        force=force,
        sync_fn=sync_lineups,
        settings=settings,
        provider=provider,
        session=session,
        sync_kwargs={"fixture_external_ids": fixture_external_ids},
    )


__all__ = ["JOB_NAME", "run_sync_lineups"]

"""Job: sync team statistics.

Pulls aggregated team stats for one (team, league, season, as_of_date)
snapshot per team. Stamped with ``today`` so successive runs produce a
new row each day (slowly-changing dimension). Default cadence: 6h.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.providers.base import DataProvider
from app.services.sync import sync_team_statistics
from app.tasks._runner import JobResult, run_job

JOB_NAME = "sync_team_statistics"


async def run_sync_team_statistics(
    *,
    settings: Settings | None = None,
    provider: DataProvider | None = None,
    session: Any = None,
    force: bool = False,
    league_ids: list[int] | None = None,
) -> JobResult:
    """Run the team-statistics sync job."""
    return await run_job(
        job_name=JOB_NAME,
        enabled_flag_fn=lambda s: s.sync_team_statistics_enabled,
        force=force,
        sync_fn=sync_team_statistics,
        settings=settings,
        provider=provider,
        session=session,
        sync_kwargs={"league_ids": league_ids},
    )


__all__ = ["JOB_NAME", "run_sync_team_statistics"]

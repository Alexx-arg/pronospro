"""Job: sync teams.

Refreshes the team roster of each configured league for the current
season. Idempotent upsert keyed by ``external_id``. Runs at a low
cadence (default 12h) since teams rarely change mid-season.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.providers.base import DataProvider
from app.services.sync import sync_teams
from app.tasks._runner import JobResult, run_job

JOB_NAME = "sync_teams"


async def run_sync_teams(
    *,
    settings: Settings | None = None,
    provider: DataProvider | None = None,
    session: Any = None,
    force: bool = False,
    league_ids: list[int] | None = None,
) -> JobResult:
    """Run the teams sync job."""
    return await run_job(
        job_name=JOB_NAME,
        enabled_flag_fn=lambda s: s.sync_teams_enabled,
        force=force,
        sync_fn=sync_teams,
        settings=settings,
        provider=provider,
        session=session,
        sync_kwargs={"league_ids": league_ids},
    )


__all__ = ["JOB_NAME", "run_sync_teams"]

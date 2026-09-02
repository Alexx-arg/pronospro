"""Job: sync player statistics.

Most expensive job: iterates every team of every configured league, then
every player of every team, fetching the per-player statistics payload.
The HTTP client enforces the per-minute rate limit so the run stays
within the provider's plan. Default cadence: 6h.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.providers.base import DataProvider
from app.services.sync import sync_player_statistics
from app.tasks._runner import JobResult, run_job

JOB_NAME = "sync_player_statistics"


async def run_sync_player_statistics(
    *,
    settings: Settings | None = None,
    provider: DataProvider | None = None,
    session: Any = None,
    force: bool = False,
    league_ids: list[int] | None = None,
) -> JobResult:
    """Run the player-statistics sync job."""
    return await run_job(
        job_name=JOB_NAME,
        enabled_flag_fn=lambda s: s.sync_player_statistics_enabled,
        force=force,
        sync_fn=sync_player_statistics,
        settings=settings,
        provider=provider,
        session=session,
        sync_kwargs={"league_ids": league_ids},
    )


__all__ = ["JOB_NAME", "run_sync_player_statistics"]

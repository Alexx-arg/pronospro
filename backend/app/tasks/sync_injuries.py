"""Job: sync injuries.

Pulls current + historical injuries for each configured league's current
season. The repository's ``injuries.player_id`` / ``team_id`` FK columns
expect internal ids; the sync service resolves external → internal
before upserting. Default cadence: 3h.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.providers.base import DataProvider
from app.services.sync import sync_injuries
from app.tasks._runner import JobResult, run_job

JOB_NAME = "sync_injuries"


async def run_sync_injuries(
    *,
    settings: Settings | None = None,
    provider: DataProvider | None = None,
    session: Any = None,
    force: bool = False,
    league_ids: list[int] | None = None,
) -> JobResult:
    """Run the injuries sync job."""
    return await run_job(
        job_name=JOB_NAME,
        enabled_flag_fn=lambda s: s.sync_injuries_enabled,
        force=force,
        sync_fn=sync_injuries,
        settings=settings,
        provider=provider,
        session=session,
        sync_kwargs={"league_ids": league_ids},
    )


__all__ = ["JOB_NAME", "run_sync_injuries"]

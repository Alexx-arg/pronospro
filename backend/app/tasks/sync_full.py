"""Job: full bootstrap sync (ingest competitions, teams, fixtures, stats).

Not registered in the scheduler by default — it's the one-shot ingest fired
on demand via the admin endpoint (``POST /api/v1/admin/sync``) or CLI.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.providers.base import DataProvider
from app.services.sync.full import run_full_bootstrap, run_full_bootstrap_metrics
from app.tasks._runner import JobResult, run_job

JOB_NAME = "sync_full"


async def run_sync_full(
    *,
    settings: Settings | None = None,
    provider: DataProvider | None = None,
    session: Any = None,
    force: bool = False,
    league_ids: list[int] | None = None,
    days: int | None = None,
) -> JobResult:
    """Run the full bootstrap ingest.

    ``force`` is honoured but the bootstrap is not gated by an ENABLED flag:
    it's always runnable on demand (the flag is a no-op gate here).
    """
    return await run_job(
        job_name=JOB_NAME,
        enabled_flag_fn=lambda s: True,
        force=force,
        sync_fn=_bootstrap_sync,
        settings=settings,
        provider=provider,
        session=session,
        sync_kwargs={"league_ids": league_ids, "days": days},
    )


async def _bootstrap_sync(
    *,
    provider: DataProvider,
    session: Any,
    settings: Settings,
    league_ids: list[int] | None = None,
    days: int | None = None,
):
    """Adapter that runs the bootstrap and folds its report into SyncMetrics."""
    report = await run_full_bootstrap(
        provider=provider,
        session=session,
        settings=settings,
        league_ids=league_ids,
        days=days,
    )
    return await run_full_bootstrap_metrics(report)


__all__ = ["JOB_NAME", "run_sync_full"]
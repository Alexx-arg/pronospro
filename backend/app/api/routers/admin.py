"""Admin endpoints (protected by ``X-Admin-Key``).

Operational triggers for the sync jobs. Not exposed to the mobile client.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.api.dependencies.security import require_admin_key_dep
from app.config import Settings, get_settings
from app.tasks import run_sync_full, run_sync_team_statistics, run_sync_teams, run_sync_upcoming

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin_key_dep)])


class SyncResult(BaseModel):
    job: str
    skipped: bool
    metrics: dict[str, Any] | None


@router.post("/sync", response_model=SyncResult)
async def trigger_full_sync(
    league_ids: list[int] | None = Query(None),
    days: int | None = Query(None, ge=1, le=30),
) -> SyncResult:
    """Run the full bootstrap ingest (competitions → teams → fixtures → stats)."""
    settings: Settings = get_settings()
    result = await run_sync_full(settings=settings, force=True, league_ids=league_ids, days=days)
    return SyncResult(
        job=result.job,
        skipped=result.skipped,
        metrics=result.metrics.as_dict() if result.metrics else None,
    )


@router.post("/sync/upcoming", response_model=SyncResult)
async def trigger_upcoming(
    league_ids: list[int] | None = Query(None),
    days: int | None = Query(None, ge=1, le=30),
) -> SyncResult:
    """Run only the upcoming-fixtures sync."""
    settings: Settings = get_settings()
    result = await run_sync_upcoming(settings=settings, force=True, league_ids=league_ids, days=days)
    return SyncResult(
        job=result.job,
        skipped=result.skipped,
        metrics=result.metrics.as_dict() if result.metrics else None,
    )


@router.post("/sync/teams", response_model=SyncResult)
async def trigger_teams(league_ids: list[int] | None = Query(None)) -> SyncResult:
    settings: Settings = get_settings()
    result = await run_sync_teams(settings=settings, force=True, league_ids=league_ids)
    return SyncResult(
        job=result.job,
        skipped=result.skipped,
        metrics=result.metrics.as_dict() if result.metrics else None,
    )


@router.post("/sync/team-statistics", response_model=SyncResult)
async def trigger_team_statistics(league_ids: list[int] | None = Query(None)) -> SyncResult:
    settings: Settings = get_settings()
    result = await run_sync_team_statistics(settings=settings, force=True, league_ids=league_ids)
    return SyncResult(
        job=result.job,
        skipped=result.skipped,
        metrics=result.metrics.as_dict() if result.metrics else None,
    )
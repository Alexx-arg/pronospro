"""Admin endpoints (protected by ``X-Admin-Key``).

Devuelve el error en el cuerpo (status 200 con ``error``) para facilitar
diagnóstico desde Render free (sin acceso a shell/logs).
"""

from __future__ import annotations

import traceback
from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.api.dependencies.security import require_admin_key_dep
from app.config import get_settings
from app.tasks import run_sync_full, run_sync_team_statistics, run_sync_teams, run_sync_upcoming

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin_key_dep)])


class SyncResult(BaseModel):
    job: str
    skipped: bool
    metrics: dict[str, Any] | None
    error: str | None = None


async def _safe_run(fn, **kwargs) -> SyncResult:
    """Run a job and return a SyncResult that never raises (error embedded)."""
    try:
        result = await fn(settings=get_settings(), force=True, **kwargs)
        return SyncResult(
            job=result.job,
            skipped=result.skipped,
            metrics=result.metrics.as_dict() if result.metrics else None,
        )
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc(limit=10)
        return SyncResult(
            job=getattr(fn, "__name__", "unknown"),
            skipped=False,
            metrics=None,
            error=f"{exc!r}\n{tb[-2000:]}",
        )


@router.post("/sync", response_model=SyncResult)
async def trigger_full_sync(
    league_ids: list[int] | None = Query(None),
    days: int | None = Query(None, ge=1, le=30),
) -> JSONResponse:
    """Run the full bootstrap ingest (competitions → teams → fixtures → stats)."""
    out = await _safe_run(run_sync_full, league_ids=league_ids, days=days)
    status_code = 500 if out.error else 200
    return JSONResponse(status_code=status_code, content=out.model_dump())


@router.post("/sync/upcoming", response_model=SyncResult)
async def trigger_upcoming(
    league_ids: list[int] | None = Query(None),
    days: int | None = Query(None, ge=1, le=30),
) -> JSONResponse:
    out = await _safe_run(run_sync_upcoming, league_ids=league_ids, days=days)
    status_code = 500 if out.error else 200
    return JSONResponse(status_code=status_code, content=out.model_dump())


@router.post("/sync/teams", response_model=SyncResult)
async def trigger_teams(league_ids: list[int] | None = Query(None)) -> JSONResponse:
    out = await _safe_run(run_sync_teams, league_ids=league_ids)
    status_code = 500 if out.error else 200
    return JSONResponse(status_code=status_code, content=out.model_dump())


@router.post("/sync/team-statistics", response_model=SyncResult)
async def trigger_team_statistics(league_ids: list[int] | None = Query(None)) -> JSONResponse:
    out = await _safe_run(run_sync_team_statistics, league_ids=league_ids)
    status_code = 500 if out.error else 200
    return JSONResponse(status_code=status_code, content=out.model_dump())